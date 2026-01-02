"""Pre-trainer implementation for VLM models.

This module implements the PreTrainer class which handles the complete
training loop for vision-language models, including optimizer setup,
learning rate scheduling, and model checkpointing.
"""

import os
import time
import math
import wandb
import torch
from dataclasses import dataclass
from torch.optim import Muon, AdamW
from torch.nn.utils import clip_grad_norm_
from torch.amp import GradScaler


from src.trainer.base import BaseTrainer
from src.utils import apply_chat_template
from src.logger import logger


@dataclass
class PreTrainer(BaseTrainer):
    """
    Pre-trainer for Vision-Language Models.

    This class implements the training loop with features like gradient accumulation,
    learning rate scheduling, model freezing, and periodic inference.
    """

    def __post_init__(self):
        """
        Initialize the pre-trainer with model setup and configuration.

        Sets up device, freezes models if specified, compiles model if needed,
        and initializes optimizer.
        """
        # Initialize wandb
        wandb.init(
            project="indic-vlm-pretraining",
        )

        # Move model to device = "mps/cuda/cpu"
        self.model.to(self.device)
        logger.info(f"Model moved to device: {self.device}")

        ## freeze the vision model if specified
        if self.train_config.freeze_vision_model:
            for param in self.model.vision_model.parameters():
                param.requires_grad = False
            logger.info("Vision model frozen.")

        # Freeze the language model if specified
        if self.train_config.freeze_language_model:
            for param in self.model.language_model.parameters():
                param.requires_grad = False
            logger.info("Language model frozen.")

        # Compile the model if specified
        if self.train_config.compile:
            self.model = torch.compile(self.model)
            logger.info("Model compiled with torch.compile.")

        # Use baseline optimizer if custom optimizer is not passed
        if self.optimizer is None:
            self.setup_optimizer()
            logger.info(
                "Baseline optimizer set up, since no custom optimizer was passed."
            )

        # Basic configuration setup
        self.total_steps = (
            self.train_config.num_epochs * self.train_dataloader.num_batches
        ) // self.train_config.gradient_accumulation_steps + 1

        # Initialize GradScaler for mixed precision training
        self.scaler = GradScaler(enabled=(self.device == "cuda"))

        logger.info(
            f"Total number of parameters to train: {sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6:.2f} Million"
        )
        logger.info(
            f"Total number of parameters in the model: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f} Billion"
        )
        logger.info(f"Training for {self.total_steps} steps.")

    def setup_optimizer(self) -> None:
        """
        Set up the optimizer (Muon or AdamW) based on configuration.

        Returns:
            None
        """
        # Setup optimizer - Muon or AdamW whichever is specified
        optimizer = Muon if self.train_config.use_muon else AdamW

        ## find trainable parameters
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optimizer(
            trainable_params, lr=self.train_config.learning_rate, fused=True
        )
        return None

    def get_lr(self, step: int) -> float:
        """
        Calculate learning rate using cosine schedule with linear warmup.

        Args:
            step (int): Current training step.

        Returns:
            float: Learning rate for the current step.
        """
        # Cosine learning rate scheduler with linear warmup
        warmup_steps = int(0.1 * self.total_steps)
        min_lr = self.train_config.learning_rate * 0.01
        if step <= warmup_steps:
            return (
                min_lr
                + (self.train_config.learning_rate - min_lr) * step / warmup_steps
            )
        progress = (step - warmup_steps) / (self.total_steps - warmup_steps)
        return min_lr + 0.5 * (self.train_config.learning_rate - min_lr) * (
            1 + math.cos(math.pi * progress)
        )

    def save_projector(self):
        """
        Save the projector checkpoint to disk.

        Returns:
            None
        """
        # Save the projector checkpoint
        projector_state = self.model.projector.state_dict()
        ## create directory if not exists
        os.makedirs(
            os.path.dirname(self.train_config.projector_checkpoint_path), exist_ok=True
        )
        torch.save(projector_state, self.train_config.projector_checkpoint_path)
        logger.info(
            f"Projector checkpoint saved at {self.train_config.projector_checkpoint_path}"
        )
        return None

    def train(self):
        """
        Execute the main training loop.

        Performs gradient accumulation, gradient clipping, optimizer steps,
        learning rate scheduling, validation, and periodic inference.

        Returns:
            None
        """
        logger.info("Starting training loop...")
        for step in range(1, self.total_steps + 1):
            # Training step
            self.model.train()
            self.optimizer.zero_grad(
                set_to_none=True
            )  # More memory efficient than zero_grad()

            train_loss = 0.0
            total_tokens = 0
            start = time.time()

            for _ in range(self.train_config.gradient_accumulation_steps):
                batch = self.train_dataloader.next_batch()
                batch = {
                    k: v.to(self.device, non_blocking=True)
                    if isinstance(v, torch.Tensor)
                    else v
                    for k, v in batch.items()
                }

                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    _, loss = self.model(**batch)
                loss = loss / self.train_config.gradient_accumulation_steps

                # Use scaler for backward pass
                self.scaler.scale(loss).backward()
                train_loss += loss.item()
                total_tokens += batch.get("num_tokens", 0)

                del batch  # free memory

            end = time.time()
            dt = end - start
            throughput = total_tokens / dt if total_tokens > 0 else 0.0

            # Unscale gradients before clipping
            self.scaler.unscale_(self.optimizer)
            norm = clip_grad_norm_(
                self.model.parameters(), self.train_config.max_grad_norm
            )

            # Scaler step and update
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update learning rate
            lr = self.get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Evaluation step can be added here
            val_loss = None if self.val_dataloader is None else 0.0
            if self.val_dataloader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for _ in range(self.train_config.gradient_accumulation_steps):
                        batch = self.val_dataloader.next_batch()
                        batch = {
                            k: v.to(self.device, non_blocking=True)
                            if isinstance(v, torch.Tensor)
                            else v
                            for k, v in batch.items()
                        }
                        with torch.autocast(
                            device_type=self.device, dtype=torch.bfloat16
                        ):
                            _, loss = self.model(**batch)
                        loss /= self.train_config.gradient_accumulation_steps
                        val_loss += loss.item()

                        del batch  # free memory

            # Logging training and validation loss
            log_message = (
                f"Step [{step}/{self.total_steps}] - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {'N/A' if val_loss is None else f'{val_loss:.4f}'} - "
                f"LR: {lr:.6f} - "
                f"Grad Norm: {norm:.4f} - "
                f"Throughput: {throughput:.2f} tokens/sec - "
                f"dt: {dt:.2f} seconds"
            )
            logger.info(log_message)

            # Log to wandb
            wandb.log(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss if val_loss is not None else None,
                    "learning_rate": lr,
                    "grad_norm": norm,
                    "throughput": throughput,
                }
            )

            # Inference at every 100 steps can be added here
            if step % 100 == 0:
                self.model.eval()
                output = self.inference()
                joined = "\n---\n".join(output)
                logger.info(f"Inference output at step {step}: {joined}")
                wandb.log({"inference_output": joined})
                self.save_projector()

        self.save_projector()
        wandb.finish()
        return None

    @torch.inference_mode()
    def inference(self) -> str:
        """
        Perform inference on the example conversation.

        Returns:
            str: Generated text output from the model.
        """
        outputs = list()
        for conversation in self.example_conversation:
            result = apply_chat_template(
                self.tokenizer,
                conversation,
                add_generation_prompt=True,
            )

            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                generated_ids = self.model.generate(
                    input_ids=result["input_ids"].unsqueeze(0).to(self.device),
                    pixel_values=result["pixel_values"].to(self.device)
                    if result["pixel_values"] is not None
                    else None,
                    attention_mask=result["attention_mask"]
                    .unsqueeze(0)
                    .to(self.device),
                )

            output_text = self.tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            )
            outputs.append(output_text)

        return outputs
