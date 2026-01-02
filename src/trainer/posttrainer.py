import time
import wandb
import torch

from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from src.logger import logger
from src.trainer.base import BaseTrainer


class PostTrainer(BaseTrainer):
    def __post_init__(self):
        wandb.init(project="indic-vlm-posttraining")

        self.model.to(self.device)

        self.model.train()
        if self.train_config.freeze_vision_model:
            for param in self.model.vision_model.parameters():
                param.requires_grad = False
            logger.info("Vision model frozen.")

        if self.train_config.compile:
            self.model = torch.compile(self.model)
            logger.info("Model compiled with torch.compile().")

        self.total_steps = (
            self.train_config.num_epochs * self.train_dataloader.num_batches
        ) // self.train_config.gradient_accumulation_steps + 1

        self.scaler = GradScaler(enabled=(self.device == "cuda"))

        if self.optimizer is None:
            self.setup_optimizer()

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        logger.info(f"Total parameters: {total_params / 1e6:.2f}M")
        logger.info(f"Trainable parameters: {trainable_params / 1e9:.3f}B")

    def setup_optimizer(self):
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = AdamW(
            trainable_params,
            lr=self.train_config.learning_rate,
        )
        return None

    def save_checkpoint(self, path: str):
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model checkpoint saved at {path}")
        return None

    def train(self):
        logger.info("Starting post-training...")

        for _ in range(self.total_steps):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            train_loss = 0.0
            total_tokens = 0

            start = time.time()
            for _ in range(self.train_config.gradient_accumulation_steps):
                batch = self.train_dataloader.next_batch()
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    _, loss = self.model(**batch)

                loss /= self.train_config.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
                train_loss += loss.item()
                total_tokens += batch.get("num_tokens", 0)

                del batch

            end = time.time()
            throughput = total_tokens / (end - start)

            # Unscale gradients before clipping
            self.scaler.unscale_(self.optimizer)
            norm = clip_grad_norm_(
                self.model.parameters(), self.train_config.max_grad_norm
            )

            # Scaler step and update
            self.scaler.step(self.optimizer)
            self.scaler.update()

            val_loss = 0.0
            for _ in range(self.train_config.gradient_accumulation_steps):
                batch = self.val_dataloader.next_batch()
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                with torch.no_grad():
                    with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                        _, loss = self.model(**batch)
                val_loss += loss.item()
                del batch
            val_loss /= self.train_config.gradient_accumulation_steps

            wandb.log(
                {
                    "Train Loss": train_loss,
                    "Validation Loss": val_loss,
                    "Gradient Norm": norm,
                    "Throughput (tokens/sec)": throughput,
                }
            )

        self.model.save(self.train_config.checkpoint_dir)

        return None
