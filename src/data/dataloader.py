"""DataLoader for VLM training.

This module provides a custom DataLoader class for loading and batching
conversation data for vision-language model training.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from src.utils import apply_chat_template


class VLMDataset(Dataset):
    """Dataset for processing conversation data."""

    def __init__(self, conversations_path: str):
        with open(conversations_path, "r") as f:
            self.conversations = [
                json.loads(line.strip()) for line in f if line.strip()
            ]

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        return self.conversations[idx]


class DataLoader:
    """DataLoader for processing conversation data.

    Attributes:
        conversations_path (str): Path to the conversations JSONL file.
        tokenizer (object): Tokenizer for text processing.
        vision_processor (object): Processor for image preprocessing.
        batch_size (int): Number of conversations per batch.
        max_length (int): Maximum sequence length.
        num_workers (int): Number of worker processes for data loading.
    """

    def __init__(
        self,
        conversations_path: str,
        tokenizer: object,
        vision_processor: object,
        batch_size: int,
        max_length: int,
        num_workers: int = 4,
    ):
        self.tokenizer = tokenizer
        self.vision_processor = vision_processor
        self.batch_size = batch_size
        self.max_length = max_length

        self.dataset = VLMDataset(conversations_path)
        self.dataloader = TorchDataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=True,
        )
        self.iterator = iter(self.dataloader)
        self.num_batches = len(self.dataloader)

    def next_batch(self) -> dict[str, torch.Tensor | None]:
        """
        Get the next batch of conversations.

        Returns:
            dict: Batch dictionary containing tokenized inputs and processed images.
        """
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)

    def collate_fn(self, batch) -> dict[str, torch.Tensor | int | None]:
        """
        Collate function to process a batch of conversations.

        This function tokenizes conversations, pads sequences to equal length,
        and processes images for model input.

        Args:
            batch (list): A list of conversations.

        Returns:
            dict: A dictionary containing:
                - input_ids: Padded tensor of token IDs.
                - attention_mask: Padded attention mask tensor.
                - pixel_values: Processed image tensors.
                - targets: Padded target IDs for training.
                - num_tokens: Number of non-padded tokens in the batch.
        """

        input_ids = []
        attention_mask = []
        pixel_values = []
        targets = []

        for conversation in batch:
            assert isinstance(conversation, list), "Each conversation must be a list."
            result = apply_chat_template(
                self.tokenizer, conversation, add_generation_prompt=False
            )
            input_ids.append(result["input_ids"])
            attention_mask.append(result["attention_mask"])
            pixel_values.append(result["pixel_values"])
            targets.append(result["targets"])

        max_len = max(len(ids) for ids in input_ids)
        padded_input_ids = []
        padded_attention_mask = []
        padded_targets = []

        for ids, mask, tgt in zip(input_ids, attention_mask, targets):
            pad_len = max_len - ids.size(0)
            padded_input_ids.append(
                torch.cat([ids, torch.full((pad_len,), self.tokenizer.pad_token_id)])
            )
            padded_attention_mask.append(
                torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
            )
            padded_targets.append(torch.cat([tgt, torch.full((pad_len,), -100)]))

        input_ids_tensor = torch.stack(padded_input_ids)[:, : self.max_length]

        return {
            "input_ids": input_ids_tensor,
            "attention_mask": torch.stack(padded_attention_mask)[:, : self.max_length],
            "pixel_values": torch.stack(pixel_values)
            if len(pixel_values) > 0
            else None,
            "targets": torch.stack(padded_targets)[:, : self.max_length],
            "num_tokens": (input_ids_tensor != self.tokenizer.pad_token_id)
            .sum()
            .item(),
        }
