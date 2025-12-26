from src.utils import apply_chat_template
from dataclasses import dataclass

import json


@dataclass
class DataLoader:
    conversations_path: str  ## Path to the conversations JSONL file
    tokenizer: object
    vision_processor: object
    batch_size: int
    max_length: int

    def __post_init__(self):
        with open(self.conversations_path, "r") as f:
            self.conversation = [json.loads(line.strip()) for line in f]

        self.start = 0

    def next_batch(self):
        if self.start + self.batch_size >= len(self.conversation):
            batch = self.conversation[self.start :]
            self.start = 0
        else:
            batch = self.conversation[self.start : self.start + self.batch_size]
            self.start += self.batch_size
        return self.collate_fn(batch)

    def collate_fn(self, batch):
        """
        Collate function to process a batch of conversations.
        Args:
            batch (list): A list of conversations.
        Returns:
            dict: A dictionary containing tokenized inputs and processed images.
        """

        input_prompts = []
        image_paths = []
        prefix_lengths = []

        for conversation in batch:
            prompt, imgs = apply_chat_template(
                self.tokenizer, conversation, add_generation_prompt=False
            )
            input_prompts.append(prompt)
            image_paths.extend(imgs)

            prompt, _ = apply_chat_template(
                self.tokenizer, conversation[:-1], add_generation_prompt=True
            )
            prefix_lengths.append(len(self.tokenizer(prompt).input_ids))

        # Tokenize the prompts
        inputs = self.tokenizer(
            input_prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        ## Create labels for language modeling
        targets = inputs.input_ids.clone()
        # Mask out special tokens
        special_token_ids = [
            self.tokenizer.pad_token_id,
            self.tokenizer.image_token_id,
            self.tokenizer.boi_token_id,
            self.tokenizer.eoi_token_id,
        ]
        for token_id in special_token_ids:
            targets[targets == token_id] = -100  # Mask out special tokens

        for idx, prefix_len in enumerate(prefix_lengths):
            end = min(prefix_len, targets.size(1))
            targets[idx, :end] = -100  # Mask out the prefix tokens

        # Process images if any
        pixel_values = None
        if image_paths:
            pixel_values = self.vision_processor(
                images=image_paths, return_tensors="pt"
            ).pixel_values

        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "pixel_values": pixel_values,
            "targets": targets,
        }
