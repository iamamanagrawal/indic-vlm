"""Post-processing module for conversation data translation.

This module provides functions to transform conversation formats and translate
text content using NLLB translation models.
"""

import json
import torch
from dataclasses import dataclass
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.utils import get_device


@dataclass
class TranslationConfig:
    """Configuration for the translation service."""

    model_path: str = "models/nllb-200-distilled-600M"
    src_lang: str = "eng_Latn"
    tgt_lang: str = "hin_Deva"
    batch_size: int = 1024
    max_length: int = 300


class Translator:
    """Translator class for batch translation using NLLB models."""

    def __init__(self, config: TranslationConfig | None = None) -> None:
        """
        Initialize the translator with model and tokenizer.

        Args:
            config: Translation configuration. Uses defaults if None.
        """
        self.config = config or TranslationConfig()
        self.device = get_device()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            src_lang=self.config.src_lang,
        )

    def translate(self, texts: list[str]) -> list[str]:
        """
        Translate a list of texts from source language to target language.

        Args:
            texts: List of texts to translate.

        Returns:
            List of translated texts.
        """
        translated_texts = []

        for idx in tqdm(range(0, len(texts), self.config.batch_size)):
            batch_texts = texts[idx : idx + self.config.batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
            ).to(self.device)

            with (
                torch.inference_mode(),
                torch.autocast(device_type=self.device, dtype=torch.bfloat16),
            ):
                outputs = self.model.generate(
                    **inputs,
                    num_beams=1,
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(
                        self.config.tgt_lang
                    ),
                    num_return_sequences=1,
                    max_length=self.config.max_length,
                    use_cache=True,
                    do_sample=True,
                    top_p=0.9,
                )

            decoded_texts = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
            )

            # Save results on jsonl for backup
            mapp = dict(zip(batch_texts, decoded_texts))
            with open("data/translation_map.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(mapp, ensure_ascii=False) + "\n")

            translated_texts.extend(decoded_texts)

        return translated_texts


def transform_conversation(item: dict) -> list[dict]:
    """
    Transform conversation format from old to new schema.

    Args:
        item: Dictionary containing 'conversations' and 'image' keys.

    Returns:
        List of message dictionaries with 'role', 'content', and optional 'image' keys.
    """
    conversation = []
    for conv in item["conversations"]:
        message = {
            "role": "user" if conv["from"] == "human" else "assistant",
            "content": conv["value"],
        }

        if "<image>" in message["content"]:
            message["content"] = message["content"].replace("<image>", "").strip()
            message["image"] = item["image"]

        conversation.append(message)
    return conversation


def process_conversations(input_path: str, output_path: str) -> None:
    """
    Process conversations from input file and save transformed data to output file.

    Args:
        input_path: Path to input JSON file with conversations.
        output_path: Path to output JSONL file for processed conversations.

    Returns:
        None
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    conversations = [transform_conversation(item) for item in data]

    contents = [message["content"] for conv in conversations for message in conv]

    translator = Translator()
    translated_contents = translator.translate(contents)

    for conv in conversations:
        for message in conv:
            message["content"] = translated_contents.pop(0)

    with open(output_path, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")


if __name__ == "__main__":
    input_file = "data/llava_v1_5_mix665k.json"
    output_file = "data/processed_conversations.jsonl"
    process_conversations(input_file, output_file)
