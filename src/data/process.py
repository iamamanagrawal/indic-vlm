"""Data processing module for translating captions and creating conversation data.

This module processes image captions by translating them to target languages
and creating conversation format datasets for VLM training.
"""

import argparse
import pandas as pd
import random
import json
from collections import deque

from src.data.translation import TranslationService
from src.utils import EN_USERS_QUESTIONS, HI_USERS_QUESTIONS
from tqdm import tqdm


def parse_args():
    """
    Parse command-line arguments for the translation service.

    Returns:
        argparse.Namespace: Parsed arguments containing input file path,
            language codes, batch size, and buffer settings.
    """
    parser = argparse.ArgumentParser(description="Translation Service")

    parser.add_argument(
        "--input_file",
        type=str,
        default="processed_data/processed_data.parquet",
        help="Path to the input file containing text to be translated.",
    )

    parser.add_argument(
        "--src_lang",
        type=str,
        default="eng_Latn",
        help="Source language code for translation.",
    )

    parser.add_argument(
        "--tgt_lang",
        type=str,
        default="hin_Deva",
        help="Target language code for translation.",
    )

    parser.add_argument(
        "--batch_size", type=int, default=160, help="Batch size for translation."
    )

    parser.add_argument(
        "--write_buffer_size",
        type=int,
        default=5,
        help="Number of batches to buffer before writing to disk.",
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """
    Main function to translate captions and create conversation datasets.

    Reads image captions from a parquet file, translates them to the target language,
    and creates conversation format data in both source and target languages.
    Outputs are written to a JSONL file with buffering for efficiency.

    Args:
        args: Command-line arguments containing input file, language codes,
            batch size, and write buffer size.

    Returns:
        None
    """
    df = pd.read_parquet(args.input_file)

    service = TranslationService(src_lang=args.src_lang, tgt_lang=args.tgt_lang)

    output_file = args.input_file.replace(".parquet", ".jsonl")

    # Use a buffer to batch disk writes, reducing I/O overhead
    write_buffer = deque(maxlen=args.write_buffer_size)

    with open(output_file, "a", encoding="utf-8") as f:
        total_batches = (len(df) + args.batch_size - 1) // args.batch_size

        for i in tqdm(
            range(155040, len(df), args.batch_size),
            desc="Translating captions and writing to file",
            total=total_batches,
        ):
            batch_sentences = df["caption"].iloc[i : i + args.batch_size].tolist()
            batch_indices = range(i, min(i + args.batch_size, len(df)))

            # English conversations
            en_conversations = [
                [
                    {
                        "role": "user",
                        "content": random.choice(EN_USERS_QUESTIONS),
                        "image_path": df["image_path"].iloc[idx],
                    },
                    {"role": "assistant", "content": batch_sentences[idx - i]},
                ]
                for idx in batch_indices
            ]
            write_buffer.append(
                "".join(
                    [
                        json.dumps(conv, ensure_ascii=False) + "\n"
                        for conv in en_conversations
                    ]
                )
            )

            # Translate in one go (not per-conversation)
            translated_batch = service.translate(batch_sentences)

            # Hindi conversations
            hi_conversations = [
                [
                    {
                        "role": "user",
                        "content": random.choice(HI_USERS_QUESTIONS),
                        "image_path": df["image_path"].iloc[idx],
                    },
                    {"role": "assistant", "content": translated_batch[idx - i]},
                ]
                for idx in batch_indices
            ]
            write_buffer.append(
                "".join(
                    [
                        json.dumps(conv, ensure_ascii=False) + "\n"
                        for conv in hi_conversations
                    ]
                )
            )

            # Write buffer if full or at end
            if len(
                write_buffer
            ) == args.write_buffer_size or i + args.batch_size >= len(df):
                while write_buffer:
                    f.write(write_buffer.popleft())
                f.flush()  # Ensure data is written to disk

    service.cleanup()
    return None


if __name__ == "__main__":
    args = parse_args()
    main(args)
