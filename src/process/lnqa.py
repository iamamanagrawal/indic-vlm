"""LNQA dataset processing module.

This module provides functionality for translating the LNQA (Long-form Natural
Question Answering) dataset from English to Hindi and uploading to Hugging Face.
"""

import os
import json
import torch
import argparse

from tqdm import tqdm
from huggingface_hub import HfApi
from IndicTransToolkit.processor import IndicProcessor
from datasets import load_dataset, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.utils import get_device

API = HfApi(token=os.getenv("HF_TOKEN"))


def init_model(args: argparse.Namespace) -> tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    """Initialize translation model and tokenizer.

    Args:
        args: Command-line arguments containing model path and device.

    Returns:
        tuple: (model, tokenizer) for translation.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="flex_attention",
    ).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    return model, tokenizer


def save_upload(repo_id: str, batch: dict, counter: int) -> None:
    """
    Save batch to parquet and upload to Hugging Face Hub.

    Args:
        repo_id: Hugging Face repository ID.
        batch: Dictionary containing image and qa data.
        counter: Batch counter for filename.

    Returns:
        None
    """
    ds = Dataset.from_dict(batch)
    filename = f"train-{counter:010d}.parquet"
    ds.to_parquet(filename)

    API.upload_file(
        path_or_fileobj=filename,
        path_in_repo=f"data/{filename}",
        repo_id=repo_id,
        repo_type="dataset",
    )

    os.remove(filename)
    return None


def main(args: argparse.Namespace) -> None:
    """
    Main function to translate LNQA dataset and upload to Hugging Face.

    Args:
        args: Command-line arguments containing paths and configuration.

    Returns:
        None
    """
    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        conversations = [json.loads(line) for line in f.readlines()]

    ## gather the textual contents
    contents = []
    for conversation in conversations:
        contents.append(conversation["question"])
        contents.append(conversation["answer"])

    ## initialize model and tokenizer
    model, tokenizer = init_model(args)
    ip = IndicProcessor(inference=True)
    model.eval()

    ## process in batches
    print("Translating contents...")
    translated_texts = dict()
    for idx in tqdm(
        range(0, len(contents), args.batch_size),
        total=len(contents) // args.batch_size + 1,
    ):
        ## prepare batch and tokenize
        batch_texts = contents[idx : idx + args.batch_size]
        texts = ip.preprocess_batch(
            batch_texts, src_lang=args.src_lang, tgt_lang=args.tgt_lang
        )
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_attention_mask=True,
        ).to(args.device)

        ## generate translations
        with torch.inference_mode():
            generated_tokens = model.generate(
                **inputs,
                num_beams=1,
                num_return_sequences=1,
                max_length=args.max_length,
                use_cache=False,
                do_sample=True,
                top_p=0.9,
            )

        ## decode translations
        decoded_texts = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        decoded_texts = ip.postprocess_batch(decoded_texts, lang=args.tgt_lang)

        ## store translations
        for original, translated in zip(batch_texts, decoded_texts):
            translated_texts[original] = translated

        ## always save intermediate results
        with open("lnqa_hi_en.json", "w", encoding="utf-8") as f:
            json.dump(translated_texts, f, ensure_ascii=False, indent=4)

    print("Translation Completed!!")

    batch = {"image": [], "qa": []}
    count = 0
    ds = load_dataset(args.hf_dataset, split="train", streaming=True)

    for idx, example in enumerate(ds):
        original = example["qa"]
        new = list()

        for conv in original:
            question = translated_texts.get(conv["question"], conv["question"])
            answer = translated_texts.get(conv["answer"], conv["answer"])

            new.append({"question": question, "answer": answer})

        batch["image"].append(example["image"])
        batch["qa"].append(new)

        batch["image"].append(example["image"])
        batch["qa"].append(original)

        if len(batch["qa"]) >= args.shard_size:
            save_upload(args.hf_save_repo, batch, count)
            batch = {"image": [], "qa": []}
            count += 1
            print(f"Saved batch {count}!")

    if len(batch["qa"]) > 0:
        save_upload(args.hf_save_repo, batch, count)

    print("All done!!")

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl_path",
        type=str,
        default="lnqa.jsonl",
        help="Path to the LNQA JSONL file",
    )
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default="vikhyatk/lnqa",
        help="Hugging Face dataset identifier for LNQA",
    )
    parser.add_argument(
        "--hf_save_repo",
        type=str,
        default="Dark7Devil/lnqa-hi-en",
        help="Hugging Face repository to save the translated dataset",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="models/nllb-200-distilled-600M",
        help="Model name or path for translation",
    )
    parser.add_argument(
        "--src_lang",
        type=str,
        default="eng_Latn",
        help="Source language code",
    )
    parser.add_argument(
        "--tgt_lang",
        type=str,
        default="hin_Deva",
        help="Target language code",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=get_device(),
        help="Device to run the model on",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=64,
        help="Maximum sequence length for the model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for processing the dataset",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=5000,
        help="Number of examples to process before saving intermediate results",
    )

    args = parser.parse_args()
    main(args)
