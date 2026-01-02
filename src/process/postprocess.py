import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
SRC_LANG = "eng_Latn"
TGT_LANG = "hin_Deva"
BATCH_SIZE = 1024
MAX_LENGTH = 300
MODEL = AutoModelForSeq2SeqLM.from_pretrained(
    "models/nllb-200-distilled-600M", trust_remote_code=True, dtype=torch.bfloat16
).to(DEVICE)
TOKENIZER = AutoTokenizer.from_pretrained(
    "models/nllb-200-distilled-600M", trust_remote_code=True, src_lang=SRC_LANG
)


def transform_conversation(item):
    """Transform conversation format from old to new schema."""
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


def translate(texts: list[str]) -> list[str]:
    """Translate a list of texts from source language to target language."""
    num = len(texts)
    translated_texts = []

    for idx in tqdm(range(0, num, BATCH_SIZE)):
        batch_texts = texts[idx : idx + BATCH_SIZE]
        inputs = TOKENIZER(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(DEVICE)

        with torch.inference_mode() and torch.autocast(
            device_type=DEVICE, dtype=torch.bfloat16
        ):
            outputs = MODEL.generate(
                **inputs,
                num_beams=1,
                forced_bos_token_id=TOKENIZER.convert_tokens_to_ids(TGT_LANG),
                num_return_sequences=1,
                max_length=MAX_LENGTH,
                use_cache=True,
                do_sample=True,
                top_p=0.9,
            )

        decoded_texts = TOKENIZER.batch_decode(
            outputs,
            skip_special_tokens=True,
        )

        ## save results on jsonl for backup
        mapp = dict(zip(batch_texts, decoded_texts))
        with open("data/translation_map.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(mapp, ensure_ascii=False) + "\n")

        translated_texts.extend(decoded_texts)

    return translated_texts


def process_conversations(input_path, output_path):
    """Process conversations from input file and save transformed data to output file."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    conversations = [transform_conversation(item) for item in data]

    contents = [message["content"] for conv in conversations for message in conv]

    translated_contents = translate(contents)

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
