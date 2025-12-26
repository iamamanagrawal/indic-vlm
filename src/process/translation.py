import torch
import gc
from IndicTransToolkit import IndicProcessor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class TranslationService:
    def __init__(
        self,
        model_name: str = "models/indictrans2-en-indic-1B",
        src_lang: str = "eng_Latn",
        tgt_lang: str = "hin_Deva",
        flash_attention: bool = True,
        max_length: int = 256,
    ) -> None:
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length

        self.ip = IndicProcessor(inference=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if flash_attention else "default",
        ).to(self.device)

    def translate(self, sentences: list[str]) -> list[str]:
        batch = self.ip.preprocess_batch(
            sentences, src_lang=self.src_lang, tgt_lang=self.tgt_lang, visualize=False
        )
        batch = self.tokenizer(
            batch,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **batch,
                num_beams=1,
                num_return_sequences=1,
                max_length=self.max_length,
                use_cache=False,
            )

        outputs = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.ip.postprocess_batch(outputs, lang=self.tgt_lang)

    def cleanup(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if hasattr(self, "ip"):
            del self.ip

        # Clear GPU/MPS cache
        if self.device != "cpu":
            torch.cuda.empty_cache() if self.device == "cuda" else torch.mps.empty_cache()
        gc.collect()


if __name__ == "__main__":
    service = TranslationService(flash_attention=False)

    test_sentences = [
        "This is a test sentence.",
        "Here is another longer test sentence to check the translation quality.",
        "Please send an SMS to 9876543210 and an email to newemail123@xyz.com by 15th October, 2023.",
    ]
    outputs = service.translate(test_sentences)
    for inp, out in zip(test_sentences, outputs):
        print(f"Input: {inp}\nTranslated: {out}\n")

    service.cleanup()
