"""Translation service module using IndicTrans models.

This module provides a translation service for translating text between
Indian languages and English using the IndicTrans2 model.
"""

import torch
import gc
from IndicTransToolkit import IndicProcessor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class TranslationService:
    """
    Translation service for Indic languages using IndicTrans2 models.

    Attributes:
        device (str): Device to run the model on ('cuda', 'mps', or 'cpu').
        src_lang (str): Source language code.
        tgt_lang (str): Target language code.
        ip (IndicProcessor): Indic processor for text preprocessing.
        tokenizer (AutoTokenizer): Tokenizer for the translation model.
        model (AutoModelForSeq2SeqLM): Translation model.
    """

    def __init__(
        self,
        model_name: str = "models/indictrans2-en-indic-dist-200M",
        src_lang: str = "eng_Latn",
        tgt_lang: str = "hin_Deva",
        flash_attention: bool = True,
    ) -> None:
        """
        Initialize the TranslationService.

        Args:
            model_name (str): Path to the IndicTrans2 model.
            src_lang (str): Source language code (e.g., 'eng_Latn').
            tgt_lang (str): Target language code (e.g., 'hin_Deva').
            flash_attention (bool): Whether to use Flash Attention 2 for faster inference.
        """
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.ip = IndicProcessor(inference=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if flash_attention else "sdpa",
            dtype=torch.bfloat16,
        ).to(self.device)
        self.model.eval()

        if self.device == "cuda":
            torch.set_float32_matmul_precision("high")
            self.model = torch.compile(self.model)
            print("Model compiled with torch.compile for CUDA device.")

    @torch.inference_mode()
    def translate(self, sentences: list[str]) -> list[str]:
        """
        Translate a batch of sentences from source to target language.

        Args:
            sentences (list[str]): List of sentences to translate.

        Returns:
            list[str]: List of translated sentences.
        """
        batch = self.ip.preprocess_batch(
            sentences, src_lang=self.src_lang, tgt_lang=self.tgt_lang, visualize=False
        )
        batch = self.tokenizer(
            batch,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            outputs = self.model.generate(
                **batch,
                num_beams=1,
                num_return_sequences=1,
                max_new_tokens=200,
                use_cache=False,
            )

        outputs = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.ip.postprocess_batch(outputs, lang=self.tgt_lang)

    def cleanup(self):
        """
        Clean up resources by deleting model, tokenizer, and processor.

        This method frees GPU/MPS memory and performs garbage collection.
        """
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
