"""Translation service module using IndicTrans models.

This module is deprecated. Use src.data.translation.TranslationService instead.
"""

# Re-export the TranslationService from the canonical location
from src.data.translation import TranslationService

__all__ = ["TranslationService"]


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
