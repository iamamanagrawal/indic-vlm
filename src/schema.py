"""
Configuration dataclasses for the Indic VLM project.

This module defines configuration classes for different components of the
vision-language model, including model architecture, training, translation,
and inference configurations.
"""

from dataclasses import dataclass


@dataclass
class VLMModelConfig:
    """
    Configuration for Vision-Language Model architecture.

    Attributes:
        language_model: Path to the language model.
        vision_model: Path to the vision model.
        num_image_tokens: Number of tokens to represent each image.
    """

    language_model: str
    vision_model: str

    num_image_tokens: int
    attn_implementation: str


@dataclass
class VLMTrainerConfig:
    """
    Configuration for Vision-Language Model training.

    Attributes:
        use_muon: Whether to use the Muon optimizer instead of AdamW.
        batch_size: Training batch size.
        learning_rate: Learning rate for optimization.
        num_epochs: Number of training epochs.
        gradient_accumulation_steps: Steps to accumulate gradients before updating.
        max_grad_norm: Maximum gradient norm for clipping.
        compile: Whether to compile the model with torch.compile.
        freeze_vision_model: Whether to freeze vision model parameters.
        freeze_language_model: Whether to freeze language model parameters.
        projector_checkpoint_path: Path to save/load projector checkpoints.
        num_workers: Number of worker processes for data loading.
    """

    batch_size: int
    learning_rate: float
    num_epochs: int
    gradient_accumulation_steps: int
    max_grad_norm: float
    compile: bool
    use_muon: bool

    freeze_vision_model: bool
    freeze_language_model: bool
    projector_checkpoint_path: str
    num_workers: int = 4


@dataclass
class TranslationConfig:
    """
    Configuration for translation service.

    Attributes:
        model_name: Path to the translation model.
        src_lang: Source language code (e.g., 'eng_Latn').
        tgt_lang: Target language code (e.g., 'hin_Deva').
        flash_attention: Whether to use Flash Attention.
        max_length: Maximum sequence length for translation.
        batch_size: Batch size for translation.
    """

    model_name: str = "models/indictrans2-en-indic-1B"
    src_lang: str = "eng_Latn"
    tgt_lang: str = "hin_Deva"
    flash_attention: bool = True
    max_length: int = 256
    batch_size: int = 16


@dataclass
class VLMGenerationConfig:
    """
    Configuration for text generation.

    Attributes:
        max_new_tokens: Maximum number of new tokens to generate.
        top_p: Nucleus sampling probability threshold.
        temperature: Sampling temperature for randomness control.
        do_sample: Whether to use sampling instead of greedy decoding.
    """

    max_new_tokens: int = 100
    top_p: float = 0.9
    temperature: float = 0.7
    do_sample: bool = True
    repetition_penalty: float = 1.2
    use_cache: bool = True
