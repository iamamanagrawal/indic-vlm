"""Base trainer class for VLM training.

This module defines the abstract base class for VLM trainers,
providing a common interface for different training strategies.
"""

import torch

from torch.optim import Optimizer
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.schema import VLMTrainerConfig
from src.data.dataloader import DataLoader
from src.vlm.model import VisionLanguageModel


@dataclass
class BaseTrainer(ABC):
    """
    Abstract base class for VLM trainers.

    Attributes:
        model (VisionLanguageModel): The VLM model to train.
        train_config (VLMTrainerConfig): Training configuration.
        tokenizer (object): Tokenizer for text processing.
        vision_processor (object): Processor for image preprocessing.
        train_dataloader (DataLoader): DataLoader for training data.
        device (str): Device to run training on ('mps', 'cuda', or 'cpu').
        val_dataloader (DataLoader | None): Optional validation dataloader.
        optimizer (Optimizer | None): Optional custom optimizer.
        example_conversation (list[dict[str, str]]): Example conversation for inference.
    """

    model: VisionLanguageModel
    train_config: VLMTrainerConfig
    tokenizer: object
    vision_processor: object

    train_dataloader: DataLoader
    val_dataloader: DataLoader
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    optimizer: Optimizer | None = None
    # Example conversation for inference during training
    example_conversation: list[dict[str, str]] = field(
        default_factory=lambda: [
            [
                {
                    "role": "user",
                    "content": "Describe the image.",
                    "image_path": "examples/image.png",
                }
            ],
            [
                {
                    "role": "user",
                    "content": "तस्वीर के बारे में बताओ।",
                    "image_path": "examples/image.png",
                }
            ],
        ]
    )

    @abstractmethod
    def train(self):
        """
        Execute the training loop.

        This method must be implemented by subclasses.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        return NotImplementedError("Train method not implemented.")

    @abstractmethod
    def setup_optimizer(self):
        """
        Set up the optimizer for training.

        This method must be implemented by subclasses.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        return NotImplementedError("Setup optimizer method not implemented.")
