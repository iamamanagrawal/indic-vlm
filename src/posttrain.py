import argparse
import torch

from transformers import AutoTokenizer, SiglipImageProcessor

from src.trainer.posttrainer import PostTrainer
from src.schema import VLMTrainerConfig
from src.vlm.model import VisionLanguageModel
from src.data.dataloader import DataLoader


# Enable TF32 for faster matrix multiplications on Ampere+ GPUs
torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"

# Enable cudnn benchmarking for consistent input sizes
torch.backends.cudnn.benchmark = True

# Set to high precision for better speed
torch.set_float32_matmul_precision("high")


def parse_args():
    """
    Parse command-line arguments for VLM posttraining.

    Returns:
        argparse.Namespace: Parsed arguments containing model configuration,
            training parameters, and data paths.
    """
    parser = argparse.ArgumentParser(description="VLM Posttraining")

    ## Model config
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/indic_vlm_pretrain",
        help="Pretrained vision language model checkpoint directory",
    )

    # Training config
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm"
    )
    parser.add_argument("--compile", type=bool, default=True, help="Compile model")
    parser.add_argument(
        "--use_muon", type=bool, default=False, help="Use Muon optimizer"
    )
    parser.add_argument(
        "--freeze_vision_model", type=bool, default=True, help="Freeze vision model"
    )
    parser.add_argument(
        "--freeze_language_model", type=bool, default=True, help="Freeze language model"
    )
    parser.add_argument(
        "--projector_checkpoint_path",
        type=str,
        default="checkpoints/projector.pth",
        help="Projector checkpoint path",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loader workers"
    )

    # Data config
    parser.add_argument(
        "--train_conversations",
        type=str,
        default="data/sbu_captions-hindi-english-pretrain/train.jsonl",
        help="Training conversations path",
    )
    parser.add_argument(
        "--val_conversations",
        type=str,
        default="data/sbu_captions-hindi-english-pretrain/test.jsonl",
        help="Validation conversations path",
    )
    parser.add_argument(
        "--max_length", type=int, default=360, help="Maximum sequence length"
    )

    return parser.parse_args()


def main():
    """
    Main function for VLM pretraining.

    Initializes model components, data loaders, and trainer, then starts the training process.

    Returns:
        None
    """
    args = parse_args()

    checkpoint_dir = args.checkpoint_dir
    model = VisionLanguageModel.from_pretrained(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    vision_processor = SiglipImageProcessor.from_pretrained(checkpoint_dir)

    train_config = VLMTrainerConfig(
        use_muon=args.use_muon,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        compile=args.compile,
        freeze_vision_model=args.freeze_vision_model,
        freeze_language_model=args.freeze_language_model,
        projector_checkpoint_path=args.projector_checkpoint_path,
        num_workers=args.num_workers,
    )

    train_dataloader = DataLoader(
        conversations_path=args.train_conversations,
        tokenizer=tokenizer,
        vision_processor=vision_processor,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
    )

    val_dataloader = DataLoader(
        conversations_path=args.val_conversations,
        tokenizer=tokenizer,
        vision_processor=vision_processor,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
    )

    trainer = PostTrainer(
        model=model,
        tokenizer=tokenizer,
        vision_processor=vision_processor,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        train_config=train_config,
    )

    trainer.train()

    return None


if __name__ == "__main__":
    main()
