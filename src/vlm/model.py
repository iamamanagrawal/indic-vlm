"""Vision-Language Model implementation.

This module implements the VisionLanguageModel class which combines vision and
language models with a modality projector for multimodal understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.schema import VLMGenerationConfig


class ModalityProjector(nn.Module):
    """
    Modality projector to bridge vision and language representations.

    This module projects vision embeddings into the language model's embedding space
    using pixel unshuffling and MLP layers.

    Attributes:
        proj (nn.Sequential): Sequential layers for projection.
    """

    def __init__(self, vision_dim: int, hidden_dim: int) -> None:
        """
        Initialize the ModalityProjector.

        Args:
            vision_dim (int): Dimension of vision embeddings.
            hidden_dim (int): Dimension of language model embeddings.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.RMSNorm(4 * vision_dim),
            nn.Linear(4 * vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.apply(self.__init_weights__)

    def __init_weights__(self, module) -> None:
        """
        Initialize weights of the projection layers.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project vision embeddings to language embedding space.

        Args:
            x (torch.Tensor): Vision embeddings tensor.

        Returns:
            torch.Tensor: Projected embeddings.
        """
        return self.proj(x)


class VisionLanguageModel(nn.Module):
    """
    Vision-Language Model combining vision and language understanding.

    This model integrates a vision encoder, language model, and modality projector
    to process multimodal inputs (text and images).

    Attributes:
        language_model: The language model for text generation.
        vision_model: The vision model for image encoding.
        projector (ModalityProjector): Projector to align vision and language spaces.
    """

    def __init__(self, language_model, vision_model) -> None:
        """
        Initialize the VisionLanguageModel.

        Args:
            language_model: Pre-trained language model.
            vision_model: Pre-trained vision model.
        """
        super().__init__()
        self.language_model = language_model
        self.vision_model = vision_model
        self.projector = ModalityProjector(
            vision_dim=vision_model.config.hidden_size,
            hidden_dim=language_model.config.hidden_size,
        )

    def from_pretrained_projection(self, projection_path: str) -> None:
        """
        Load pre-trained projector weights from checkpoint.

        Args:
            projection_path (str): Path to the projector checkpoint file.

        Returns:
            None
        """
        self.projector.load_state_dict(torch.load(projection_path))
        return None

    def _get_vision_embeds(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract and process vision embeddings from pixel values.

        Applies pixel unshuffling (2x2) to reduce spatial dimensions and increase
        channel dimensions, then projects to language embedding space.

        Args:
            pixel_values (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Projected vision embeddings of shape (B, H*W/4, hidden_dim).
        """
        # Forward pass through vision model and get the last hidden state
        vision_embeds = self.vision_model(pixel_values=pixel_values).last_hidden_state

        ## apply pixel unshuffling to reduce spatial dimensions
        batch_size, num_patches, hidden_dim = vision_embeds.size()
        h, w = int(num_patches**0.5), int(num_patches**0.5)
        assert num_patches % 4 == 0, (
            "Number of patches must be divisible by 4 for pixel unshuffling."
        )

        # Apply pixel unshuffling
        vision_embeds = vision_embeds.view(batch_size, h, w, hidden_dim)  # (B, H, W, D)
        vision_embeds = (
            (
                vision_embeds.view(batch_size, h // 2, 2, w // 2, 2, hidden_dim)
                .permute(0, 1, 3, 2, 4, 5)
                .contiguous()
            )
            .view(batch_size, h // 2, w // 2, -1)
            .contiguous()
        )  # (B, H/2, W/2, 4*D)
        vision_embeds = vision_embeds.view(
            batch_size, -1, vision_embeds.size(-1)
        ).contiguous()  # (B, H*W/4, 4*D)
        return self.projector(vision_embeds)

    def _prepare_inputs(
        self,
        input_ids: torch.Tensor,
        vision_embeds: torch.Tensor | None,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare inputs by replacing image tokens with vision embeddings.

        Args:
            input_ids (torch.Tensor): Token IDs including image token placeholders.
            vision_embeds (torch.Tensor | None): Vision embeddings to insert.
            attention_mask (torch.Tensor): Attention mask for the inputs.

        Returns:
            tuple: (input_embeddings, attention_mask) ready for language model.
        """
        device = input_ids.device
        input_embds = self.language_model.get_input_embeddings()(input_ids)

        if vision_embeds is None:
            return input_embds, attention_mask

        vision_embeds = vision_embeds.to(device=device, dtype=input_embds.dtype)

        # Find image token masks
        img_token_mask = (input_ids == self.language_model.image_token_id).to(
            device=device, dtype=torch.bool
        )
        img_token_mask = img_token_mask.unsqueeze(-1).expand(
            -1, -1, input_embds.size(-1)
        )

        expected_count = vision_embeds.numel()
        actual_count = img_token_mask.sum()
        torch._assert(
            actual_count == expected_count,
            "Number of image tokens in input_ids must match number of vision embeddings.",
        )

        # Replace image tokens with vision embeddings
        vision_embeds = vision_embeds.to(device=device, dtype=input_embds.dtype)
        input_embds = input_embds.masked_scatter(img_token_mask, vision_embeds)

        return input_embds, attention_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None,
        attention_mask: torch.Tensor,
        targets: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, float | None]:
        """
        Forward pass through the VLM.

        Args:
            input_ids (torch.Tensor): Token IDs with image token placeholders.
            pixel_values (torch.Tensor | None): Image pixel values.
            attention_mask (torch.Tensor): Attention mask.
            targets (torch.Tensor | None): Target token IDs for loss computation.

        Returns:
            tuple: (logits, loss) where loss is None if targets not provided.
        """
        assert targets is None or input_ids.shape == targets.shape, (
            "Input IDs and targets must have the same shape."
        )

        # Get vision embeddings
        vision_embeds = None
        if pixel_values is not None:
            vision_embeds = self._get_vision_embeds(pixel_values)

        ## prepare inputs for language model
        input_embds, attention_mask = self._prepare_inputs(
            input_ids=input_ids,
            vision_embeds=vision_embeds,
            attention_mask=attention_mask,
        )

        # Forward pass through language model
        output = self.language_model(
            inputs_embeds=input_embds,
            attention_mask=attention_mask,
        )

        # Compute logits and loss
        logits: torch.Tensor = output.logits
        loss = None
        if targets is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None,
        attention_mask: torch.Tensor,
        generation_config: VLMGenerationConfig = VLMGenerationConfig(),
    ) -> torch.Tensor:
        """
        Generate text autoregressively using the VLM.

        Args:
            input_ids (torch.Tensor): Input token IDs with image token placeholders.
            pixel_values (torch.Tensor | None): Image pixel values.
            attention_mask (torch.Tensor): Attention mask.
            generation_config (VLMGenerationConfig): Configuration for text generation.

        Returns:
            torch.Tensor: Generated token IDs.
        """
        # Get vision embeddings
        vision_embeds = None
        if pixel_values is not None:
            vision_embeds = self._get_vision_embeds(pixel_values)

        ## prepare inputs for language model
        input_embds, attention_mask = self._prepare_inputs(
            input_ids=input_ids,
            vision_embeds=vision_embeds,
            attention_mask=attention_mask,
        )

        # Generate sequences using language model
        return self.language_model.generate(
            inputs_embeds=input_embds,
            attention_mask=attention_mask,
            max_new_tokens=generation_config.max_new_tokens,
            top_p=generation_config.top_p,
            temperature=generation_config.temperature,
            do_sample=generation_config.do_sample,
        )
