"""Utility functions and constants for the Indic VLM project.

This module provides utilities for chat template processing, model loading,
and predefined question templates in English and Hindi for image description tasks.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SiglipImageProcessor,
    SiglipVisionModel,
)
from src.schema import VLMModelConfig

# Generated image description queries from users from Gemini 3 Flash model
EN_USERS_QUESTIONS = [
    "What do you see in this picture?",
    "Can you tell me what is happening here?",
    "Please provide an overview of this visual.",
    "How would you explain this image to someone who cannot see it?",
    "What is being depicted in this scene?",
    "Could you walk me through the contents of this image?",
    "What are the main elements present in this photo?",
    "How would you summarize what is shown here?",
    "What is your interpretation of this visual?",
    "Please break down what we are looking at.",
    "What captures your attention in this image?",
    "Can you give me a rundown of this scene?",
    "What story does this image tell?",
    "What are the key features of this picture?",
    "How would you characterize this photograph?",
    "What do you notice first when looking at this?",
    "Can you offer a detailed account of what is shown?",
    "What information can be gathered from this visual?",
    "Please paint a picture with words of what you see.",
    "What is the overall impression of this image?",
    "What is being showcased in this frame?",
    "Could you provide a general analysis of this picture?",
    "What can you tell me about the composition of this image?",
    "How would you describe the subject matter here?",
    "What is the context of this visual?",
    "What is the visual narrative here?",
    "What does this scene consist of?",
    "How would you relay the details of this image?",
    "What is the essence of this picture?",
    "Can you describe the layout of this visual?",
    "What are the components of this photograph?",
    "How would you present this image to a listener?",
    "What can be observed in this frame?",
    "What is the primary focus of this image?",
    "Could you elaborate on what is visible here?",
    "What is the nature of this visual content?",
    "How would you define the scene depicted?",
    "What is the overall look of this image?",
    "What details stand out to you in this picture?",
    "Can you give a general report on this visual?",
    "What is the setting shown in this image?",
    "How would you map out what is in this photo?",
    "What do you perceive in this visual representation?",
    "What is the theme of this image?",
    "Can you share your observations of this picture?",
    "What is the arrangement of elements here?",
    "How would you narrate this visual scene?",
    "What is the core subject of this photograph?",
    "What can you deduce from looking at this image?",
    "How would you sketch this scene using only words?",
]

HI_USERS_QUESTIONS = [
    "इस तस्वीर में आपको क्या दिख रहा है?",
    "क्या आप मुझे बता सकते हैं कि यहाँ क्या हो रहा है?",
    "कृपया इस दृश्य का एक अवलोकन प्रदान करें।",
    "आप इस छवि को किसी ऐसे व्यक्ति को कैसे समझाएंगे जो इसे देख नहीं सकता?",
    "इस दृश्य में क्या दर्शाया जा रहा है?",
    "क्या आप मुझे इस छवि की सामग्री के बारे में विस्तार से बता सकते हैं?",
    "इस फोटो में मौजूद मुख्य तत्व क्या हैं?",
    "यहाँ जो दिखाया गया है उसका आप सारांश कैसे देंगे?",
    "इस दृश्य के बारे में आपकी क्या व्याख्या है?",
    "कृपया विश्लेषण करें कि हम क्या देख रहे हैं।",
    "इस छवि में आपका ध्यान क्या खींचता है?",
    "क्या आप मुझे इस दृश्य का विवरण दे सकते हैं?",
    "यह छवि क्या कहानी बताती है?",
    "इस तस्वीर की मुख्य विशेषताएं क्या हैं?",
    "आप इस तस्वीर को कैसे परिभाषित करेंगे?",
    "इसे देखते समय आप सबसे पहले क्या नोटिस करते हैं?",
    "क्या आप जो दिखाया गया है उसका विस्तृत विवरण दे सकते हैं?",
    "इस दृश्य से क्या जानकारी प्राप्त की जा सकती है?",
    "आप जो देखते हैं उसे शब्दों के माध्यम से चित्रित करें।",
    "इस छवि का समग्र प्रभाव क्या है?",
    "इस फ्रेम में क्या प्रदर्शित किया जा रहा है?",
    "क्या आप इस तस्वीर का एक सामान्य विश्लेषण प्रदान कर सकते हैं?",
    "आप मुझे इस छवि की रचना (composition) के बारे में क्या बता सकते हैं?",
    "आप यहाँ की विषय-वस्तु का वर्णन कैसे करेंगे?",
    "इस दृश्य का संदर्भ क्या है?",
    "यहाँ दृश्य कथानक क्या है?",
    "इस दृश्य में क्या-क्या शामिल है?",
    "आप इस छवि के विवरण दूसरों तक कैसे पहुँचाएंगे?",
    "इस तस्वीर का सार क्या है?",
    "क्या आप इस दृश्य के लेआउट का वर्णन कर सकते हैं?",
    "इस तस्वीर के घटक क्या हैं?",
    "आप इस छवि को किसी श्रोता के सामने कैसे प्रस्तुत करेंगे?",
    "इस फ्रेम में क्या देखा जा सकता है?",
    "इस छवि का प्राथमिक फोकस क्या है?",
    "यहाँ जो दिखाई दे रहा है, क्या आप उस पर विस्तार से बता सकते हैं?",
    "इस दृश्य सामग्री की प्रकृति क्या है?",
    "आप इस चित्रित दृश्य को कैसे परिभाषित करेंगे?",
    "इस छवि का समग्र रूप कैसा है?",
    "इस तस्वीर में कौन से विवरण आपको सबसे अलग लगते हैं?",
    "क्या आप इस दृश्य पर एक सामान्य रिपोर्ट दे सकते हैं?",
    "इस छवि में दिखाया गया परिवेश क्या है?",
    "आप इस फोटो में मौजूद चीजों का खाका कैसे तैयार करेंगे?",
    "आप इस दृश्य प्रतिनिधित्व में क्या देख पा रहे हैं?",
    "इस छवि का विषय (theme) क्या है?",
    "क्या आप इस तस्वीर के बारे में अपने अवलोकन साझा कर सकते हैं?",
    "यहाँ तत्वों की व्यवस्था क्या है?",
    "आप इस दृश्य का वर्णन कैसे करेंगे?",
    "इस तस्वीर का मूल विषय क्या है?",
    "इस छवि को देखने से आप क्या निष्कर्ष निकाल सकते हैं?",
    "आप केवल शब्दों का उपयोग करके इस दृश्य का खाका कैसे खींचेंगे?",
]


def apply_chat_template(
    tokenizer, vision_processor, conversation, add_generation_prompt=False
) -> dict[str, torch.Tensor | list[str]]:
    """
    Apply chat template to the conversation and convert to token IDs.

    Args:
        tokenizer: The tokenizer used for tokenization. Must have 'num_image_tokens' attribute.
        conversation (list[dict]): A list of conversation turns, where each turn is a
            dictionary with 'role', 'content', and optional 'pixel_values' keys.
        add_generation_prompt (bool): Whether to add a generation prompt for the
            assistant at the end of the conversation.

    Returns:
        dict: A dictionary containing:
            - input_ids: Tensor of token IDs for the conversation.
            - attention_mask: Tensor of attention mask.
            - pixel_values: Tensor of processed image pixel values.
            - targets: Tensor of target IDs for training (with -100 for non-target tokens).

    Raises:
        AssertionError: If tokenizer lacks 'num_image_tokens' or conversation is empty.
        ValueError: If last turn is not from user when add_generation_prompt is True.
    """
    assert len(conversation) > 0, "Conversation must have at least one turn."

    if add_generation_prompt:
        if conversation[-1]["role"] != "user":
            raise ValueError(
                "The last turn must be from the user when add_generation_prompt is True."
            )

    start_turn = "<start_of_turn>"
    end_turn = "<end_of_turn>"

    prompt_ids = [tokenizer.bos_token_id]
    attention_mask = [1]
    mask = [0]
    image_path = []

    for turn in conversation:
        assert "role" in turn and "content" in turn, (
            "Each turn must have 'role' and 'content' keys."
        )
        role = turn["role"]
        content = turn["content"]
        ids = tokenizer.encode(
            f"{start_turn}{role}\n{content}{end_turn}\n",
        )[1:]
        prompt_ids.extend(ids)
        attention_mask.extend([1] * len(ids))

        if role == "user":
            mask.extend([0] * len(ids))
            if "image_path" in turn and turn["image_path"] is not None:
                if isinstance(turn["image_path"], list):
                    num_images = len(turn["image_path"])
                    image_path.extend(turn["image_path"])
                else:
                    num_images = 1
                    image_path.append(turn["image_path"])

                image_ids = (
                    [tokenizer.boi_token_id]
                    + [tokenizer.image_token_id]
                    * tokenizer.init_kwargs["num_image_tokens"]
                    + [tokenizer.eoi_token_id]
                )
                prompt_ids.extend(image_ids * num_images)
                attention_mask.extend([1] * (len(image_ids) * num_images))
                mask.extend([0] * (len(image_ids) * num_images))
        else:
            mask.extend([0, 0] + [1] * (len(ids) - 2))

    if add_generation_prompt:
        ids = tokenizer.encode(f"{start_turn}assistant\n")[1:]
        prompt_ids.extend(ids)
        attention_mask.extend([1] * len(ids))
        mask.extend([1] * len(ids))

    targets = [-100 if m == 0 else pid for pid, m in zip(prompt_ids, mask)]
    return {
        "input_ids": torch.tensor(prompt_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "pixel_values": vision_processor(
            images=image_path, return_tensors="pt"
        ).pixel_values
        if len(image_path) > 0
        else None,
        "targets": torch.tensor(targets, dtype=torch.long)
        if add_generation_prompt
        else None,
    }


def load_tokenizer_and_model(
    config: VLMModelConfig,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load the tokenizer and language model with custom configurations.

    This function loads the model and tokenizer, adds special image tokens,
    and resizes the token embeddings accordingly.

    Args:
        config (VLMModelConfig): Configuration containing model paths and settings.

    Returns:
        tuple: A tuple containing:
            - tokenizer (AutoTokenizer): The tokenizer for the language model.
            - model (AutoModelForCausalLM): The language model for text generation.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.language_model)
    model = AutoModelForCausalLM.from_pretrained(
        config.language_model,
        attn_implementation=config.attn_implementation,
        dtype=torch.bfloat16,
    )

    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                tokenizer.image_token,
            ]
        }
    )
    tokenizer.init_kwargs = {"num_image_tokens": config.num_image_tokens}
    model.config.image_token_id = tokenizer.convert_tokens_to_ids(tokenizer.image_token)
    model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model


def load_vision_processor_and_model(
    config: VLMModelConfig,
) -> tuple[SiglipImageProcessor, SiglipVisionModel]:
    """
    Load the vision processor and vision model.

    Args:
        config (VLMModelConfig): Configuration containing vision model path.

    Returns:
        tuple: A tuple containing:
            - image_processor (SiglipImageProcessor): The image processor for preprocessing images.
            - vision_model (SiglipVisionModel): The vision model for extracting image features.
    """
    image_processor = SiglipImageProcessor.from_pretrained(config.vision_model)
    vision_model = SiglipVisionModel.from_pretrained(
        config.vision_model, dtype=torch.bfloat16
    )
    return image_processor, vision_model
