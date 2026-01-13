"""FastAPI client for VLM inference.

This module provides a REST API for generating text from images
using the Vision-Language Model.
"""

import time
import json
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import traceback

from transformers import AutoTokenizer, SiglipImageProcessor

from src.schema import VLMGenerationConfig
from src.utils import apply_chat_template, get_device
from src.vlm.model import VisionLanguageModel


app = FastAPI(title="Indic VLM API", description="Vision-Language Model inference API")


class Message(BaseModel):
    """A single message in a conversation."""

    role: str
    content: str
    image_path: str | None = None


class Request(BaseModel):
    """Request body for generation endpoint."""

    messages: list[Message]


class Response(BaseModel):
    """Response body from generation endpoint."""

    status: bool
    response: str | None = None
    generation_time: float | None = None
    error: str | None = None


@app.get("/health")
async def read_root():
    return {"status": "ok"}


@app.post("/generate")
async def generate(request: Request) -> Response:
    result = apply_chat_template(
        tokenizer,
        vision_processor,
        [json.loads(message.model_dump_json()) for message in request.messages],
        add_generation_prompt=True,
    )

    result = {
        "input_ids": result["input_ids"].unsqueeze(0),
        "pixel_values": result["pixel_values"],
        "attention_mask": result["attention_mask"].unsqueeze(0),
    }

    try:
        result = {k: v.to(device) if v is not None else None for k, v in result.items()}
        start_time = time.time()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            generated_ids = model.generate(
                input_ids=result["input_ids"],
                pixel_values=result["pixel_values"],
                attention_mask=result["attention_mask"],
                generation_config=VLMGenerationConfig(),
            )
        end_time = time.time()
        generated_text = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return {
            "status": True,
            "response": generated_text,
            "generation_time": end_time - start_time,
        }
    except Exception:
        return {"status": False, "error": traceback.format_exc()}


@app.on_event("startup")
def init_vlm() -> None:
    """Initialize the VLM model, tokenizer, and vision processor on startup."""
    global tokenizer, vision_processor, model, device
    checkpoint_dir = "checkpoints/indic-vlm"
    model = VisionLanguageModel.from_pretrained(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(f"{checkpoint_dir}/tokenizer")
    vision_processor = SiglipImageProcessor.from_pretrained(
        f"{checkpoint_dir}/vision_processor"
    )

    device = get_device()
    model.to(device=device)
    model.eval()
    return None
