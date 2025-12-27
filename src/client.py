import time
import json
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import traceback

from src.schema import VLMModelConfig, VLMGenerationConfig
from src.utils import apply_chat_template
from src.vlm.model import VisionLanguageModel
from src.utils import load_tokenizer_and_model, load_vision_processor_and_model


app = FastAPI()


class Message(BaseModel):
    role: str
    content: str
    image_path: str | None = None


class Request(BaseModel):
    messages: list[Message]


class Response(BaseModel):
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
        [json.loads(message.model_dump_json()) for message in request.messages],
        add_generation_prompt=True,
    )
    pixel_values = (
        vision_processor(images=result["image_path"], return_tensors="pt")[
            "pixel_values"
        ]
        if len(result["image_path"]) > 0
        else None
    )

    result = {
        "input_ids": result["input_ids"].unsqueeze(0),
        "pixel_values": pixel_values,
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
    config = VLMModelConfig(
        language_model="models/gemma-3-1b-it",
        vision_model="models/siglip-base-patch16-256-multilingual",
        num_image_tokens=64,
        attn_implementation="sdpa",
    )
    global tokenizer, vision_processor, model, device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, language_model = load_tokenizer_and_model(config)
    vision_processor, vision_model = load_vision_processor_and_model(config)

    model = VisionLanguageModel(language_model, vision_model)
    model.from_pretrained_projection("checkpoints/projector.pth")

    model.to(device)
    model.eval()
    return None
