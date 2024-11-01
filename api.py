import os
import time
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import base64
from io import BytesIO
from PIL import Image
import tempfile

import os
import logging
from huggingface_hub import login

# Try to login with HF token if available
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    logging.warn("HUGGINGFACE_TOKEN not found in environment variables. Some model downloads may fail.")


from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

# Initialize FastAPI app
app = FastAPI(title="DeepSeek-VL API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model (global variables)
CHECKPOINT_PATH = os.getenv("DEEPSEEK_MODEL_PATH")
if not CHECKPOINT_PATH:
    raise ValueError("DEEPSEEK_MODEL_PATH environment variable not set")

# Load model and tokenizer globally
logging.info("Loading model...")
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(CHECKPOINT_PATH)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = MultiModalityCausalLM.from_pretrained(
    CHECKPOINT_PATH, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# Pydantic models for API
class ImageRequest(BaseModel):
    image: str  # base64 encoded image
    prompt: Optional[str] = "<TOPAZ AUTO CLIP CAPTION> Caption this image."
    max_new_tokens: Optional[int] = 128


class CaptionResponse(BaseModel):
    caption: str


def process_base64_image(image_data: str) -> Image.Image:
    """Process base64 image data into PIL Image"""
    try:
        # Remove data URL prefix if present
        if image_data.startswith("data:image"):
            image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")


@app.post("/v1/caption", response_model=CaptionResponse)
async def generate_caption(request: ImageRequest):
    try:
        # Process the image
        image = process_base64_image(request.image)

        # Generate caption using temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp_file:
            image.save(tmp_file.name)
            query = f"<img>{tmp_file.name}</img>{request.prompt}"
            conversation = [
                {
                    "role": "User",
                    "content": f"<image_placeholder>{request.prompt}",
                    "images": [tmp_file.name],
                },
                {"role": "Assistant", "content": ""},
            ]

            pil_images = load_pil_images(conversation)
            prepare_inputs = vl_chat_processor(
                conversations=conversation, images=pil_images, force_batchify=True
            ).to(vl_gpt.device)

            # run image encoder to get the image embeddings
            inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

            # run the model to get the response
            outputs = vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=request.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        response = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return CaptionResponse(caption=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
