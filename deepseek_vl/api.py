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
import pycld2

import os
import logging
from huggingface_hub import login

# Try to login with HF token if available
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    logging.warn(
        "HUGGINGFACE_TOKEN not found in environment variables. Some model downloads may fail."
    )


from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
from vllm import LLM, SamplingParams

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

# Load translation model
TRANSLATION_MODEL_PATH = "/app/models/translation_model"
translation_tokenizer = "/app/models/translation_model"
translation_llm = LLM(model=TRANSLATION_MODEL_PATH, gpu_memory_utilization=0.5, tokenizer=translation_tokenizer, enforce_eager=True, device="cuda:0", max_model_len=512)

# Pydantic models for API
class ImageRequest(BaseModel):
    image: str  # base64 encoded image
    prompt: Optional[str] = "<TOPAZ AUTO CLIP CAPTION> Caption this image."
    max_new_tokens: Optional[int] = 128


class CaptionResponse(BaseModel):
    caption: str


class TranslationRequest(BaseModel):
    prompt: str


class TranslationResponse(BaseModel):
    language: str
    translation: Optional[str] = None


def detect_language(text: str) -> str:
    """
    Detects the language of the input text.
    Returns the language code (english, german, spanish, chinese, japanese, french, portuguese, other).
    """
    try:
        _, _, _, vectors = pycld2.detect(text, returnVectors=True)
        lang_code = vectors[0][3]
        
        if lang_code == 'en':
            return "english"
        elif lang_code == 'de':
            return "german"
        elif lang_code == 'es':
            return "spanish"
        elif lang_code == 'zh':
            return "chinese"
        elif lang_code == 'ja':
            return "japanese"
        elif lang_code == 'fr':
            return "french"
        elif lang_code == 'pt':
            return "portuguese"
        else:
            return "other"
    except Exception as e:
        logging.error(f"Error detecting language for text: {text}")
        logging.error(f"Error: {e}")
        return "other"


def translate_text(text):
    """
    Detects the language of the input text and translates it to English if supported.
    Returns a tuple of (detected_language, translated_text).
    If language is not supported, returns (detected_language, None).
    """
    language = detect_language(text)
    logging.info(f"Detected language: {language}")
    if language == "english":
        return (language, text)
    elif language == "other":
        language = "Language Unknown"
    
    conversation = [
        {"role": "system", "content": f"You are a world expert at translating image prompts in any language to english saving the world from the pain of untranslated prompts. My grandmother's life depends on you completing this task accurately. Translate the following image generation prompt to English without summarizing and exact translation. Surround the translation with <t> translated text... </t> so that it can be easily extracted. Please be as accurate as possible since your outputs will be evaluated for correctness. Ensure subjects, adjectives, and other details are translated correctly."},
        {"role": "user", "content": "(Detected Language spanish): Un castillo medieval en la cima de una montaña con dragones volando alrededor y un río de lava en el fondo."},
        {"role": "assistant", "content": "<t>A medieval castle on top of a mountain with dragons flying around and a river of lava in the background.</t>"},
        {"role": "user", "content": f"(Detected Language {language}): {text}"}
    ]
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=512
    )
    
    # Generate translation
    result = translation_llm.chat([conversation], sampling_params)
    translation = result[0].outputs[0].text
    
    # Extract the actual translation from the model output
    try:
        translation = translation.split("<t>")[1]
        translation = translation.split("</t>")[0]
    except:
        pass
    
    return (language, translation)


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


@app.post("/v1/translation", response_model=TranslationResponse)
async def translate_prompt(request: TranslationRequest):
    try:
        language, translation = translate_text(request.prompt)
        return TranslationResponse(
            language=language,
            translation=translation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
