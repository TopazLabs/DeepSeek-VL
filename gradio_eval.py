import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import tempfile

import gradio as gr
from swift.llm import (
    get_model_tokenizer,
    get_template,
    inference,
    ModelType,
    get_default_template_type,
)
from swift.utils import seed_everything

from swift.llm import (
    get_model_tokenizer,
    get_template,
    inference,
    ModelType,
    get_default_template_type,
)
from swift.tuners import Swift

from PIL import Image
import torch
from torchvision import transforms

# ----------------------------- Configuration -----------------------------

# Default system prompt
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language."
)

# Path to the LoRA checkpoint
CHECKPOINT_PATH = "/home/topaz_koch/dev/imageunderstanding/SemanticAnalysis/DeepSeek-VL/deepseek_vl_finetuned/deepseek-vl-1_3b-chat/v18-20241031-002822/checkpoint-2400"

# Ensure the checkpoint path exists
if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint path not found: {CHECKPOINT_PATH}")

# Path to example images
EXAMPLES_DIR = "examples"  # Ensure this directory exists with images

# ----------------------------- Model Loading -----------------------------


def load_model(checkpoint_path, device="auto"):
    """
    Load the Swift model with the specified LoRA checkpoint.

    Args:
        checkpoint_path (str): The local path to the LoRA checkpoint.
        device (str): Device to load the model on ('auto', 'cpu', 'cuda').

    Returns:
        model: The loaded model.
        tokenizer: The associated tokenizer.
        template: The template used by the model.
    """
    model_type = ModelType.deepseek_vl_1_3b_chat  # Update to your specific model type
    print(f"Loading model type: {model_type}")

    # Get the default template type for the model
    template_type = get_default_template_type(model_type)
    print(f"Template type: {template_type}")

    kwargs = {}
    kwargs["use_flash_attn"] = True  # Uncomment if you wish to use flash attention

    model_id_or_path = None  # Assuming local model loading

    # Load model and tokenizer
    model, tokenizer = get_model_tokenizer(
        model_type,
        model_id_or_path=model_id_or_path,
        model_kwargs={"device_map": "auto"},
        **kwargs,
    )
    model = Swift.from_pretrained(model, CHECKPOINT_PATH, inference_mode=True)
    model.generation_config.max_new_tokens = 80
    template = get_template(template_type, tokenizer)

    # Modify generation parameters if necessary
    if hasattr(model, "generation_config"):
        model.generation_config.max_new_tokens = 128

    # Get the template
    template = get_template(template_type, tokenizer)

    # Seed for reproducibility
    seed_everything(42)

    print("Model loaded successfully.")
    return model, tokenizer, template


# ----------------------------- Inference Function -----------------------------


def generate_caption(image, system_prompt, user_prompt, model, tokenizer, template):
    """
    Generate a caption for the given image using the Swift model.

    Args:
        image (PIL.Image.Image): The input image.
        system_prompt (str): The system prompt for the model.
        user_prompt (str): The user prompt for the model.
        model: The loaded model.
        tokenizer: The tokenizer associated with the model.
        template: The template used by the model.

    Returns:
        str: The generated caption.
    """
    if model is None:
        return "Model not loaded properly."

    if image is None:
        return "Please provide an image."

    try:

        # tokenizer.to(device)

        # Prepare the prompt
        # Assuming the model can accept image inputs, pass them as additional parameters
        # This depends on how the `inference` function and model handle images
        # If `inference` does not support image parameters, you may need to modify it
        if not isinstance(image, Image.Image):
            return "Invalid image format. Please provide a valid PIL image."

        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp_file:
            image.save(tmp_file.name)
            query = f"<img>{tmp_file.name}</img>{user_prompt}"
            response, history = inference(model, template, query)

        return response

    except Exception as e:
        return f"Error during caption generation: {str(e)}"


# ----------------------------- Gradio Interface -----------------------------


def create_demo(checkpoint_path):
    """
    Create and launch the Gradio interface.

    Args:
        checkpoint_path (str): The local path to the LoRA checkpoint.

    Returns:
        gr.Blocks: The Gradio Blocks interface.
    """
    # Load the model, tokenizer, and template
    model, tokenizer, template = load_model(checkpoint_path)

    # Create Gradio interface using Blocks
    with gr.Blocks() as demo:
        gr.Markdown("# DeepSeek-VL Image Captioning")
        gr.Markdown(
            "Upload an image and generate a descriptive caption using the DeepSeek-VL model."
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload an Image")
                system_prompt = gr.Textbox(
                    value=DEFAULT_SYSTEM_PROMPT, label="System Prompt", lines=4
                )
                user_prompt = gr.Textbox(
                    value="<TOPAZ AUTO CLIP CAPTION> Caption this image.",
                    label="User Prompt",
                )
                submit_btn = gr.Button("Generate Caption")

            with gr.Column(scale=2):
                caption_output = gr.Textbox(label="Generated Caption", lines=10)

        # Set up the generation
        submit_btn.click(
            fn=lambda img, sys_p, usr_p: generate_caption(
                img, sys_p, usr_p, model, tokenizer, template
            ),
            inputs=[image_input, system_prompt, user_prompt],
            outputs=caption_output,
        )

        # Add examples (ensure example images are accessible)
        if os.path.exists(EXAMPLES_DIR):
            example_images = [
                [
                    os.path.join(EXAMPLES_DIR, "image1.jpg"),
                    DEFAULT_SYSTEM_PROMPT,
                    "Caption this image:",
                ],
                [
                    os.path.join(EXAMPLES_DIR, "image2.jpg"),
                    DEFAULT_SYSTEM_PROMPT,
                    "Describe this image in detail:",
                ],
            ]

            gr.Examples(
                examples=example_images,
                inputs=[image_input, system_prompt, user_prompt],
                outputs=caption_output,
                fn=lambda img, sys_p, usr_p: generate_caption(
                    img, sys_p, usr_p, model, tokenizer, template
                ),
                cache_examples=True,
            )
        else:
            gr.Markdown(
                f"⚠️ **Warning:** The examples directory `{EXAMPLES_DIR}` does not exist. Please create it and add example images."
            )

    return demo


# ----------------------------- Main Execution -----------------------------

if __name__ == "__main__":
    # Create and launch the Gradio demo
    demo = create_demo(CHECKPOINT_PATH)
    demo.launch(share=True)
