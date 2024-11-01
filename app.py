import torch
import requests
import tempfile
import gradio as gr

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

# Specify the ID or path to the model
model_path = "TopazLabs/DeepSeek-AutoPrompt"
vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt = MultiModalityCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()


def generate_caption(image):
    """
    Generate a caption for the uploaded image using the DeepSeek model.

    Args:
        image (PIL.Image): The input image.

    Returns:
        str: The generated caption.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp_file:
        image.save(tmp_file.name)

        conversation = [
            {
                "role": "User",
                "content": "<image_placeholder><TOPAZ AUTO CLIP CAPTION> Caption this image.",
                "images": [tmp_file.name],
            },
            {"role": "Assistant", "content": ""},
        ]

        # Load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(vl_gpt.device)

        # Run image encoder to get the image embeddings
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # Run the model to get the response
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )

        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return answer


# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# DeepSeek-AutoPrompt Image Captioning")
    gr.Markdown("Upload an image to generate a descriptive caption.")

    image_input = gr.Image(type="pil", label="Upload an Image")
    caption_output = gr.Textbox(label="Generated Caption", lines=4)

    submit_btn = gr.Button("Generate Caption")
    submit_btn.click(fn=generate_caption, inputs=image_input, outputs=caption_output)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch(share=True)
