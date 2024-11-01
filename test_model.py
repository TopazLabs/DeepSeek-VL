import torch
import requests
import tempfile

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images


# specify the ID or path to the model
model_path = "TopazLabs/DeepSeek-AutoPrompt"
vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt = MultiModalityCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

print("Model:", vl_gpt) # Prints the model architecture


# Download image from URL and save to temporary file
image_url = "https://raw.githubusercontent.com/TopazLabs/DeepSeek-VL/14cdab3456c61c1ed67b5a7cd4574ba17958eea0/images/dog_c.png"
response = requests.get(image_url)
response.raise_for_status()

with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp_file:
    tmp_file.write(response.content)
    tmp_file.flush()
    
    conversation = [
        {
            "role": "User", 
            "content": "<image_placeholder><TOPAZ AUTO CLIP CAPTION> Caption this image.",
            "images": [tmp_file.name]
        },
        {
            "role": "Assistant",
            "content": ""
        }
    ]

    # load images and prepare for inputs
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True
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
        max_new_tokens=512,
        do_sample=False,
        use_cache=True
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    print(f"{prepare_inputs['sft_format'][0]}", answer)
