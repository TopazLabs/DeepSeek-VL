import requests
import base64

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Convert image to base64
image_data = image_to_base64("/home/topaz_koch/dev/imageunderstanding/buildings.png")

# Make request
response = requests.post(
    "http://localhost:8000/v1/caption",
    json={
        "image": image_data,
        "prompt": "<TOPAZ AUTO CLIP CAPTION> Caption this image.",
        "max_new_tokens": 120
    }
)

print(response.json()["caption"])