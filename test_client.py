import requests
import base64
import argparse


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost", help="Host for the API server")
    parser.add_argument("--port", type=int, default=8000, help="Port number for the API server")
    parser.add_argument("--image", type=str, default="images/buildings.png", help="Path to the image file")
    args = parser.parse_args()

    # Convert image to base64
    image_data = image_to_base64(args.image)

    # Make request
    response = requests.post(
        f"http://localhost:{args.port}/v1/caption",
        json={
            "image": image_data,
            "prompt": "<TOPAZ AUTO CLIP CAPTION> Caption this image.",
            "max_new_tokens": 120,
        },
    )

    print(response.json()["caption"])
    
    response = requests.post(
        f"http://localhost:{args.port}/v1/translation",
        json={"prompt": "Ein mittelalterliches Schloss auf einem Berg mit Drachen, die herumfliegen, und einem Fluss aus Lava im Hintergrund."},
    )
    
    print(response.json())


if __name__ == "__main__":
    main()
