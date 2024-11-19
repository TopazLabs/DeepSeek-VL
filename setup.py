from setuptools import setup, find_packages

setup(
    name="deepseek_vl",
    version="0.1.0",
    packages=find_packages(include=['deepseek_vl*']),
    install_requires=[
        "transformers==4.46.0",
        "torch",
        "timm>=1.0.11",
        "accelerate==1.0.1",
        "sentencepiece==0.0.2",
        "attrdict==2.0.1",
        "einops==0.8.0",
        "fastapi==0.115.3",
        "uvicorn==0.32.0",
        "python-multipart==0.0.12",
        "pillow==10.4.0",
        "pydantic==2.9.2",
        "click>=8.0.0",
    ],
    extras_require={
        "gradio": ["gradio>=4.0.0"],
    },
    entry_points={
        "console_scripts": [
            "deepseek-vl=deepseek_vl.cli:cli",
        ],
    },
    package_data={
        'deepseek_vl': ['*.py'],
    },
    include_package_data=True,
    author="TopazLabs",
    author_email="service@deepseek.com",
    description="DeepSeek Vision-Language Model",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/TopazLabs/DeepSeek-VL",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
