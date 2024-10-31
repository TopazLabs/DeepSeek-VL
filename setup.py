from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="deepseek_vl",
    version="0.1.0",  # This version will be used in the release tag
    packages=find_packages(),
    install_requires=required,
    extras_require={
        "gradio": ["gradio>=4.0.0"],
    },
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