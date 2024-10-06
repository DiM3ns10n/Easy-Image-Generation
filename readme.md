# Easy Image Generation

## What can it do?

Simple Image Generator is a program that generates images using StableDiffusion. Though a bit slow, this web app can run StableDiffusion and Stable Diffusion XL models on low VRAM (minimum of 4 GB). No need to worry about which prompts to use, as the prompt enhancer helps create beautiful images from basic input prompts. It also supports the ability to use LoRA (Low-Rank Adaptation).

## Why?

- Simple UI
- Low VRAM usage (can run on GPUs with 4 GB RAM)

## Setup.

1. Install Python 3.11.6
2. Install PyTorch with GPU support:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    ```
3. Install Ollama (necessary for prompt enhancement)
4. Create a new directory called `Image-Generator` and clone this repo into that directory.
5. Create a virtual environment inside the `Image-Generator` directory using a terminal with Python 3.11.6:
    - For Windows:
      ```bash
      virtualenv -p 'path\to\python3.11.6\python.exe' .venv
      ```
    - For Linux:
      ```bash
      virtualenv -p 'path/to/python3.11.6/python' .venv
      ```
6. Activate the virtual environment:
    - For Windows:
      ```bash
      .venv\Scripts\activate
      ```
    - For Linux:
      ```bash
      source .venv/bin/activate
      ```
7. Install the required packages from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
8. Run `image_generator_webui.py` using Streamlit:
    ```bash
    streamlit run Home.py
    ```
