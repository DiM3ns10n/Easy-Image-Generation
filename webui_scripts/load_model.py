from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import torch
import os

def detect_model_type(model_path):
    """
    Detect if the model is from Hugging Face or CivitAI, and whether it's an XL model.
    
    Args:
    - model_path: Either a directory path (for CivitAI) or a Hugging Face model ID.

    Returns:
    - model_type: 'huggingface' or 'civitai'
    - is_xl: Boolean indicating if it's an XL model
    """
    if model_path.endswith(".safetensors"):
        # Assume it's a CivitAI model if it's a directory containing .safetensors files
        print("Detected model from CivitAI...")
        is_xl = "xl" in model_path.lower()  # Detect based on filename
        model_type = "civitai"
    else:
        # Assume it's from Hugging Face if it's a model ID string
        print("Detected model from Hugging Face...")
        is_xl = "xl" in model_path.lower() or "sdxl" in model_path.lower()  # Check for "xl" in model name
        model_type = "huggingface"
    
    return model_type, is_xl


def load_model(model_path):
    """
    Loads a diffusion model dynamically, detecting the model source and whether it's an XL model.
    
    Args:
    - model_path: Path to the model directory or identifier (for Hugging Face models).

    Returns:
    - A loaded StableDiffusionPipeline or StableDiffusionXLPipeline object.
    """
    model_type, is_xl = detect_model_type(model_path)

    if model_type == "huggingface":
        if is_xl:
            print("Loading SDXL model from Hugging Face...")
            pipe = StableDiffusionXLPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        else:
            print("Loading non-XL model from Hugging Face...")
            pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    
    elif model_type == "civitai":
        safetensor_file = model_path
        if is_xl:
            print("Loading SDXL model from CivitAI (.safetensors)...")
            pipe = StableDiffusionXLPipeline.from_single_file(safetensor_file, torch_dtype=torch.float16)
        else:
            print("Loading non-XL model from CivitAI (.safetensors)...")
            pipe = StableDiffusionPipeline.from_single_file(safetensor_file, torch_dtype=torch.float16)
    
    else:
        raise ValueError("Unknown model type! Use 'huggingface' or 'civitai'.")
    
    # Move to GPU if available
    # pipe = pipe.to("cuda")
    
    return pipe