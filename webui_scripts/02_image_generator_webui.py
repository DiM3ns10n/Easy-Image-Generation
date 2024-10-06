import torch
from diffusers import DiffusionPipeline, AutoencoderKL, StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers import KDPM2DiscreteScheduler, DPMSolverMultistepScheduler, KDPM2AncestralDiscreteScheduler, EulerAncestralDiscreteScheduler, HeunDiscreteScheduler
import streamlit as st
import numpy as np
import gc
import os

import ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain_ollama.llms import OllamaLLM

from load_model import load_model, detect_model_type
from prompt_enhancer import enhance_prompt
from generate_images import generate


ollama.pull('llama3')
## Ollama setup
prompt_enhancer_model = OllamaLLM(model= 'llama3:latest')
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)

if cwd.endswith("Image_Gen"):
    parent_dir = cwd


import os

# Set Hugging Face cache directory to a custom path
os.environ["HUGGINGFACE_HUB_CACHE"] = parent_dir

model_dir = os.path.join(parent_dir, "models")
lora_dir = os.path.join(parent_dir, "loras")

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(lora_dir):
    os.makedirs(lora_dir)

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Memory Cleaner
def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()

# Set the title and description of the web interface
st.title("Image Generator")
st.text("This is a simple web interface to generate images using the diffusion model.")

# Get the list of models in the models directory
model_file_names = os.listdir(model_dir)
# model_file_names.append('Add another model from Hugging Face')

option = st.selectbox(
     'which model do you want to use?',
     (model_file_names))

model_name = option
st.write('You selected:', option)
    # st.write('model path:', model_name)

# Get and display prompt, negative prompt, clip skip, width, height, and number of steps
col1, col2 = st.columns(2)

with col1:
    use_lora = st.radio("Do you want to use LORA? (CivitAi)", options=['Yes', 'No'], key="use_lora", index = 1)

with col2:
    use_vae = st.radio("Do you want to use VAE?", options=['Yes', 'No'], key="use_vae", index = 1)

use_lora = True if use_lora == 'Yes' else False
use_vae = True if use_vae == 'Yes' else False


## Example LORA
## https://civitai.com/api/download/models/159686?type=Model&format=SafeTensor

if use_lora:
    lora_file_names = os.listdir(lora_dir)
    st.text("You have selected to use LORA")
    lora_option = st.selectbox(
        'which LORA model do you want to use?',
        (lora_file_names))
    
    lora_name = lora_option
    st.write('You selected:', lora_option)

if use_vae:
    st.write('Use VAE from HuggingFace')
    vae_name = st.text_input("Enter the VAE model name", key="vae_name")
    use_subfolder_vae = st.radio("Do you want to use a subfolder for the VAE model?", options=['Yes', 'No'], key="vae_subfolder_radio", index = 1)
    use_subfolder_vae = True if use_subfolder_vae == 'Yes' else False
    if use_subfolder_vae:
        vae_subfolder = st.text_input("Enter the VAE model subfolder", key="vae_subfolder")
    else:
        vae_subfolder = "" # Default value

prompt = st.text_area("Enter the prompt", key="prompt", value="A beautiful sunset over the ocean.")
negative_prompt = st.text_area("Enter the negative prompt", key="negative_prompt", value="low resolution.")



# Create a 5-column layout for width height steps clip_skip and lora_scale
col1,col2,col3,col4, col5 = st.columns(5)

with col1:
   width = st.text_input('Width', key='width', value=512)
with col2:
   height = st.text_input('Height', key='height', value=512)
with col3:
   steps = st.text_input('Number of steps', key='steps', value=30)
with col4:
    clip_skip = st.text_input('clip skip', key='clip_skip', value=0)
if use_lora:
    with col5:
        lora_scale = st.text_input('LORA scale', key='lora_scale', value=0.5)

## Prompt Enhancement
if 'enhanced_positive_prompt' not in st.session_state:
    st.session_state.enhanced_positive_prompt = None

if 'enhanced_negative_prompt' not in st.session_state:
    st.session_state.enhanced_negative_prompt = None

if 'is_enhanced' not in st.session_state:
    st.session_state.is_enhanced = False

if st.button('enhance_prompt', key='enhance_prompt'):
    enhance_prompt(prompt, prompt_enhancer_model, model_name)

# Store the prompt, negative prompt, width, height, steps, and clip_skip in the session state
st.write("Prompt:", st.session_state.prompt)
st.write("Negative Prompt:", st.session_state.negative_prompt)

if st.session_state.is_enhanced:
    enhanced_prompt = st.session_state.enhanced_positive_prompt
    enhanced_negative_prompt = st.session_state.enhanced_negative_prompt
    st.write("Enhanced prompt:",enhanced_prompt)
    st.write("Enhanced negative prompt:", enhanced_negative_prompt)


use_enhanced_prompt = st.radio("Use enhanced prompt", options=['Yes', 'No'], key="use_enhanced_prompt", index = 1 )
use_enhanced_prompt = True if use_enhanced_prompt == 'Yes' else False


## Generate Image    
if st.button("Generate Image"):
    clear_cache()
    
    model_path = os.path.join(model_dir, model_name.replace("/", "_"))
    
            
    pipeline = load_model(model_path)
    
     # VAE
    if use_vae:
        vae = AutoencoderKL.from_pretrained(vae_name, subfolder=vae_subfolder)
        pipeline.vae = vae

    # LORA
    if use_lora:
        lora_path = os.path.join(lora_dir, lora_name.replace(" ", "_"))
        pipeline.load_lora_weights(lora_path, scale=float(lora_scale))
    
    # Use enhanced prompt if selected 
    if use_enhanced_prompt:
        prompt = enhanced_prompt
        negative_prompt = enhanced_negative_prompt

    image = generate(
                     pipeline, 
                     width, height, int(steps), clip_skip, 
                     prompt, negative_prompt, 
                     )


    # st.text(st.session_state.is_enhanced)

    st.image(image)
    
    clear_cache()