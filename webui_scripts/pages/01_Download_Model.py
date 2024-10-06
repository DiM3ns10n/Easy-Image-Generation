import streamlit as st
from load_model import load_model
import os
from diffusers import DiffusionPipeline, AutoencoderKL
import requests
from dotenv import load_dotenv

load_dotenv()
civit_ai_login_key = os.getenv('CIVIT_AI_API_KEY')
  
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)

if cwd.endswith("Image_Gen"):
    parent_dir = cwd

st.markdown("# Download Model and Lora")
## Model
st.markdown("## Model")

st.text('Download a model from HuggingFace')

model_name = st.text_input('Enter the model name from Hugging Face', key='model_name', value="lantianai/anything_v3_full")

model_dir = os.path.join(parent_dir, "models")
model_path = os.path.join(model_dir, model_name.replace("/", "_"))

model_download_status = st.empty() 
if st.button('Download Model'):
    
    if not os.path.exists(model_path):
        with st.spinner('Downloading Model. Please wait...'):
            # model_download_status.write('Downloading Model. Please wait...')
            pipe = DiffusionPipeline.from_pretrained(model_name)
            pipe.save_pretrained(model_path)
        

        if os.path.exists(model_path):
            st.success('Model Downloaded successfully')
            # model_download_status.write('Model Downloaded successfully')
        else:
            model_download_status.write('There was an error')
    else:
        st.write('Model already exist')


## Lora
lora_dir = os.path.join(parent_dir, "loras")
st.markdown("## LORA")

st.text('Download a Lora from Civitai')
st.text_area('Please paste the download link for the LoRA model. Ensure that the correct LoRA URL is copied, and remember to add your CivitAI API key from the keys page to download LoRA models.')

lora_name = st.text_input("Enter the LORA model name", key="lora_name")
lora_url = st.text_input("Enter the LORA model url", key="lora_url")


lora_path = os.path.join(lora_dir, lora_name.replace(" ", "_"))

lora_download_status = st.empty()
if st.button('submit', key='submit_lora'):
  
    if civit_ai_login_key is None or civit_ai_login_key == "":
        st.error("Please add your CivitAI API key to the using the keys page")
        st.stop()
    
    lora_url = f'{lora_url}&token={civit_ai_login_key}'

    if not os.path.exists(lora_path):
        with st.spinner('Downloading Model. Please wait...'):
    
            response = requests.get(lora_url)
            with open(lora_path, 'wb') as f:
                f.write(response.content)
        if os.path.exists(lora_path):
            st.success('LORA model downloaded successfully')
        else:
            lora_download_status.write('There was an error')
            
    else:
            lora_download_status.write("LORA model already exists, Loading LORA model")
