import streamlit as st
import os


def save_api_key(api_key):
    with open('.env', 'w') as f:
        f.write(f"CIVIT_AI_API_KEY = '{api_key}'")
    

# 58e39ab836a0308b4d2e2b592fc7be24
st.write("Add Civitai's api key to login while downloading models")


if  os.getenv('CIVIT_AI_API_KEY'):
    change_key = st.radio('Api key already exists do you want to change it?', options=['Yes', 'No'], key = 'key_exist', index = 1)

    change_key = True if change_key == 'Yes' else False

    if change_key:
        api_key = st.text_input('Enter the api key', key= 'key')
        if st.button('Change API KEY'):
            save_api_key(api_key=api_key)
            st.write('Changed API Key')
    
else:
   api_key = st.text_input('Enter the api key', key= 'key')
   if st.button('submit key'):
        save_api_key(api_key=api_key)
        st.write("Key created")

