import ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain_ollama.llms import OllamaLLM
import streamlit as st

if 'enhanced_positive_prompt' not in st.session_state:
    st.session_state.enhanced_positive_prompt = None

if 'enhanced_negative_prompt' not in st.session_state:
    st.session_state.enhanced_negative_prompt = None

if 'is_enhanced' not in st.session_state:
    st.session_state.is_enhanced = False


def enhance_prompt(prompt, prompt_enhancer_model, model_name):    
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an experienced prompt engineer who specializes in improving and enhancing given image generation prompt for the ai model {ai_model}. You provide the positive prompt and negative prompt."),
            ("human", "Please enhance and provide positive and negative prompts based on this positive prompt {positive_prompt}."),
        ]
    )

    def get_positive_prompt(prompt):
        positive_prompt_template = ChatPromptTemplate.from_messages(
            [
                ('system', "You are an experienced prompt engineer who specializes in improving and enhancing given image generation prompt for the ai model. You provide the positive prompt and nothing else. You do not provide any other information. Just go directly to the point, no need to tell that this is a positive prompt."),
                ("human", "Given the prompt: {prompt}, get the positive prompt."),

            ]
        )

        return positive_prompt_template.format_prompt(prompt=prompt)

    def get_negative_prompt(prompt):
        negative_prompt_template = ChatPromptTemplate.from_messages(
            [
                ('system', "You are an experienced prompt engineer who specializes in improving and enhancing given image generation prompt for the ai model. You provide the negative prompt and nothing else. You do not provide any other information.Just go directly to the point, no need to tell that this is a negative prompt"),
                ("human", "Given the prompt: {prompt}, get the negative prompt."),

            ]
        )

        return negative_prompt_template.format_prompt(prompt=prompt)

    def combine_prompt(positive_prompt, negative_prompt):
        return {
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt
        }

    positive_branch_chain = RunnableLambda(lambda x : get_positive_prompt(x)) | prompt_enhancer_model | StrOutputParser()
    negative_branch_chain = RunnableLambda(lambda x : get_negative_prompt(x)) | prompt_enhancer_model | StrOutputParser()

    positive_negative_chain = RunnableParallel(branches = {"positive_prompt":positive_branch_chain, "negative_prompt": negative_branch_chain})

    chain = (
        prompt_template
        | prompt_enhancer_model
        | StrOutputParser()
        | positive_negative_chain
        | RunnableLambda(lambda x: combine_prompt(x['branches']["positive_prompt"], x['branches']["negative_prompt"]))
    )
    
    st.session_state.is_enhanced = True
    result = chain.invoke({"positive_prompt": st.session_state.prompt, "ai_model": model_name}) 

    st.session_state.enhanced_positive_prompt = result['positive_prompt']
    st.session_state.enhanced_negative_prompt = result['negative_prompt']

    return st.session_state.enhanced_positive_prompt,  st.session_state.enhanced_negative_prompt
