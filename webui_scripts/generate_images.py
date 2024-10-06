from diffusers import DiffusionPipeline, AutoencoderKL
from diffusers import KDPM2DiscreteScheduler, DPMSolverMultistepScheduler, KDPM2AncestralDiscreteScheduler, EulerAncestralDiscreteScheduler, HeunDiscreteScheduler
import os
import streamlit as st

def generate(pipeline, width, height, 
             steps, clip_skip, prompt, negative_prompt, 
             ):
    
    # remove the safety checker
    pipeline.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

    # setting the scheduler 
    scheduler_config = pipeline.scheduler.config
    scheduler_config['final_sigmas_type'] = 'sigma_min'
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(scheduler_config)
    # pipeline.scheduler = KDPM2DiscreteScheduler.from_config(scheduler_config)
    
    pipeline.enable_sequential_cpu_offload()

    st.text("Generating image...")
    st.text(f"Steps: {steps}")
    progress_bar = st.progress(0)
    def progress(pipe, step, timestep, callback_kwargs):
        p = step/steps * 100
        p = int(round(p, 0))

        if p < 90:
            progress_bar.progress(p, f"Progress: {p}%")
        else:
            text = f'Almost done!' if p%2 == 0 else 'Please wait'
            progress_bar.progress(90, f"Progress: 90%. {text}")
        return callback_kwargs

    image = pipeline( prompt, negative_prompt= negative_prompt, width=int(width), 
                     height=int(height), num_inference_steps=steps, 
                     clip_skip=int(clip_skip), callback_on_step_end = progress
                     ).images[0]
    progress_bar.progress(100, "Progress: 100%. Done!")
    
    return image