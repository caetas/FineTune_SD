from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

@st.cache_resource
def load_model():
    
    base_model_path = "runwayml/stable-diffusion-v1-5"
    controlnet_path = "./../../models/checkpoint-10000/controlnet"

    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16, use_safetensors=True)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path, controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True, safety_checker = None
    )

    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # remove following line if xformers is not installed
    pipe.enable_xformers_memory_efficient_attention()

    pipe.enable_model_cpu_offload()

    return pipe

@st.cache_data
def generate_image(_pipe, control_image, guidance_scale, controlnet_conditioning_scale, prompt):
    image = pipe(prompt, num_inference_steps=30, image=control_image, guidance_scale=guidance_scale, controlnet_conditioning_scale=controlnet_conditioning_scale).images[0]
    return image

pipe = load_model()
# create title 
st.title("Stable Diffusion ControlNet Demo for Pokemon Generation")

# create drawable canvas that is black, 512x512, and that you can draw in white
canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=10,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=700,
    height=700,
    drawing_mode="freedraw",
    key="canvas",
)

# add a slider to control guidance scale
guidance_scale = st.slider("Guidance Scale", min_value=0.0, max_value=10.0, value=7.5, step=0.5)
# add a slider to control the controlnet_conditioning_scale
controlnet_conditioning_scale = st.slider("ControlNet Conditioning Scale", min_value=0.0, max_value=1.0, value=0.8, step=0.1)
# add a box to enter the prompt
prompt = st.text_input("Prompt", value="a pokemon that looks like a dragon")

# get the image from the canvas when you click the button
if st.button('Generate Pokemon'):
    if canvas_result.image_data is not None:
        control_image = canvas_result.image_data.copy()
        control_image = Image.fromarray(control_image)
        # resize the Pillow image to 512x512
        control_image = control_image.resize((512, 512))
        image = generate_image(pipe, control_image, guidance_scale, controlnet_conditioning_scale, prompt)
        st.image(image, caption="Generated Pokemon", use_column_width=True)