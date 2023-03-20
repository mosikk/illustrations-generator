from PIL import Image
from typing import List, Tuple

import gradio as gr

from text_splitters import split_text
from prompt_generators import get_prompt_generator
from models import get_model


def handle(
        text: str,
        pictures_num: int,
        prompt_generating_method: str,
        generator_type: str,
        num_inference_steps: int,
        guidance_scale: float,
        progress=gr.Progress(),
) -> List[Tuple[Image.Image, str]]:
    progress(0, 'Splitting text')
    text_splitted = split_text(text, pictures_num)
    progress(0, 'Initializing prompt generator')
    generate_prompt = get_prompt_generator(prompt_generating_method)
    progress(0, 'Initializing stable diffusion')
    model = get_model(generator_type)

    result = list()
    prev_latents = None
    for sample in progress.tqdm(text_splitted, desc='Generating pictures'):
        prompt = generate_prompt(sample)
        print(prompt)
        if generator_type == 'Stable Diffusion with latents inheritance':
            response = model(
                prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, latents=prev_latents,
            )
            illustration = response.images[0]
            prev_latents = response.latents
        else:
            illustration = model(
                prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps
            ).images[0]
        result.append((illustration, prompt))
    return result
