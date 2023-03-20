import copy
import torch
from tqdm import tqdm
from typing import List, Optional, Union
from PIL import Image

from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import PNDMScheduler


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class MyStableDiffusionResponse:
    def __init__(self, images: List[Image.Image], latents: torch.FloatTensor):
        self.images = images
        self.latents = latents


class MyStableDiffusionPipeline:
    def __init__(
            self,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            scheduler: PNDMScheduler,
            unet: UNet2DConditionModel,
            vae: AutoencoderKL,
    ):
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.unet = unet
        self.vae = vae

    def __call__(
            self, prompt, height=512, width=512, num_inference_steps=50,
            generator=None, guidance_scale=7.5, latents=None, save_latents_step=15,
    ):
        # get text embeddings of the prompt
        text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors="pt")
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]

        # unconditional text embeddings for classifier-free guidance
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""], padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(DEVICE))[0]

        # concatenate both embeddings to avoid double-forward pass
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # initialise latents if they were not given
        if latents is None:
            latents = torch.randn((1, self.unet.in_channels, height // 8, width // 8))

        latents = latents.to(DEVICE)

        # init scheduler
        self.scheduler.set_timesteps(num_inference_steps - 1)
        latents = latents * self.scheduler.init_noise_sigma

        latents_saved = None

        # denoising
        for t in tqdm(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            if t == save_latents_step:
                latents_saved = copy.deepcopy(latents)

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample

        # postprocess image
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        response = MyStableDiffusionResponse(pil_images, latents_saved)
        return response

    def to(self, device: str):
        self.vae = self.vae.to(device)
        self.text_encoder = self.text_encoder.to(device)
        self.unet = self.unet.to(device)
        return self


class ModelsStorage:
    def __init__(self):
        self._loaded_models = dict()
        self._gpu_model_name = None

    def _load_stable_diffusion(self, model_name: str) -> None:
        if model_name == 'sd_v1.4':
            self._loaded_models[model_name] = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        elif model_name == 'sd_v2.1':
            self._loaded_models[model_name] = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
        elif model_name == 'sd_latent_inheritance':
            text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            scheduler = PNDMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
            unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
            vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

            vae = vae.to(DEVICE)
            text_encoder = text_encoder.to(DEVICE)
            unet = unet.to(DEVICE)

            self._loaded_models[model_name] = MyStableDiffusionPipeline(
                text_encoder, tokenizer,
                scheduler, unet, vae,
            )

        else:
            raise RuntimeError(f"Can't load model: {model_name}")

    def get_stable_diffusion(self, version: str) -> Optional[Union[StableDiffusionPipeline, MyStableDiffusionPipeline]]:
        model_name = f'sd_{version}'
        if model_name not in self._loaded_models.keys():
            # model is not loaded
            self._load_stable_diffusion(model_name)

        if self._gpu_model_name != model_name:
            # requested model is not on gpu
            if self._gpu_model_name is not None:
                self._loaded_models[self._gpu_model_name] = self._loaded_models[self._gpu_model_name].to('cpu')
                torch.cuda.empty_cache()

            self._loaded_models[model_name] = self._loaded_models[model_name].to(DEVICE)
            self._gpu_model_name = model_name
        return self._loaded_models[model_name]


ACTIVE_MODELS = ModelsStorage()


# public function: get model
def get_model(version: str) -> Optional[StableDiffusionPipeline]:
    if version == 'Stable Diffusion v1.4':
        return ACTIVE_MODELS.get_stable_diffusion(version='v1.4')
    elif version == 'Stable Diffusion v2.1':
        return ACTIVE_MODELS.get_stable_diffusion(version='v2.1')
    elif version == 'Stable Diffusion with latents inheritance':
        return ACTIVE_MODELS.get_stable_diffusion(version='latent_inheritance')
    else:
        raise RuntimeError(f"Can't load model: {version}")
