import torch
from typing import Optional

from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import PNDMScheduler


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class ModelsStorage:
    def __init__(self):
        self._loaded_models = dict()
        self._gpu_model_name = None

    def _load_stable_diffusion(self, model_name: str) -> None:
        if model_name == 'sd_v1.4':
            self._loaded_models[model_name] = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        elif model_name == 'sd_v2.1':
            self._loaded_models[model_name] = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
        else:
            raise RuntimeError(f"Can't load model: {model_name}")

    def get_stable_diffusion(self, version: str) -> Optional[StableDiffusionPipeline]:
        model_name = f'sd_v{version}'
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
        return ACTIVE_MODELS.get_stable_diffusion(version='1.4')
    elif version == 'Stable Diffusion v2.1':
        return ACTIVE_MODELS.get_stable_diffusion(version='2.1')
    else:
        raise RuntimeError(f"Can't load model: {version}")
