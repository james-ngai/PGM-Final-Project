from torchmetrics.functional.multimodal import clip_score
from functools import partial
from diffusers import DiffusionPipeline
import torch


model_ckpt = ""

# not float16
pipe = DiffusionPipeline.from_pretrained.from_pretrained(model_ckpt, torch_dtype=torch.float16).to("cuda")

# load prompt
prompts = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# prompts is a list of prompt


images = pipe(prompts, num_images_per_prompt=1, output_type="np").images

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def calculate_clip_score(images, prompts):
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)

sd_clip_score = calculate_clip_score(images, prompts)
print(f"CLIP score: {sd_clip_score}")
# CLIP score: 35.7038


