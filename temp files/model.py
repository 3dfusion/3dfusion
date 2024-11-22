from diffusers import DiffusionPipeline
from PIL import Image
import torch
text_pipeline = DiffusionPipeline.from_pretrained(
    "dylanebert/mvdream",
    custom_pipeline="dylanebert/multi-view-diffusion",
    torch_dtype=torch.float16,
    trust_remote_code=True,
).to("cuda")

image_pipeline = DiffusionPipeline.from_pretrained(
    "dylanebert/multi-view-diffusion",
    custom_pipeline="dylanebert/multi-view-diffusion",
    torch_dtype=torch.float16,
    trust_remote_code=True,
).to("cuda")


def create_image_grid(images):
    images = [Image.fromarray((img * 255).astype("uint8")) for img in images]

    width, height = images[0].size
    grid_img = Image.new("RGB", (2 * width, 2 * height))

    grid_img.paste(images[0], (0, 0))
    grid_img.paste(images[1], (width, 0))
    grid_img.paste(images[2], (0, height))
    grid_img.paste(images[3], (width, height))

    return grid_img

def text_to_mv(prompt):
    images = text_pipeline(
        prompt, guidance_scale=5, num_inference_steps=30, elevation=0
    )
    return create_image_grid(images)

text_to_mv("a cat running")