from diffusers import DiffusionPipeline

# Specify the model name
model_name = "PixArt-alpha/PixArt-XL-2-512x512"
save_directory = "/usr3/hcontant/pixart-project-recent/ckpts/pixart"

# Load the diffusion model
pipe = DiffusionPipeline.from_pretrained(model_name)

# Save the model locally
pipe.save_pretrained(save_directory)

print(f"Model saved to {save_directory}")
