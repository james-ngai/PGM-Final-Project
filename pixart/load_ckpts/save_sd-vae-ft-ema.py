from diffusers.models import AutoencoderKL

# Define VAE model name
vae_model = "stabilityai/sd-vae-ft-ema"

# Load the VAE model
vae = AutoencoderKL.from_pretrained(vae_model)

# Define local directory to save the VAE model
save_directory = '/usr3/hcontant/pixart-project-recent/ckpts/pixart/sd-vae-ft-ema'

# Save the VAE model locally
vae.save_pretrained(save_directory)
