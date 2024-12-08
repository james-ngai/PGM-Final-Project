from .PixArt import PixArt, PixArt_XL_2
from .PixArtMS import PixArtMS_XL_2

import os 
import torch 
from diffusers import AutoencoderKL
from transformers import T5EncoderModel, AutoTokenizer

from torch_utils import distributed as dist

def load_raw_pixart(**kwargs):
    return PixArt_XL_2(**kwargs)

def load_raw_pixart_ms(**kwargs):
    return PixArtMS_XL_2(**kwargs)

class PixArt_alpha_DDIM(torch.nn.Module):
    
    def __init__(self, 
        img_channels    = 4,                # Number of color channels.
        C_1             = 0.001,            # Timestep adjustment at low noise levels.
        C_2             = 0.008,            # Timestep adjustment at high noise levels.
        M               = 1000,             # Original number of timesteps in the DDPM formulation.
        beta_start      = .0001,
        beta_end        = .02,
        pretrained      = True,
        ckpt_path       = "path/to/pixart",
        load_encoders   = False,
        load_vae        = False,
        ms              = False,
        **model_kwargs
    ):
        super().__init__()
        self.img_channels = img_channels
        self.C_1 = C_1
        self.C_2 = C_2
        self.M = M
        self.beta_start = beta_start
        self.beta_end = beta_end
        model = load_raw_pixart(**model_kwargs) if not ms else load_raw_pixart_ms(**model_kwargs)
        self.ckpt_dir = os.path.dirname(ckpt_path)

        u = torch.zeros(M + 1)
        for j in range(M, 0, -1): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (1 - self.beta_t(j)).clip(min=C_1) - 1).sqrt()
        self.register_buffer('u', u)
        # (M-1, 0): tensor(0.0100) tensor(156.6155)

        if load_encoders:
            self.load_encoders()

        if pretrained:
            print("Loading pretrained model...")
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt['state_dict'])

        # disable dropout in caption
        model.y_embedder.uncond_prob = 0
        self.model = model
        
    def load_encoders(self):
        print("Loading text encoder...")
        if load_text:
            text_encoder = T5EncoderModel.from_pretrained(
                os.path.join(self.ckpt_dir, "t5-v1_1-xxl")
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(self.ckpt_dir, "t5-v1_1-xxl")
            )
            text_encoder.eval().requires_grad_(False)
            self.text_encoder = text_encoder

    def load_vae(self, load_text: bool = True, load_vae: bool = True):
        vae = AutoencoderKL.from_pretrained(
            # os.path.join(self.ckpt_dir, "sd-vae-ft-ema")
            # Hardcoded result
            '/usr3/hcontant/pixart-project-recent/ckpts/pixart/sd-vae-ft-ema'
        )
        vae.eval().requires_grad_(False)
        return vae

    @torch.no_grad()
    def encode_prompts(self, prompts, device=torch.device('cuda')):
        token_and_mask = self.tokenizer(
            prompts, 
            max_length=120,# self.tokenizer.model_max_length, 
            padding="max_length", 
            truncation=True, 
            return_attention_mask=True,
            return_tensors="pt",
            add_special_tokens=True
        )
        input_ids = token_and_mask.input_ids.to(device)
        mask = token_and_mask.attention_mask.to(device)

        encoder_hidden_states = self.text_encoder(
            input_ids=input_ids,
            attention_mask=mask,
        )['last_hidden_state'].detach()
        
        encoder_hidden_states = encoder_hidden_states.unsqueeze(dim=1)
        
        return encoder_hidden_states, mask

    @torch.no_grad()
    def decode_latents(self, latents):
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)

        return image

    def _decode_latents(self, latents):
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)

        return image

    def forward(self, x, encoder_hidden_states, sigma=None, mask=None, return_eps=False):
        latents = x 
        device = 'cuda'

        rnd_idx = self.round_sigma(sigma, return_index=True).reshape(-1,)
        sigma = sigma.reshape(-1, 1, 1, 1)
        
        # iDDPM+DDIM preconditioning
        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = self.M - 1 - rnd_idx

        F_x = self.model(
                x=c_in * latents, 
                timestep=c_noise,
                y=encoder_hidden_states,
                mask=mask,
            )
        D_x = c_skip * latents + c_out * F_x[:, :self.img_channels]

        if return_eps:
            return F_x[:, :self.img_channels]
        return D_x
    
    def forward_with_cfg(self, x, encoder_hidden_states, sigma=None, cfg_scale=4.5, mask=None, rescale=False):
        latents = x 
        device = 'cuda'
        if sigma is None:
            rnd_j = torch.randint(0, self.M - 1, (latents.shape[0],), device=device)
            sigma = self.u[rnd_j].reshape(-1, 1, 1, 1)
        else:
            rnd_j = self.round_sigma(sigma, return_index=True).reshape(-1,)
        sigma = sigma.reshape(-1, 1, 1, 1)
        
        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = self.M - 1 - rnd_j
        
        F_x = self.model.forward_with_cfg(
            x=c_in * latents, 
            timestep=c_noise,
            y=encoder_hidden_states,
            cfg_scale=cfg_scale,
            mask=mask
        ) 
    
        D_x = c_skip * latents + c_out * F_x[:, :self.img_channels]

        return D_x

    def beta_t(self, j):
        j = torch.as_tensor(j)
        return self.beta_end + (self.beta_start - self.beta_end) * j / self.M

    def round_sigma(self, sigma, return_index=False):
        sigma = torch.as_tensor(sigma)
        index = torch.cdist(sigma.to(self.u.device).to(torch.float32).reshape(1, -1, 1), self.u.reshape(1, -1, 1).to(torch.float32)).argmin(2)
        result = index if return_index else self.u[index.flatten()].to(sigma.dtype)
        return result.reshape(sigma.shape).to(sigma.device)

    def round_sigma_idx(self, sigma, clamp_idx=False):
        sigma = torch.as_tensor(sigma)
        index = torch.cdist(sigma.to(self.u.device).to(torch.float32).reshape(1, -1, 1), self.u.reshape(1, -1, 1).to(torch.float32)).argmin(2)
        index = torch.clamp(index, min=0, max=len(self.u)-1-20) if clamp_idx else index
        return index.reshape(sigma.shape).to(sigma.device)
    
    def idx_to_sigma(self, idx, dtype):
        idx = torch.clamp(idx, min=0, max=len(self.u)-1)
        return self.u[idx].to(dtype)

