import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from torch import Tensor

from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

from torch_utils import distributed as dist


def log_normal(num, P_mean, P_std, max_val, min_val):
    rest_num = num
    sigmas = []
    while rest_num > 0:
        sigma = (torch.randn((rest_num, ), device='cuda') * P_std + P_mean).exp()
        valid_sigma = sigma[(min_val < sigma) & (sigma < max_val)]
        sigmas.append(valid_sigma.view(-1, 1))
        rest_num -= valid_sigma.shape[0]
    sigmas = torch.concat(sigmas, dim=0)
    assert sigmas.shape[0] == num
    return sigmas


class ECALoss:
    def __init__(
        self, 
        t_max: int = 800,
        T: int = 1000,
        sigma_init: float = 80.0,
        offset_dfun: str = 'phuber',
        wgt_type: str = 'nogwt',
        phuber_c: float = .1,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        sigma_dist: str = 'log_normal',
        p_lognormal: float = .6, 
        rewrad_loss_scale: float = 0.0, 
        cfg_scale: float = 4.5,
        loss_type: str = 'eca',
    ) -> None:
        self.t_max = t_max
        self.T = T
        self.sigma_init = torch.tensor([sigma_init]).float().view(1, 1, 1, 1)
        
        self.offset_dfun = offset_dfun
        self.wgt_type = wgt_type
        self.phuber_c = phuber_c

        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_dist = sigma_dist
        self.p_lognormal = p_lognormal

        self.rewrad_loss_scale = rewrad_loss_scale
        self.cfg_scale = cfg_scale
        self.loss_type = loss_type

        # For ECA
        self.k = 8.0
        self.b = 2.0
        self.q = 256
        self.stage = 1
    
    def t_to_r_sigmoid(self, t):
        adj = 1 + self.k * torch.sigmoid(-self.b * t)
        decay = 1 / self.q ** (self.stage+1)
        ratio = 1 - decay * adj
        r = t * ratio
        return torch.clamp(r, min=0)
        
    def sample_sigma(self, net, num: int) -> Tensor: 
        device = 'cuda'
        P_mean = self.P_mean
        P_std = self.P_std
        if isinstance(net, DDP):
            M = net.module.M
            u = net.module.u 
        else:
            M = net.M
            u = net.u
        
        num_steps = self.T
        sigma_max = u[0]
        sigma_min = u[M-1]
        if self.sigma_dist == 'log_normal':
            sigma = log_normal(num, P_mean, P_std, sigma_max, sigma_min)
        elif self.sigma_dist == 'sampling':
            rho = 7
            step_indices = torch.randint(0, self.t_max - 1, (num,), device=device)
            sigma = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        elif self.sigma_dist == 'uniform':
            idx = torch.randint(0, M, (num,), device=device)
            sigma = u[idx]
        else:
            raise NotImplementedError(f"sigma_dist {self.sigma_dist} not implemented")
        
        return sigma
    
    def __call__(self, stu_net, tch_net, vae, reward_model, batch, vae_decode_fn=None, rm_preprocess_fn=None):
        device = 'cuda'
        C = 4
        H = W = 64

        prompt_emb = batch['prompt_emb'].to(device)
        masks = batch['masks'].to(device)
        B = prompt_emb.shape[0]
        
        # Generate img from model
        sigma_init = self.sigma_init.view(1, 1, 1, 1).repeat(B, 1, 1, 1).to(device)
        noisy_x = torch.randn(B, C, H, W, device=device) * sigma_init
        D_yn = stu_net(noisy_x, prompt_emb, sigma_init, mask=masks)
        img = D_yn.clone().detach()
        
        # Sample timesteps/noise levels
        t = self.sample_sigma(stu_net, B).view(-1, 1, 1, 1)
        t_idx = stu_net.module.round_sigma_idx(t, clamp_idx=True)
        r_idx = t_idx + 20

        dtype = t.dtype
        t, r = stu_net.module.idx_to_sigma(t_idx, dtype), stu_net.module.idx_to_sigma(r_idx, dtype)
        dist.print0('t', t_idx.flatten(), t.flatten())
        dist.print0('r', r_idx.flatten(), r.flatten())
        assert (t - r).min() > 0

        # Consistency Regularization
        x_t_grad = img + torch.randn_like(img) * t
        with torch.no_grad():
            x_t = x_t_grad.clone().detach()

        # Shared Dropout Mask
        rng_state = torch.cuda.get_rng_state()
        D_yt = stu_net(x_t, prompt_emb, t, mask=masks)

        with torch.no_grad():
            # D_teacher = tch_net(x_t, prompt_emb, t, mask=masks)
            # x_r = x_t + (r - t) * (x_t - D_teacher) / t

            eps_tch = tch_net(x_t, prompt_emb, t, mask=masks, return_eps=True)
            x_r = x_t + (r - t) * eps_tch

        torch.cuda.set_rng_state(rng_state)
        with torch.no_grad():
            D_yr = stu_net(x_r, prompt_emb, r, mask=masks)
            D_yr = torch.nan_to_num(D_yr)

        # L2 Loss
        loss = (D_yt - D_yr) ** 2
        loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1)

        wt = 1 / t**2
        loss = loss * wt.flatten()

        # Compute reward
        if reward_model is not None:
            image = vae_decode_fn(D_yt, vae)
            image = rm_preprocess_fn(image)
            rewards = -self.rewrad_loss_scale * reward_model.score_gard(batch["rm_input_ids"], batch["rm_attention_mask"], image)
        else:
            rewards = torch.zeros_like(sigma_init)

        # Adaptive Weighting
        # if self.phuber_c > 0:
        #     loss = torch.sqrt(loss + self.phuber_c ** 2) - self.phuber_c
        # else:
        #     loss = torch.sqrt(loss)

        loss = rewards.view(-1,) + loss   
        return loss 