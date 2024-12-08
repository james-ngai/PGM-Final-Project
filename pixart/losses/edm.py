import torch
from diffusers import AutoencoderKL
from transformers import T5EncoderModel, AutoTokenizer
import torch_utils.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


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

class iDDPMLoss:
    """
    only linear scheduler for now
    """
    def __init__(
        self, 
        sigma_init: float = 2.5,
        T: int = 1000,
        t_max: int = 800,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        sigma_dist: str = 'log_normal',
        p_lognormal: float = .6,
        wgt_type: str = 'nowgt'
    ) -> None:        
        self.sigma_init = torch.tensor([sigma_init]).view(1, 1, 1, 1)
        self.T = T
        self.t_max = t_max 
     
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_dist = sigma_dist
        self.p_lognormal = p_lognormal
        self.wgt_type=wgt_type
    

    def sample_sigma(self, net, num: int): 
        device = "cuda"
        P_mean = self.P_mean
        P_std = self.P_std
        if isinstance(net, DDP):
            M = net.module.M 
            u = net.module.u 
        else:
            M = net.M
            u = net.u
        # tensor(0.0100) tensor(156.6155)
        
        num_steps = self.T
        sigma_max = u[0]
        sigma_min = u[M-1]
        rho = 7
        if self.sigma_dist == 'uniform':
            sigma = u[torch.randint(0, M, (num, ), device=device)] 
        elif self.sigma_dist == 'log_normal':
            sigma = log_normal(num, P_mean, P_std, sigma_max, sigma_min)
        elif self.sigma_dist == 'sampling':
            step_indices = torch.randint(0, self.t_max - 1, (num,), device=device)
            sigma = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        else:
            raise NotImplementedError
            
        return sigma


    def __call__(self, stu_net, fake_net, batch):
        """
        PixArt-alpha loss
        """       
        device = 'cuda'
        # texts 
        encoder_hidden_states = batch["encoder_hidden_states"].to(device)
        masks = batch['masks'].to(device)
        
        B = encoder_hidden_states.shape[0]
        C = 4
        H = W = 64
        
        # get generator score
        latents = torch.randn(B, C, H, W, device=device)
        sigma_init = self.sigma_init.repeat(B, 1, 1, 1).to(device)
        n = latents * sigma_init
        with torch.no_grad():
            stu_net.eval()
            D_yn = stu_net(
                x=n, 
                encoder_hidden_states=encoder_hidden_states, 
                sigma=sigma_init, 
                mask=masks
            )
            stu_net.train()
            
        # predict generator score
        sigma = self.sample_sigma(stu_net, B).view(-1, 1, 1, 1)
        
        fake_D_yn = fake_net(
            x=D_yn + torch.randn_like(D_yn) * sigma,
            encoder_hidden_states=encoder_hidden_states,
            sigma=sigma,
            mask=masks,
        )
        if self.wgt_type == 'nowgt':
            weights = 1.0 
        elif self.wgt_type == 'orig':
            weights = (1 / sigma ** 2).reshape(-1, 1, 1, 1)
        
        loss = weights * ((fake_D_yn - D_yn) ** 2)
        
        
        return loss
