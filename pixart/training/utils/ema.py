import torch 
import torch.nn as nn
from copy import deepcopy
import torch_utils.distributed as dist

class ModelEmaV2(nn.Module):
    """
    code from timm
    """
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval().requires_grad_(False)
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class SimpleEMA(nn.Module):
    
    def __init__(self, model, decay=0.9999, device=None):
        super(SimpleEMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model) # .state_dict()
        self.module.eval().requires_grad_(False)
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)
    
    @torch.no_grad()
    def update(self, model):
        ema_beta = self.decay
        for p_ema, p_net in zip(self.module.parameters(), model.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))


class EDMEma(nn.Module):
    
    def __init__(self, model, batch_size, ema_halflife_kimg, ema_rampup_ratio: float = .05, device=None):
        super(EDMEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model) # .state_dict()
        self.module.eval().requires_grad_(False)
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)
        self.batch_size = batch_size * dist.get_world_size() 
        self.ema_halflife_nimg = ema_halflife_kimg * 1000
        self.register_buffer("cur_nimg", torch.tensor(0.))
        self.ema_rampup_ratio = ema_rampup_ratio
    
    @torch.no_grad()
    def update(self, model):
        self.cur_nimg += self.batch_size
        if self.ema_rampup_ratio is not None:
            ema_halflife_nimg = min(self.ema_halflife_nimg, self.cur_nimg.item() * self.ema_rampup_ratio)
        else:
            ema_halflife_nimg = self.ema_halflife_nimg
        ema_beta = 0.5 ** (self.batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(self.module.parameters(), model.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
