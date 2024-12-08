# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from diffusers import AutoencoderKL
from copy import deepcopy

import torch.distributed
import dnnlib
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.optim.lr_scheduler import LambdaLR
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc

from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

from transformers import T5Tokenizer, T5EncoderModel


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])


@torch.no_grad()
def encode_prompts(prompts, tokenizer, text_encoder, max_length=128):
    token_and_mask = tokenizer(
        prompts, 
        max_length=max_length,
        padding="max_length", 
        truncation=True, 
        return_attention_mask=True,
        return_tensors="pt",
        add_special_tokens=True
    )
    input_ids = token_and_mask.input_ids.to("cuda")
    mask = token_and_mask.attention_mask.to("cuda")

    encoder_hidden_states = text_encoder(
        input_ids=input_ids,
        attention_mask=mask,
        return_dict=False
    )[0]
    
    return encoder_hidden_states, mask

#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for dataset.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    reward_kwargs = {},
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    total_steps         = 200000,   # Training duration, measured in thousands of training images.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    step_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    image_tick          = 10, 
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
    grad_accumulation   = 1,
    ema_kwargs          = {},
    precision           = "fp16",
    resume_pt           = None,
    resume_state_dump   = None,
    resume_step         = 0,
    use_fsdp            = False,
    use_rlhf            = False, 
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    
    num_accumulation_rounds = grad_accumulation

    # TODO: Build a dataloader for the prompt data & encoding
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    dist.print0('Dataset length:', len(dataset_obj))

    dataset_sampler = misc.InfiniteSampler(
        dataset=dataset_obj, 
        rank=dist.get_rank(), 
        num_replicas=dist.get_world_size(), 
        seed=seed
    )
    dataloader = iter(torch.utils.data.DataLoader(
        dataset=dataset_obj,
        sampler=dataset_sampler,
        **data_loader_kwargs
    ))
    dist.print0('Dataset length:', len(dataset_obj))

    # Load reward model
    if use_rlhf:
        dist.print0('Loading Reward Model')
        rm_state_dict = torch.load(reward_kwargs['rm_ckpt_path'], map_location='cpu')
        reward_model = dnnlib.util.construct_class_by_name(device=device, class_name=reward_kwargs['class_name'], med_config=reward_kwargs['med_config'])
        reward_model.load_state_dict(rm_state_dict, strict=False)
        reward_model.eval().requires_grad_(False).to(device)
        del rm_state_dict
    else:
        reward_model = None

    def _transform():
        return Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    rm_preprocess = _transform()

    # Construct network.
    dist.print0('Loading text encoder...')
    tokenizer = T5Tokenizer.from_pretrained(network_kwargs['text_encoder_ckpt_path'])
    text_encoder = T5EncoderModel.from_pretrained(network_kwargs['text_encoder_ckpt_path'], torch_dtype=torch.float16).to('cuda')
    # text_encoder = T5EncoderModel.from_pretrained(network_kwargs['text_encoder_ckpt_path']).to('cuda')
    text_encoder.eval().requires_grad_(False)

    dist.print0('Constructing network...')
    net = dnnlib.util.construct_class_by_name(**network_kwargs) # subclass of torch.nn.Module

    dist.print0('Loading VAE...')
    vae = net.load_vae()
    vae.eval().requires_grad_(False).to(device)

    def vae_decode_latents(latents, vae):
        image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    # del net.text_encoder
    # del net.vae

    # Offload the vae to cpu
    # cpuvae = deepcopy(vae).to(torch.float32).to('cpu')

    net.train().requires_grad_(True).to(device)
    tch_net = deepcopy(net)
    tch_net.eval().requires_grad_(False).to(device)
    dist.print0(f'type: {type(tch_net)}')
    
    # useless parameters
    del net.model.y_embedder.y_embedding 
    
    dist.print0('Setting up DDP...')
    if use_fsdp:
        ddp = FSDP(net, # device_id=device,
                    sharding_strategy=ShardingStrategy.SHARD_GRAD_OP
                )
    else:
        ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device])
    
    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) 
    
    optimizer_student = dnnlib.util.construct_class_by_name(
        params=misc.all_trainable_params(net), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    ema = dnnlib.util.construct_class_by_name(model=net, **ema_kwargs)

    # Resume training from previous snapshot.
    if resume_pt is not None:
        dist.print0(f'Loading network weights from "{resume_pt}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        data = torch.load(resume_pt, map_location='cpu')
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        ema.load_state_dict(data['ema'], strict=False)
        del data # conserve memory
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location='cpu')
        net.load_state_dict(data['net'])
            
        optimizer_student.load_state_dict(data['optimizer_student_state'])
        del data # conserve memory
    
    # Train.
    dist.print0(f'Training for {total_steps} steps...')
    dist.print0()
    cur_tick = 0
    cur_nimg = 0
    training_step = resume_step
    tick_start_step = training_step
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    stats_jsonl = None
    
    amp_enabled = precision != "fp32"
    precision_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    assert not amp_enabled or precision in ['fp16', 'bf16']
    
    # validation setting
    if dist.get_rank() == 0:
        valid_text_embs, valid_masks = torch.load("./metrics/parti20.pt", map_location='cpu')
        valid_text_embs = valid_text_embs[:4].to(device)
        valid_masks = valid_masks[:4].to(device)
        rnd = StackedRandomGenerator('cpu', list(range(len(valid_text_embs))))
        validate_noise = rnd.randn([len(valid_text_embs), 4, 64, 64]).to(device)
    
    # Generating images
    if dist.get_rank() == 0 and (cur_tick % image_tick == 0):
        with torch.autocast(device_type="cuda", enabled=amp_enabled, dtype=precision_dtype):
            sigma_init = loss_fn.sigma_init.repeat(len(valid_text_embs), 1, 1, 1).to(device)
            noisy_x = validate_noise * sigma_init

            steps = 20
            x = noisy_x
            stride = len(tch_net.u) // steps
            step = torch.tensor([0]).repeat(len(valid_text_embs), 1, 1, 1).to(device)

            with torch.no_grad():
                for i in range(steps):
                    next_step = step + stride
                    t    = tch_net.idx_to_sigma(step, dtype=torch.float32).to(device)
                    r    = tch_net.idx_to_sigma(next_step, dtype=torch.float32).to(device)
                    eps  = tch_net(x, valid_text_embs, t, mask=valid_masks, return_eps=True)
                    x    = x + (r - t) * eps
                    assert (t - r).min() > 0
                    dist.print0(f"Step {step.flatten()} -> {next_step.flatten()}")
                    dist.print0(f"Noise level {t.flatten()} -> {r.flatten()}")
                    step = next_step
            
            latents = x

            # Decode latents
            latents = latents.to(torch.float32)
            images = vae_decode_latents(latents, vae)
            images_np = (images * 255).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

            for j in range(len(images_np)):
                pil_img = Image.fromarray(images_np[j], 'RGB')
                pil_img.save(os.path.join(run_dir, f'teacher-{j}.jpg'))
    
    dist.print0(f"Grad accumlation: {num_accumulation_rounds}")
    while True:        

        # Gradient accumulation for training generator 
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                imgs, prompts = next(dataloader)
                dist.print0(f"Prompts: {prompts}")

                # Initialize batch dictionary to store all inputs
                batch = {'prompts': prompts}
                encoder_hidden_states, attention_masks = encode_prompts(prompts, tokenizer, text_encoder, max_length=128)
                batch["prompt_emb"] = encoder_hidden_states
                batch["masks"] = attention_masks 
        
                with torch.autocast(device_type="cuda", enabled=amp_enabled, dtype=precision_dtype):
                    imgs = imgs.to(device=device)

                    # Encode in chunks to avoid OOM if batch size is large
                    with torch.no_grad():
                        latents = []
                        for i in range(0, imgs.shape[0], 8):
                            latents.append(vae.encode(imgs[i : i + 8]).latent_dist.sample())
                        latents = torch.cat(latents, dim=0)
                    
                    # Scale latents according to VAE's scaling factor (usually defined in its config)
                    latents = latents * vae.config.scaling_factor
                    batch["latents"] = latents

                    if reward_model is not None:
                        # Add reward model related term
                        tokenize_output = reward_model.blip.tokenizer(batch['prompts'], padding='max_length', truncation=True, max_length=35, return_tensors="pt")
                        batch["rm_input_ids"] = tokenize_output.input_ids.to(device)
                        batch["rm_attention_mask"] = tokenize_output.attention_mask.to(device)
                    
                    loss = loss_fn(stu_net=ddp, tch_net=tch_net, vae=vae, reward_model=reward_model, batch=batch, vae_decode_fn=vae_decode_latents, rm_preprocess_fn=rm_preprocess)
                
                # sdist.print0(f"Loss: {loss.mean().item()}")
                training_stats.report('Loss/loss', loss)
                loss.sum().mul(loss_scaling / batch_size / num_accumulation_rounds).backward()
        
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
        # Update weights.
        optimizer_student.step()
        optimizer_student.zero_grad(set_to_none=True)

        # Update EMA.
        ema.update(net)

        cur_nimg += batch_size * num_accumulation_rounds * dist.get_world_size()
        done = (training_step >= total_steps)
        training_step += 1
        # Perform maintenance tasks once per tick.
        if (not done) and (cur_tick != 0) and (training_step < tick_start_step + step_per_tick):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"step {training_stats.report0('Progress/step', training_step)}"]
        fields += [f"loss {training_stats.default_collector['Loss/loss']:<5.5f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / cur_nimg * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0('\t'.join(fields))

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0) and (cur_tick > 0):
            data = dict(ema=ema.state_dict())
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                torch.save(data, os.path.join(run_dir, f'network-snapshot-{training_step:06d}.pt'))
            del data # conserve memory

        # Save full dump of the training state.
        # if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
        #     torch.save(dict(
        #         net=net.state_dict(), 
        #         optimizer_student_state=optimizer_student.state_dict()), 
        #         os.path.join(run_dir, f'training-state-{training_step:06d}.pt'))

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()

        # Generating images
        if dist.get_rank() == 0 and (cur_tick % image_tick == 0) and (cur_tick > 0):
            with torch.autocast(device_type="cuda", enabled=amp_enabled, dtype=precision_dtype):
                sigma_init = loss_fn.sigma_init.repeat(len(valid_text_embs), 1, 1, 1).to(device)
                noisy_x = validate_noise * sigma_init

                mid_t = 4.0
                mid_t = torch.tensor([mid_t]).repeat(len(valid_text_embs), 1, 1, 1).to(device)

                x = noisy_x
                with torch.no_grad():
                    x = ema.module(x, valid_text_embs, sigma_init, mask=valid_masks)
                    x = x + mid_t * torch.randn_like(x)
                    x = ema.module(x, valid_text_embs, mid_t, mask=valid_masks)
                latents = x

                # Decode latents
                latents = latents.to(torch.float32)
                images = vae_decode_latents(latents, vae)
                images_np = (images * 255).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

                for j in range(len(images_np)):
                    pil_img = Image.fromarray(images_np[j], 'RGB')
                    pil_img.save(os.path.join(run_dir, f'{training_step:06d}-{j}.jpg'))

                # images = vae_decode(vae, latents)
                # images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                # im5x5 = make_grid_5x5(images_np)
                # Image.fromarray(im5x5, 'RGB').save(os.path.join(run_dir, f'{training_step:06d}.jpg'))
        
        torch.distributed.barrier()
        # Update state.
        cur_tick += 1
        cur_nimg = 0
        tick_start_step = training_step
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------
