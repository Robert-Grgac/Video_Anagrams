import torch
import sys, einops
import numpy as np
sys.path.append('..')
import os, re
 
from tqdm import tqdm
 
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video
import torch.nn.functional as F
from pathlib import Path
import logging
from accelerate import Accelerator
 
# Simple implementations of the required classes
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from torch import nn
import random
print("Importing done...")
 
# for reproducibility
seed = 0
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)
os.environ['HF_HUB_OFFLINE']='1'
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
print("Environment setup done...")
 
dtype = torch.bfloat16
model_path = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
print('--- VAE loaded ---')
pipe = WanPipeline.from_pretrained(model_path, vae=vae, torch_dtype=dtype)
print('--- Pipe loaded ---')
# Create generator for reproducibility
device = "cuda" if torch.cuda.is_available() else "cpu"
generator = torch.Generator(device=device)
generator.manual_seed(seed)

# Generate an initial noise latent that can be reused for each run
height = 528
width = 528
num_frames = 61
latents = pipe.prepare_latents(
    batch_size=1,
    num_channels_latents=16,  
    height=height,
    width=width,
    num_frames=num_frames,
    dtype=torch.float32,
    device=device,
    generator=generator,
    latents=None,
)
print(f"Generated latents shape: {latents.shape}")
print(f"Latents mean: {latents.mean().item():.6f}")
print(f"Latents std: {latents.std().item():.6f}")

# Save the latents to a file
torch.save(latents, "/home/s2710099/projects/latents/initial_latents.pt")