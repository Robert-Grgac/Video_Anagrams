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
loaded_latents = torch.load("/home/s2710099/projects/latents/initial_latents.pt")
print("Environment setup done...")
 
dtype = torch.bfloat16
model_path = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
print('--- VAE loaded ---')
pipe = WanPipeline.from_pretrained(model_path, vae=vae, torch_dtype=dtype)
print('--- Pipe loaded ---')
pipe.enable_model_cpu_offload()
#negative_prompt = "色调艳丽, 过曝, 静态, 细节模糊不清,字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, 最差质量, 低质量, JPEG压缩残留, 丑陋的, 残缺的, 多余的手指, 画得不好的手部, 画得不好的脸部, 畸形的, 毁容的, 形态畸形的肢体, 手指融合, 静止不动的画面, 杂乱的背景, 三条腿, 背景人很多, 倒着走"
prompt1 = "A calm ocean at sunset, cinematic lighting, wide shot."
prompt2 = "A majestic mountain range under a starry night sky, cinematic lighting, wide shot."
# Video parameters
height = 528
width = 528
num_frames = 61
num_inference_steps = 50
guidance_scale = 4.0
video = pipe(
        prompt=prompt2,
        #negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        guidance_scale_2=3.0,
        latents=loaded_latents.clone(),
    )


if isinstance(video, dict):
    print("Use video.get to access frames")
    frames = video.get('frames', video.get('images', video))
elif hasattr(video, 'frames'):
    print("Use video.frames to access frames")
    frames = video.frames
else:
    print("Video is a tensor or list")
    frames = video

print(f"Frames shape: {frames.shape}")

# Remove batch dimension and convert to list of frames
if len(frames.shape) == 5:  # [batch, frames, height, width, channels]
    frames = frames[0]  # Remove batch dimension -> [frames, height, width, channels]

# Convert to list of individual frames
frame_list = [frames[i] for i in range(frames.shape[0])]

print(f"Number of frames: {len(frame_list)}")
print(f"Individual frame shape: {frame_list[0].shape}")

export_to_video(frame_list, "/home/s2710099/outputs/output_test_run_3.mp4", fps=16)

print("Video saved to /home/s2710099/outputs/output_test_run_3.mp4")