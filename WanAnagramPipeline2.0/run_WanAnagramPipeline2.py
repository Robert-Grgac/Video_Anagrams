#Library imports
import torch
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
import numpy as np
import os
import random
import sys
#sys.path.append('..')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)
print("Importing libraries done...")

#importing pipeline
from WanAnagramPipeline2 import WanAnagramPipeline2
from utils.views import HorizontalFlipView, VerticalFlipView, TimeReverseView
print("Importing pipeline done...")

#For reproducibility
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
loaded_latents = torch.load("/home/s2710099/projects/Video_Anagrams/latents/initial_latents.pt")

#Pipeline setup
dtype = torch.bfloat16
model_path = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
print('--- VAE loaded ---')
pipe = WanAnagramPipeline2.from_pretrained(model_path, vae=vae, torch_dtype=dtype)
print('--- Anagram pipeline loaded ---')
pipe.enable_model_cpu_offload()
print("Pipeline setup done...")

#Video generation parameters
##Different animation styles
WRAP_A = "clean vector animation, flat shading, bold outlines, minimal palette, locked camera"
WRAP_B = "paper cutout collage animation, flat colors, subtle paper texture, locked camera"
WRAP_C = "woodblock print animation, high contrast, limited colors, carved texture, locked camera"
WRAP_D = "watercolor ink wash animation, soft edges, painterly, simple background, locked camera"
WRAP_E = "particle swarm animation forming recognizable shapes, smooth motion, simple background, locked camera"

prompt1 = f"Video of a mushroom growing from the ground, time-lapse feel, {WRAP_C}, high quality."
prompt2 = f"Video of a rocket launching upward with exhaust plume, stylized, {WRAP_C}, high quality."
negative_prompt = "blurry, low quality, worst quality, jpeg artifacts, text, subtitles, watermark, static image, still frame, distorted anatomy,  inconsistent motion"
height = 528 
width = 528 
num_frames = 61
num_inference_steps = 80
#Set joint diffusions steps to -1 if you want to run only anagram pipeline
joint_diffusion_steps = -1
#There are 3 options for cfg: None (put both cond and neg to None), only conditional (put neg to none), conditional + negative (assign to both some values)
conditional_guidance_scale = 8.0
negative_guidance_scale = 7.0
view = VerticalFlipView()
apply_mean_reduction = True
print("Generating video...")
if not apply_mean_reduction and joint_diffusion_steps < num_inference_steps:
    print("Generating two videos (flipped and not flipped)")
    video_a, video_b = pipe(
        # New params
            prompt_a=prompt1,
            prompt_b=prompt2,
        #Joint diffusion params
            conditional_guidance_scale=conditional_guidance_scale,
            negative_guidance_scale=negative_guidance_scale,
            joint_diffusion_steps=joint_diffusion_steps,
        # Anagram params
            view=view,
            apply_mean_reduction=apply_mean_reduction,
        # Standard params
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            latents=loaded_latents.clone(),
        )
else:
    print("Generating single video (not flipped)")
    video_a = pipe(
        # New params
            prompt_a=prompt1,
            prompt_b=prompt2,
        #Joint diffusion params
            conditional_guidance_scale=conditional_guidance_scale,
            negative_guidance_scale=negative_guidance_scale,
            joint_diffusion_steps=joint_diffusion_steps,
        # Anagram params
            view=view,
            apply_mean_reduction=apply_mean_reduction,
        # Standard params
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            latents=loaded_latents.clone(),
        )


frames_a = video_a.get('frames', video_a.get('images', video_a))

print(f"Frames A shape: {frames_a.shape}")

frames_a = frames_a[0]  # Remove batch dimension -> [frames, height, width, channels]
frame_list_a = [frames_a[i] for i in range(frames_a.shape[0])]

export_to_video(frame_list_a, "/home/s2710099/outputs/WanAnagramPipeline2_not_flipped.mp4", fps=16)

if joint_diffusion_steps < num_inference_steps and not apply_mean_reduction:
    frames_b = video_b.get('frames', video_b.get('images', video_b))

    print(f"Frames B shape: {frames_b.shape}")

    frames_b = frames_b[0]  # Remove batch dimension -> [frames, height, width, channels]
    frame_list_b = [frames_b[i] for i in range(frames_b.shape[0])]

    export_to_video(frame_list_b, "/home/s2710099/outputs/WanAnagramPipeline2_flipped.mp4", fps=16)
    print("Video B saved to /home/s2710099/outputs/WanAnagramPipeline2_flipped.mp4")

print("Video A saved to /home/s2710099/outputs/WanAnagramPipeline2_not_flipped.mp4")