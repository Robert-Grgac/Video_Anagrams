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
from ModWanPipeline import ModWanPipeline
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
pipe = ModWanPipeline.from_pretrained(model_path, vae=vae, torch_dtype=dtype)
print('--- Anagram pipeline loaded ---')
pipe.enable_model_cpu_offload()
print("Pipeline setup done...")

#Video generation parameters
prompt1 = "A video zooming in on an upside down blue mug on a shelf , photorealistic, high quality"
prompt2 = "A video zooming in on a red mug on a shelf , photorealistic, high quality"
negative_prompt = "blurry, low quality, worst quality,overlapping objects, double subjects, fused objects, jpeg artifacts, text, subtitles, watermark, static image, still frame, distorted anatomy, deformed objects, inconsistent motion"
height = 528 #480 for the 480p quality
width = 528 #832
num_frames = 61
num_inference_steps = 80
guidance_scale = 8
inflection_timestep = 2
view = HorizontalFlipView()
print("Generating video...")
video_a, video_b = pipe(
    # New params
        prompt_a=prompt1,
        prompt_b=prompt2,
        inflection_timestep=inflection_timestep,
        view=view,
    # Standard params
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        latents=loaded_latents.clone(),
        guidance_scale=guidance_scale,
    )


frames_a = video_a.get('frames', video_a.get('images', video_a))
frames_b = video_b.get('frames', video_b.get('images', video_b))

print(f"Frames A shape: {frames_a.shape}")
print(f"Frames B shape: {frames_b.shape}")

frames_a = frames_a[0]  # Remove batch dimension -> [frames, height, width, channels]
frames_b = frames_b[0]  # Remove batch dimension -> [frames, height, width, channels]
frame_list_a = [frames_a[i] for i in range(frames_a.shape[0])]
frame_list_b = [frames_b[i] for i in range(frames_b.shape[0])]

export_to_video(frame_list_a, "/home/s2710099/outputs/ModPipeline_not_flipped.mp4", fps=16)
export_to_video(frame_list_b, "/home/s2710099/outputs/ModPipeline_flipped.mp4", fps=16)

print("Video A saved to /home/s2710099/outputs/ModPipeline_not_flipped.mp4")
print("Video B saved to /home/s2710099/outputs/ModPipeline_flipped.mp4")