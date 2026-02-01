#Library imports
import torch
from diffusers import AutoencoderKLWan
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video
import numpy as np
import os
import random
import sys
sys.path.append('..')
print("Importing libraries done...")

#importing pipeline
from pipeline_wan_anagram import WanAnagramPipeline
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
loaded_latents = torch.load("/home/s2710099/projects/latents/initial_latents.pt")

#Pipeline setup
dtype = torch.bfloat16
model_path = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
print('--- VAE loaded ---')
pipe = WanAnagramPipeline.from_pretrained(model_path, vae=vae, torch_dtype=dtype)
print('--- Anagram pipeline loaded ---')
pipe.enable_model_cpu_offload()
# pipe.scheduler = UniPCMultistepScheduler.from_config(
#     pipe.scheduler.config,
#     flow_shift=3.0,   # ~480p
# )
print("Pipeline setup done...")

#Video generation parameters
prompt1 = "A car driving from left to right on a highway at daytime, wide shot, cinematic lighting"
prompt2 = "A motorcycle driving from right to left on a highway at daytime, wide shot, cinematic lighting"
negative_prompt = "blurry, low quality, worst quality,overlapping objects, double subjects, fused objects, jpeg artifacts, text, subtitles, watermark, static image, still frame, distorted anatomy, deformed objects, inconsistent motion"
height = 528 #480 for the 480p quality
width = 528 #832
num_frames = 61
num_inference_steps = 150
guidance_scale_joint = 6.0 #Custome giudance scale for joint stage SET BOTH MANUALLY HERE
guidance_scale_anagram = 9.0 #Custom guidance scale for anagram stage
alternating_reduction = False #Whether to use alternating view conditioning instead of mean reduction
joint_steps = 40 #Number of joint diffusion steps
view_2 = TimeReverseView()
print("Generating video...")
video = pipe(
        prompt_a=prompt1,
        prompt_b=prompt2,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        latents=loaded_latents.clone(),
        joint_steps=joint_steps,
        view_b=view_2,
        alternating_reduction=alternating_reduction,
        guidance_scale_joint=guidance_scale_joint,
        guidance_scale_anagram=guidance_scale_anagram,
    )


frames = video.get('frames', video.get('images', video))

print(f"Frames shape: {frames.shape}")

frames = frames[0]  # Remove batch dimension -> [frames, height, width, channels]

frame_list = [frames[i] for i in range(frames.shape[0])]

export_to_video(frame_list, "/home/s2710099/outputs/anagram_test_run_3.mp4", fps=16)

print("Video saved to /home/s2710099/outputs/anagram_test_run_3.mp4")