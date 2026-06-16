#Library imports
import torch
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
import numpy as np
import os
import random
import sys
import wandb
#sys.path.append('..')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)
print("Importing libraries done...")

#importing pipeline
from PTD_Pipeline.WanPTDPipeline import WanPTDiffusionPipeline
print("Importing pipeline done...")

use_same_seed = True
if use_same_seed:
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
    #loaded_latents = torch.load("/home/s2710099/projects/Video_Anagrams/latents/initial_latent/initial_latents.pt", weights_only=True)
    loaded_latents = None
    print("Using same seed for reproducibility, loaded latents are set to None...")
else:
    loaded_latents = None
    print("Not using same seed, latents will be generated on the fly...") 
    
os.environ['HF_HUB_OFFLINE']='1'
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

#Pipeline setup
dtype = torch.bfloat16
model_path = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
print('--- VAE loaded ---')
pipe = WanPTDiffusionPipeline.from_pretrained(model_path, vae=vae, torch_dtype=dtype)
print('--- PT Diffusion pipeline loaded ---')
pipe.enable_model_cpu_offload()
print("Pipeline setup done...")

#Video generation parameters
prompt1 = "snowy mountain,static, no movement, in style of cubist painting, high quality"
prompt2 = "grand canyon, static, no movement, photorealistic, high quality"
prompt3 = "park, static, no movement, in style of oil painting, high quality"
prompt4 = "seaflor, static, no movement, in style of watercolor painting, high quality"
prompt5 = "flowers, static, no movement, in style of street art, high quality"
prompt6 = "sky, static, no movement, photorealistic, high quality"
prompt7 = "a village in the mountains, static, no movement, in style of watercolor painting, high quality"
negative_prompt = "blurry, low quality, worst quality, jpeg artifacts, text, subtitles, watermark, static image, still frame, distorted anatomy,  inconsistent motion"
height = 528 
width = 528 
num_frames = 61
num_inference_steps = 101
guidance_scale = 7.0
# Single run: heuristic 4 (PI controller on spectral energy ratio), prompt2 + face_1
asset_name = "face_1"
prompt = prompt2
ref_dir = "/home/s2710099/cache/wan-beta/deterministic_invert_faces_528x528x61/face_1"

# PTM schedule
direct_transfer_steps = 45
decayed_transfer_steps = 22
initial_alpha = 0.4

# Heuristic 4 (energy-ratio PI controller) parameters
energy_target = 0.95
Kp_energy = 2.0
Ki_energy = 0.1
max_alpha_delta = 0.05

run_name = f"{asset_name}_prompt2_h4"
output_dir = "/home/s2710099/outputs/inference/ptd_h4"
os.makedirs(output_dir, exist_ok=True)

with wandb.init(
    project="PTD_inference_original_pipeline",
    name=run_name,
    config={
        "description": "Single heuristic-4 (energy-ratio PI controller) run.",
        "asset_name": asset_name,
        "prompt": prompt,
        "ref_latents_dir": ref_dir,
        "direct_transfer_steps": direct_transfer_steps,
        "decayed_transfer_steps": decayed_transfer_steps,
        "initial_alpha": initial_alpha,
        "energy_target": energy_target,
        "Kp_energy": Kp_energy,
        "Ki_energy": Ki_energy,
        "max_alpha_delta": max_alpha_delta,
    }
) as run:
    video = pipe(
            # Standard params
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                latents=None,
                guidance_scale=0.0, #NO CLASSIFIER FREE GUIDANCE FOR HEURISTIC GRID SEARCH
            #PTM params
                direct_transfer_steps=direct_transfer_steps,
                decayed_transfer_steps=decayed_transfer_steps,
                exponent=0.5,
                initial_alpha=initial_alpha,
                ref_latents_dir=ref_dir,
                use_blending_heuristic_version_1=False,
                use_blending_heuristic_version_2=False,
                use_blending_heuristic_version_3=False,
                use_blending_heuristic_version_4=True,
                energy_target=energy_target,
                Kp_energy=Kp_energy,
                Ki_energy=Ki_energy,
                max_alpha_delta=max_alpha_delta,
        )


    frames = video.get('frames', video.get('images', video))

    frames = frames[0]  # Remove batch dimension -> [frames, height, width, channels]
    frame_list = [frames[i] for i in range(frames.shape[0])]

    export_to_video(frame_list, f"{output_dir}/{run_name}.mp4")

wandb.finish()