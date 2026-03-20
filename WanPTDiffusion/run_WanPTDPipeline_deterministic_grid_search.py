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
from WanPTDiffusion.WanPTDPipeline import WanPTDiffusionPipeline
print("Importing pipeline done...")

use_same_seed = False
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
    loaded_latents = torch.load("/home/s2710099/projects/Video_Anagrams/latents/initial_latents.pt", weights_only=True)
    print("Using same seed for reproducibility, loaded latents from /home/s2710099/projects/Video_Anagrams/latents/initial_latents.pt")
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
#Gridearch parameters
asset_names_list = ["face1", "face2", "dog", "cat", "skull"]
prompt_list = [prompt1, prompt2, prompt3, prompt4, prompt5, prompt6, prompt7]
inital_alpha_list = [ 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
transfer_steps_list= [(5,2), (10,5),(15,7), (20,10), (25,12), (30,15), (45,22)]
ref_latents_dir_list = ["./precomputed_deterministic_inversion_latents_face1", "./precomputed_deterministic_inversion_latents_face2", "./precomputed_deterministic_inversion_latents_dog", "./precomputed_deterministic_inversion_latents_cat", "./precomputed_deterministic_inversion_latents_skull"]


for idx, ref_latents_dir in enumerate(ref_latents_dir_list):
    direct_transfer_steps= 45
    decayed_transfer_steps = 22
    alpha = 0.4
    for pidx, p in enumerate(prompt_list):
        
        run_name = f"{asset_names_list[idx]}_prompt_{pidx}_alpha_{alpha}"
        with wandb.init(
            project="wan-phase-experiments-2",
            name=run_name,
            config={
                "asset_name": asset_names_list[idx],
                "prompt_idx": pidx,
                "prompt": p,
                "ref_latents_dir": ref_latents_dir,
                "direct_transfer_steps": direct_transfer_steps,
                "decayed_transfer_steps": decayed_transfer_steps,
                "initial_alpha": alpha,
            }
        ) as run:
            video = pipe(
                    # Standard params
                        prompt=p,
                        negative_prompt=negative_prompt,
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        num_inference_steps=num_inference_steps,
                        latents=loaded_latents.clone() if use_same_seed else None,
                        guidance_scale=guidance_scale,
                    #PTM params
                        direct_transfer_steps=direct_transfer_steps,
                        decayed_transfer_steps=decayed_transfer_steps,
                        exponent=0.5,
                        initial_alpha=alpha,
                        ref_latents_dir=ref_latents_dir,
                    )

            frames = video.get('frames', video.get('images', video))

            frames = frames[0]  # Remove batch dimension -> [frames, height, width, channels]
            frame_list = [frames[i] for i in range(frames.shape[0])]

            export_to_video(frame_list, f"/home/s2710099/outputs/deterministic/WanPTDiffusion_{asset_names_list[idx]}_prompt_{pidx}.mp4")
    
wandb.finish()