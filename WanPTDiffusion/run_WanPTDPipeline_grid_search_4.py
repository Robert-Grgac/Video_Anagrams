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
    loaded_latents = torch.load("/home/s2710099/projects/Video_Anagrams/latents/initial_latent/initial_latents.pt", weights_only=True)
    print("Using same seed for reproducibility, loaded latents from /home/s2710099/projects/Video_Anagrams/latents/initial_latent/initial_latents.pt")
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
ref_latents_dir_list = ["./latents/precomputed_deterministic_inversion_latents_face1", "./latents/precomputed_deterministic_inversion_latents_face2", "./latents/precomputed_deterministic_inversion_latents_dog", "./latents/precomputed_deterministic_inversion_latents_cat", "./latents/precomputed_deterministic_inversion_latents_skull"]

# Heuristic grid search

Kp_list = [0.1, 0.25, 0.5, 1.0, 2.0] #proportional gain
Ki = 0.0 #integral gain next try to sweep based on the best Kp [0.01, 0.05, 0.1, 0.2] dont forge tot adjust the clamp as well
max_alpha_delta_list = [0.02, 0.05, 0.1]
direct_transfer_steps= 45
decayed_transfer_steps = 22
initial_alpha = 0.4



for Kp in Kp_list:
    for max_alpha_delta in max_alpha_delta_list:
        run_name = f"heuristic_v3_Kp_{Kp}_max_alpha_delta_{max_alpha_delta}"
        with wandb.init(
            project="wan-heuristic-experiments-4",
            name=run_name,
            config={
                "description": "Testing heuristic with different prompts and reference latents",
                "asset_name": "face1",
                "prompt_idx": 1,
                "prompt": prompt1,
                "ref_latents_dir": "./latents/precomputed_deterministic_inversion_latents_face1",
                "direct_transfer_steps": direct_transfer_steps,
                "decayed_transfer_steps": decayed_transfer_steps,
                "initial_alpha": initial_alpha,
                "Kp": Kp,
                "Ki": Ki,
                "max_alpha_delta": max_alpha_delta,
            }
        ) as run:
            video = pipe(
                    # Standard params
                        prompt=prompt1,
                        negative_prompt=negative_prompt,
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        num_inference_steps=num_inference_steps,
                        latents=loaded_latents.clone() if use_same_seed else None,
                        guidance_scale=0.0, #NO CLASSIFIER FREE GUIDANCE FOR HEURISTIC GRID SEARCH
                    #PTM params
                        direct_transfer_steps=direct_transfer_steps,
                        decayed_transfer_steps=decayed_transfer_steps,
                        exponent=0.5,
                        initial_alpha=initial_alpha,
                        ref_latents_dir="./latents/precomputed_deterministic_inversion_latents_face1",
                        use_blending_heuristic_version_1=False,
                        use_blending_heuristic_version_2=False,
                        use_blending_heuristic_version_3=True,
                        Kp=Kp,
                        Ki=Ki,
                        max_alpha_delta=max_alpha_delta
                )


            frames = video.get('frames', video.get('images', video))

            frames = frames[0]  # Remove batch dimension -> [frames, height, width, channels]
            frame_list = [frames[i] for i in range(frames.shape[0])]

            export_to_video(frame_list, f"/home/s2710099/outputs/heuristic_v3/{run_name}.mp4")
    
wandb.finish()