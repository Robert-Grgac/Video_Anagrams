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

ref_cos_threshold_list = [0.05, 0.06]
ref_factor_list = [0.01, 0.05, 0.1]
cond_mag_threshold_list = [400, 500]
cond_factor_list = [0.01, 0.05, 0.1]
direct_transfer_steps= 45
decayed_transfer_steps = 22
alpha = 0.4



for idx, ref_cos_threshold in enumerate(ref_cos_threshold_list):
    for pidx, ref_factor in enumerate(ref_factor_list):
        for cidx, cond_mag_threshold in enumerate(cond_mag_threshold_list):
            for cfidx, cond_factor in enumerate(cond_factor_list):
        
                run_name = f"face1_ref_cos_threshold_{ref_cos_threshold}_ref_factor_{ref_factor}_cond_mag_threshold_{cond_mag_threshold}_cond_factor_{cond_factor}"
                with wandb.init(
                    project="wan-heuristic-experiments-1",
                    name=run_name,
                    config={
                        "asset_name": "face1",
                        "prompt_idx": 2,
                        "prompt": prompt2,
                        "ref_latents_dir": "./latents/precomputed_deterministic_inversion_latents_face1",
                        "direct_transfer_steps": 0,
                        "decayed_transfer_steps": 0,
                        "initial_alpha": alpha,
                    }
                ) as run:
                    video = pipe(
                            # Standard params
                                prompt=prompt2, #FIXED PROMPT
                                negative_prompt=negative_prompt,
                                height=height,
                                width=width,
                                num_frames=num_frames,
                                num_inference_steps=num_inference_steps,
                                latents=loaded_latents.clone() if use_same_seed else None,
                                guidance_scale=0.0, #NO CLASSIFIER FREE GUIDANCE FOR HEURISTIC GRID SEARCH
                            #PTM params
                                direct_transfer_steps=0, #NOT USED NOW
                                decayed_transfer_steps=0, #NOT USED NOW
                                exponent=0.5,
                                initial_alpha=alpha,
                                ref_latents_dir="./latents/precomputed_deterministic_inversion_latents_face1",
                                ref_threshold = ref_cos_threshold,
                                ref_factor = ref_factor,
                                cond_threshold = cond_mag_threshold,
                                cond_factor = cond_factor,
                        )


                    frames = video.get('frames', video.get('images', video))

                    frames = frames[0]  # Remove batch dimension -> [frames, height, width, channels]
                    frame_list = [frames[i] for i in range(frames.shape[0])]

                    export_to_video(frame_list, f"/home/s2710099/outputs/deterministic/{run_name}.mp4")
    
wandb.finish()