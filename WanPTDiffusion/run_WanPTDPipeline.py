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
from WanPTDiffusion.WanPTDPipeline import WanPTDiffusionPipeline
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
loaded_latents = torch.load("/home/s2710099/projects/Video_Anagrams/latents/inital_latent/initial_latents.pt", weights_only=True)

#Pipeline setup
dtype = torch.bfloat16
model_path = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
print('--- VAE loaded ---')
pipe = WanPTDiffusionPipeline.from_pretrained(model_path, vae=vae, torch_dtype=dtype)
print('--- PT Diffusion pipeline loaded ---')
pipe.enable_model_cpu_offload()
print("Pipeline setup done...")

#helper function to load latents
def load_latents(latent_folder):
    if not os.path.exists(latent_folder):
        raise FileNotFoundError(f"Latent folder '{latent_folder}' does not exist.")
    latents = []
    image_names = []
    for filename in os.listdir(latent_folder):
        if filename.endswith(".pt"):
            latent_path = os.path.join(latent_folder, filename)
            latent = torch.load(latent_path, weights_only=True)
            latents.append(latent)
            image_names.append(filename[:-10])
    return latents, image_names

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
num_inference_steps = 100
guidance_scale = 7.0
#Gridearch parameters
prompt_list = [prompt1, prompt2, prompt3, prompt4, prompt5, prompt6, prompt7]
inital_alpha_list = [ 0.8, 0.6, 0.4, 0.2, 0.15, 0.1]
#deterministic_latent_list, deterministic_image_names = load_latents("./deterministic_inversion_latents")
#non_deterministic_latent_list, non_deterministic_image_names = load_latents("./non_deterministic_inversion_latents")



video = pipe(
        # Standard params
            prompt=prompt1,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            latents=loaded_latents.clone(),
            guidance_scale=0.0,
        #PTM params
            direct_transfer_steps=15, #Could also play with direct and decay transfer steps
            decayed_transfer_steps=7,
            exponent=0.5,
            initial_alpha=0.4,
            ref_latents_dir="./latents/precomputed_deterministic_inversion_latents_face1",
            use_preloaded_latents=True,
            use_blending_heuristic=True,
            ref_threshold = 0.05,
            ref_factor = 0.01,
            cond_threshold = 400,
            cond_factor = 0.01,
        )

frames = video.get('frames', video.get('images', video))

print(f"Frames shape: {frames.shape}")

frames = frames[0]  # Remove batch dimension -> [frames, height, width, channels]
frame_list = [frames[i] for i in range(frames.shape[0])]

export_to_video(frame_list, f"/home/s2710099/outputs/deterministic/WanPTDiffusion_test.mp4", fps=16)

print(f"Video saved to /home/s2710099/outputs/deterministic/WanPTDiffusion_test.mp4")