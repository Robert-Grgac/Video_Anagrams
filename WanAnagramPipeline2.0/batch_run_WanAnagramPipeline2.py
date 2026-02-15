#Library imports
import torch
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
import numpy as np
import os
import random
import sys
import copy
#sys.path.append('..')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)
print("Importing libraries done...")

#importing pipeline
from WanAnagramPipeline2 import WanAnagramPipeline2
from utils.views import HorizontalFlipView, VerticalFlipView, TimeReverseView, Rotate180View
from utils.prompt_generator import PromptGenerator
print("Importing pipeline done...")

#Prompt setup
prompt_generator = PromptGenerator()
number_of_prompts = 1
anagram_style = "painting" #Options are 'painting', 'street art', 'cubist painting', 'oil painting' or None for random
heuristic_style_list = ['street art','cubist painting', 'watercolor painting', 'vintage poster', 'pencil sketch', 'oil painting'] #Options are 'street art', 'minimalist illustration', 'cubist painting', 'watercolor painting', 'vintage poster', 'pencil sketch', 'oil painting' or None for random
preffered_view = "rotation" #Options are 'flip' or 'rotation' 
#anagram_prompt_list = prompt_generator.generate_static_visual_anagram_prompt(id=None, style=anagram_style, NumOfPrompts=number_of_prompts, preffered_view=preffered_view)
#heuristic_prompt_list = prompt_generator.generate_static_heuristic_prompt(id=None, styles=heuristic_style_list, NumOfPrompts=number_of_prompts, preffered_view=preffered_view)
print("Prompt generation done...")

#Temporary
view_list = [HorizontalFlipView(), VerticalFlipView(), Rotate180View()]
heuristic_prompt_list = []
for idx,style in enumerate(heuristic_style_list):
    view = view_list[idx % len(view_list)]
    prompt = prompt_generator.generate_static_heuristic_prompt(id=None, styles=[style], NumOfPrompts=1, preffered_view=preffered_view)[0]
    prompt.views = [view]
    heuristic_prompt_list.append(prompt)

print(heuristic_prompt_list)
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
loaded_latents = torch.load("/home/s2710099/projects/Video_Anagrams/latents/initial_latents.pt", weights_only=True)

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
negative_prompt = "blurry, low quality, worst quality, jpeg artifacts, text, subtitles, watermark, static image, still frame, distorted anatomy,  inconsistent motion"
height = 528 
width = 528 
num_frames = 61

#SELECT PROMPT LIST
prompt_list = heuristic_prompt_list
for prompt in prompt_list:
    print(f"Generating video with  prompt id {prompt.id}...")
    print(f"Prompt 1: {prompt.prompt1}")
    print(f"Prompt 2: {prompt.prompt2}")
    print(f"View: {prompt.views[0].__class__.__name__ if prompt.views else 'None'}")
    prompt1 = prompt.prompt1
    prompt2 = prompt.prompt2
    num_inference_steps = 80
    joint_diffusion_steps = -1
    conditional_guidance_scale = 8.0
    negative_guidance_scale = 7.0
    view = prompt.views[0]
    apply_mean_reduction = True
    if not apply_mean_reduction and joint_diffusion_steps < num_inference_steps:
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


    frames_a = frames_a[0]  # Remove batch dimension -> [frames, height, width, channels]
    frame_list_a = [frames_a[i] for i in range(frames_a.shape[0])]

    export_to_video(frame_list_a, f"/home/s2710099/outputs/WanAnagramPipeline2_{prompt.id}_{prompt.views[0].__class__.__name__}.mp4", fps=16)
    print(f"Video A saved to /home/s2710099/outputs/WanAnagramPipeline2_{prompt.id}_{prompt.views[0].__class__.__name__}.mp4")

    if joint_diffusion_steps < num_inference_steps and not apply_mean_reduction:
        frames_b = video_b.get('frames', video_b.get('images', video_b))

        frames_b = frames_b[0]  # Remove batch dimension -> [frames, height, width, channels]
        frame_list_b = [frames_b[i] for i in range(frames_b.shape[0])]

        export_to_video(frame_list_b, f"/home/s2710099/outputs/WanAnagramPipeline2_flipped_{prompt.id}_{prompt.views[0].__class__.__name__}_flipped.mp4", fps=16)
        print(f"Video B saved to /home/s2710099/outputs/WanAnagramPipeline2_flipped_{prompt.id}_{prompt.views[0].__class__.__name__}_flipped.mp4")
