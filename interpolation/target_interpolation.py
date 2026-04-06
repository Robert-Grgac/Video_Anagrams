from diffusers import AutoencoderKLWan
import os
import sys
from PIL import Image
import pip
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from WanInversionPipeline import WanInversionPipeline

reference_image_1 = Image.open("/home/s2710099/projects/Video_Anagrams/assets/face1.jpg").convert("RGB")
reference_image_2 = Image.open("/home/s2710099/projects/Video_Anagrams/assets/face2.jpg").convert("RGB")
reference_image_3 = Image.open("/home/s2710099/projects/Video_Anagrams/assets/dog.png").convert("RGB")
reference_image_4 = Image.open("/home/s2710099/projects/Video_Anagrams/assets/cat.png").convert("RGB")
reference_image_5 = Image.open("/home/s2710099/projects/Video_Anagrams/assets/skull.png").convert("RGB")

reference_image_list = [reference_image_1, reference_image_2, reference_image_3, reference_image_4, reference_image_5]
naming_list = ["face1", "face2", "dog", "cat", "skull"]

dtype = torch.bfloat16
model_path = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
print('--- VAE loaded ---')
pipe = WanInversionPipeline.from_pretrained(model_path, vae=vae, torch_dtype=dtype)
print('--- WanInversionPipeline pipeline loaded ---')
pipe.enable_model_cpu_offload()
print("Pipeline setup done...")

number_of_inference_steps = 100
# target_step = 1
# deterministic_latent_dir = "./deterministic_inversion_latents"
# non_deterministic_latent_dir = "./non_deterministic_inversion_latents"
# output_dir = "./decoded_latents"

# for i, reference_image in enumerate(reference_image_list):
#     latent_filename = f"{naming_list[i]}_step_{target_step}.pt"
#     pipe.deterministic_invert(reference_image=reference_image, num_inference_steps=number_of_inference_steps, target_step=target_step, latent_filename=latent_filename)
    
# for i, reference_image in enumerate(reference_image_list):
#     latent_filename = f"{naming_list[i]}_step_{target_step}.pt"
#     pipe.invert(reference_image=reference_image, num_inference_steps=number_of_inference_steps, latent_filename=latent_filename)

# pipe.decode_inversion_trajectory_to_images(latent_dir=deterministic_latent_dir, output_dir=output_dir, frame_index=0)
# pipe.decode_inversion_trajectory_to_images(latent_dir=non_deterministic_latent_dir, output_dir=output_dir, frame_index=0)

#save_image_dir = "./precomputed_deterministic_inversion_images"

#deterministic_latent_dir_face1 = "./precomputed_deterministic_inversion_latents_face1"
#deterministic_latent_dir_face2 = "./precomputed_deterministic_inversion_latents_face2"
#deterministic_latent_dir_dog = "./precomputed_deterministic_inversion_latents_dog"
#deterministic_latent_dir_cat = "./precomputed_deterministic_inversion_latents_cat"
#deterministic_latent_dir_skull = "./precomputed_deterministic_inversion_latents_skull"

save_images = False


#pipe.deterministic_invert(reference_image=reference_image_2, num_inference_steps=number_of_inference_steps,  save_latent_dir=deterministic_latent_dir_face2,save_image_dir = save_image_dir , save_images=save_images)
#pipe.deterministic_invert(reference_image=reference_image_3, num_inference_steps=number_of_inference_steps,  save_latent_dir=deterministic_latent_dir_dog,save_image_dir = save_image_dir , save_images=save_images)
#pipe.deterministic_invert(reference_image=reference_image_4, num_inference_steps=number_of_inference_steps,  save_latent_dir=deterministic_latent_dir_cat,save_image_dir = save_image_dir , save_images=save_images)
#pipe.deterministic_invert(reference_image=reference_image_5, num_inference_steps=number_of_inference_steps,  save_latent_dir=deterministic_latent_dir_skull,save_image_dir = save_image_dir , save_images=save_images)

reference_video_path = "./assets/Man_shaking_head.mp4"
pipe.deterministic_invert_video(reference_video=reference_video_path, num_inference_steps=number_of_inference_steps)