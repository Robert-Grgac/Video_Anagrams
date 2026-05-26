#Library imports
import torch
from diffusers import AutoencoderKLWan, WanTransformer3DModel
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.utils import export_to_video
from transformers import AutoTokenizer, UMT5EncoderModel
from safetensors.torch import load_file
from accelerate.hooks import remove_hook_from_module
from huggingface_hub import snapshot_download
import numpy as np
import os
import random
import sys
import wandb
import PIL.Image
#sys.path.append('..')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)
print("Importing libraries done...")

#importing pipeline + CN components
from PTD_Pipeline.WanCNPTDPipeline import WanCNPTDiffusionPipeline
from wan_transformer import CustomWanTransformer3DModel
from wan_controlnet import WanControlnet
from training.utils import cast_respecting_fp32_modules, detect_boundary_ratio
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
# Can't use .from_pretrained on the whole pipeline: (a) the Wan 2.2 snapshot has
# no ControlNet, (b) the high-noise transformer must be CustomWanTransformer3DModel
# to accept controlnet kwargs. So assemble each component manually, mirroring what
# from_pretrained would have given the original WanPTDPipeline.
dtype = torch.bfloat16
# Resolve paths. Under sbatch (slurm/CNPTD_inference.sbatch) WAN_MODEL /
# HED_CONFIG / CHECKPOINT are set to staged local-scratch dirs to avoid the
# /home NFS thrashing that kills the transformer shard load. Outside sbatch we
# fall back to the HF cache + checkpoint dir on /home.
model_path = os.environ.get("WAN_MODEL") or snapshot_download(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers", local_files_only=True,
)
print(f"[resolve] model_path={model_path}")
cn_checkpoint_path = os.environ.get(
    "CHECKPOINT",
    "/home/s2710099/checkpoints/wan-beta/beta-008_final.safetensors",
)
cn_config_repo = os.environ.get(
    "HED_CONFIG",
    "/home/s2710099/.cache/huggingface/hub/models--TheDenk--wan2.2-t2v-a14b-controlnet-hed-v1/snapshots/88da8b028eb64cd2c159478c8d722e5f4aa940f6",
)

# VAE in float32 — matches the original WanPTDPipeline grid-search script
vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
print('--- VAE loaded ---')

tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
text_encoder = UMT5EncoderModel.from_pretrained(
    model_path, subfolder="text_encoder", torch_dtype=dtype,
).eval()
print('--- text encoder loaded ---')

# High-noise expert: MUST be CustomWanTransformer3DModel (it's the one that
# accepts controlnet_states / controlnet_weight / controlnet_stride / teacache).
transformer = CustomWanTransformer3DModel.from_pretrained(
    model_path, subfolder="transformer", torch_dtype=dtype,
).eval()
# Low-noise expert: stock WanTransformer3DModel — matches model_index.json.
# Safe because the in_low_noise_stage guard in WanCNPTDPipeline never passes
# controlnet kwargs to transformer_2. DO NOT remove that guard without also
# switching this to CustomWanTransformer3DModel.
transformer_2 = WanTransformer3DModel.from_pretrained(
    model_path, subfolder="transformer_2", torch_dtype=dtype,
).eval()
print('--- transformers loaded ---')

# UniPCMultistepScheduler is what the snapshot's model_index.json declares —
# matches what the original WanPTDPipeline got via .from_pretrained.
scheduler = UniPCMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")
print('--- scheduler loaded ---')

# ControlNet: architecture from HED config repo, trained weights from .safetensors
# Use beta-008_final.safetensors (EMA, canonical) — not _raw (debug only).
cn_config = WanControlnet.load_config(cn_config_repo)
controlnet = WanControlnet.from_config(cn_config)
cast_respecting_fp32_modules(controlnet, dtype)
sd = load_file(cn_checkpoint_path)
missing, unexpected = controlnet.load_state_dict(sd, strict=False)
if missing:    print(f"[warn] missing CN keys: {len(missing)}")
if unexpected: print(f"[warn] unexpected CN keys: {len(unexpected)}")
controlnet.eval()
print(f'--- controlnet loaded from {cn_checkpoint_path} ---')

boundary_ratio, _src = detect_boundary_ratio(model_path, dict(transformer.config))
print(f"[detect] boundary_ratio={boundary_ratio} ({_src})")
assert transformer.dtype == transformer_2.dtype, (
    f"dtype mismatch: transformer={transformer.dtype}, transformer_2={transformer_2.dtype}"
)

pipe = WanCNPTDiffusionPipeline(
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    transformer=transformer,
    transformer_2=transformer_2,
    vae=vae,
    controlnet=controlnet,
    scheduler=scheduler,
    boundary_ratio=boundary_ratio,
)
print('--- WanCN PT Diffusion pipeline assembled ---')
pipe.enable_model_cpu_offload()

# Pin CN to GPU and strip its accelerate hook — accelerate re-attaches per
# __call__, so we re-strip inside the run loop too.
remove_hook_from_module(pipe.controlnet, recurse=True)
pipe.controlnet.to("cuda")
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
asset_names_list = ["face_0", "face_1", "face_2", "face_3", "face_4"]
prompt_list = [prompt1,prompt2]
inital_alpha_list = [ 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
transfer_steps_list= [(5,2), (10,5),(15,7), (20,10), (25,12), (30,15), (45,22)]
ref_latents_dir_list = ["/home/s2710099/cache/wan-beta/deterministic_invert_faces_528x528x61/face_0", 
                        "/home/s2710099/cache/wan-beta/deterministic_invert_faces_528x528x61/face_1"] 
                        #"/home/s2710099/cache/wan-beta/deterministic_invert_faces_528x528x61/face_2", 
                        #"/home/s2710099/cache/wan-beta/deterministic_invert_faces_528x528x61/face_3", 
                        #"/home/s2710099/cache/wan-beta/deterministic_invert_faces_528x528x61/face_4"]

# Heuristic grid search

direct_transfer_steps= 45
decayed_transfer_steps = 22
initial_alpha = 0.4
energy_target_ratios_list = [0.9, 0.95, 0.97, 0.99]


for pidx, prompt in enumerate(prompt_list):
    for idx, ref_dir in enumerate(ref_latents_dir_list):
        run_name = f"{asset_names_list[idx]}_prompt{pidx+1}"
        loaded_image = PIL.Image.open(f"/home/s2710099/data/wan-beta/input_faces/{asset_names_list[idx]}.png").convert("RGB")
        controlnet_frames = ([loaded_image] * num_frames)
        with wandb.init(
            project="CNPTD_manual",
            name=run_name,
            config={
                "description": "Looking for the best measurement for the PI controller in WanPTD.",
                "asset_name": asset_names_list[idx],
                "prompt_idx": pidx+1,
                "prompt": prompt,
                "ref_latents_dir": ref_dir,
                "direct_transfer_steps": direct_transfer_steps,
                "decayed_transfer_steps": decayed_transfer_steps,
                "initial_alpha": initial_alpha,
                "Kp": 0.5,
                "Ki": 0.2,
                "max_alpha_delta": 0.05,
            }
        ) as run:
            # Re-pin CN to GPU — accelerate re-attaches the offload hook per __call__
            remove_hook_from_module(pipe.controlnet, recurse=True)
            pipe.controlnet.to("cuda")
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
                        use_blending_heuristic_version_3=True,
                        use_blending_heuristic_version_4=False,
                        Kp=0.5,
                        Ki=0.2,
                        max_alpha_delta=0.05,
                    #ControlNet params
                        controlnet_frames=controlnet_frames
                )


            frames = video.get('frames', video.get('images', video))

            frames = frames[0]  # Remove batch dimension -> [frames, height, width, channels]
            frame_list = [frames[i] for i in range(frames.shape[0])]

            export_to_video(frame_list, f"/home/s2710099/outputs/inference/cnptd_manual/{run_name}.mp4")
    
wandb.finish()