"""Training recipe for the ControlNet.

Single source of truth for every hyperparameter that defines the published
recipe. Path-like and run-identity arguments (cache_dir, base_model_path,
output_dir, wandb_project, wandb_run_name) live on the CLI of ``train.py``
since they vary per machine and per run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TrainConfig:
    # ---- Negative prompt (kept in sync with inference) -------------------
    negative_prompt: str = (
        "blurry, low quality, worst quality, jpeg artifacts, text, subtitles, "
        "watermark, static image, still frame, distorted anatomy, "
        "inconsistent motion"
    )

    # ---- Model resolution ------------------------------------------------
    num_frames: int = 9
    height: int = 512
    width: int = 512

    # ---- Optimizer -------------------------------------------------------
    lr: float = 5e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # ---- Schedule (1 epoch over 10 000 face×prompt pairs) ----------------
    num_epochs: int = 1
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 32

    # ---- EMA -------------------------------------------------------------
    ema_decay: float = 0.99
    ema_update_after_step: int = 10
    loss_ema_window: int = 20

    # ---- ControlNet architecture ----------------------------------------
    # num_cn_layers = None reuses the HED config default (6 layers).
    controlnet_stride: int = 3
    num_cn_layers: Optional[int] = None

    # ---- Self-distillation (live ↔ EMA consistency) ----------------------
    use_self_distillation: bool = True
    lambda_consistency: float = 0.5

    # ---- D1: spatial face-weighted FM loss ------------------------------
    # FM-loss weight per latent position = 1 + alpha * silhouette_mask.
    face_weight_alpha: float = 2.0

    # ---- Flow-matching sampling ------------------------------------------
    num_train_timesteps_for_sampling: int = 1000
    boundary_ratio_override: Optional[float] = None

    # ---- Runtime guardrails ----------------------------------------------
    memory_tripwire_gb: float = 90.0
    num_workers: int = 2
    seed: int = 42
