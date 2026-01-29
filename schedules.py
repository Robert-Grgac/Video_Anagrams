from __future__ import annotations
from dataclasses import dataclass


@dataclass
class AnagramSchedule:
    """
    Scheduling policy for the WanAnagramPipeline.

    Attributes:
        joint_steps (int):
            Number of initial denoising steps that use joint diffusion
            (MatchDiffusion-style shared structure).
    """
    joint_steps: int

    def is_joint(self, step_index: int) -> bool:
        """
        Convenience helper: True if we are in the joint stage.
        """
        return step_index < self.joint_steps

    def is_anagram(self, step_index: int) -> bool:
        """
        Convenience helper: True if we are in the anagram stage.
        """
        return step_index >= self.joint_steps
