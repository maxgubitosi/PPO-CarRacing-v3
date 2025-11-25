from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ppo_clip.config import PPOConfig


@dataclass
class PCAPPOConfig(PPOConfig):
    pca_model_path: Path = field(
        default_factory=lambda: Path("scripts/latent_space_experiment/models/pca/dim_012/pca_model.pkl")
    )
    num_stack: int = 4
    crop_ratio: float = 0.13
    resize_height: int = 48
    resize_width: int = 48
    ridge_lambda: float = 1e-3
    log_root: Path = field(default_factory=lambda: Path("tensorboard_logs/pca_ppo"))
    checkpoint_root: Path = field(default_factory=lambda: Path("models/pca_ppo"))
    video_root: Path = field(default_factory=lambda: Path("videos/pca_ppo"))
    greyscale_presets_path: Path | None = None
    greyscale_label: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_stack <= 0:
            raise ValueError("num_stack must be positive")
        if self.resize_height <= 0 or self.resize_width <= 0:
            raise ValueError("resize dimensions must be positive")
        if self.crop_ratio < 0 or self.crop_ratio >= 1:
            raise ValueError("crop_ratio must be between 0 (inclusive) and 1 (exclusive)")
        if self.ridge_lambda < 0:
            raise ValueError("ridge_lambda must be non-negative")
        if self.greyscale_label and not self.greyscale_presets_path:
            raise ValueError("greyscale_presets_path is required when greyscale_label is provided")
        self.weight_decay = self.ridge_lambda
