from __future__ import annotations

from pathlib import Path

from ppo_clip import PPOTrainer
from latent.greyscale import load_greyscale_preset
from latent.paths import ROOT_DIR

from .config import PCAPPOConfig
from .env import create_pca_single_env, create_pca_vector_env


class PCAPPOTrainer:
    """Adaptador que reutiliza el PPOTrainer estándar con entornos PCA."""

    def __init__(self, config: PCAPPOConfig) -> None:
        self.config = config
        self.project_root = ROOT_DIR
        self.greyscale_preset = None
        if config.greyscale_label:
            if config.greyscale_presets_path is None:
                raise ValueError("greyscale_presets_path es obligatorio cuando se usa greyscale_label.")
            preset_path = (self.project_root / config.greyscale_presets_path).resolve()
            self.greyscale_preset = load_greyscale_preset(preset_path, config.greyscale_label)

        self._pca_model_path = (self.project_root / config.pca_model_path).resolve()

        self._trainer = PPOTrainer(
            config,
            vector_env_builder=self._build_vector_env,
            single_env_builder=self._build_single_env,
        )

    def train(self) -> None:
        self._trainer.train()

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        self._trainer.load_checkpoint(checkpoint_path)

    def __getattr__(self, name):
        return getattr(self._trainer, name)

    def _build_vector_env(self, cfg: PCAPPOConfig):
        return create_pca_vector_env(
            cfg.env_id,
            cfg.num_envs,
            cfg.seed,
            pca_model_path=self._pca_model_path,
            crop_ratio=cfg.crop_ratio,
            target_height=cfg.resize_height,
            target_width=cfg.resize_width,
            num_stack=cfg.num_stack,
            frame_skip_between_frames=cfg.frame_skip,
            offroad_penalty=cfg.offroad_penalty,
            max_offroad_seconds=cfg.max_offroad_seconds,
            continuous=cfg.continuous,
            greyscale_preset=self.greyscale_preset,
        )

    def _build_single_env(self, cfg: PCAPPOConfig, seed: int, render_mode: str | None):
        return create_pca_single_env(
            cfg.env_id,
            seed,
            render_mode=render_mode,
            pca_model_path=self._pca_model_path,
            crop_ratio=cfg.crop_ratio,
            target_height=cfg.resize_height,
            target_width=cfg.resize_width,
            num_stack=cfg.num_stack,
            frame_skip_between_frames=cfg.frame_skip,
            offroad_penalty=cfg.offroad_penalty,
            max_offroad_seconds=cfg.max_offroad_seconds,
            continuous=cfg.continuous,
            greyscale_preset=self.greyscale_preset,
        )
