from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np

from ppo_clip import PPOTrainer
from latent.greyscale import load_greyscale_preset
from latent.paths import ROOT_DIR

from .config import PCAPPOConfig
from .env import PCAObservationWrapper, create_pca_single_env, create_pca_vector_env


def _scaled_dimension(value: int, resize_level: int | None) -> int:
    if not resize_level or resize_level <= 1:
        return value
    return max(1, int(math.ceil(value / resize_level)))


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
            preset = load_greyscale_preset(preset_path, config.greyscale_label)
            if config.resize_level and config.resize_level > 1:
                preset = replace(
                    preset,
                    output_height=_scaled_dimension(preset.output_height, config.resize_level),
                    output_width=_scaled_dimension(preset.output_width, config.resize_level),
                )
            self.greyscale_preset = preset

        self._pca_model_path = (self.project_root / config.pca_model_path).resolve()

        frame_transform = self._transform_frame if config.compare_reconstruction else None

        self._trainer = PPOTrainer(
            config,
            vector_env_builder=self._build_vector_env,
            single_env_builder=self._build_single_env,
            frame_transform=frame_transform,
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

    def _transform_frame(self, frame: np.ndarray, env: gym.Env) -> np.ndarray:
        array = np.asarray(frame)
        if array.dtype != np.uint8:
            scale = 255.0 if array.max(initial=0.0) <= 1.0 else 1.0
            array = np.clip(array * scale, 0, 255).astype(np.uint8)

        wrapper = self._find_pca_wrapper(env)
        if wrapper is None:
            return array

        latent_stack = getattr(wrapper, "_latent_stack", None)
        if not latent_stack:
            return array

        latest_latent = latent_stack[-1]
        recon = wrapper.reconstruct_from_latent(latest_latent)
        recon_rgb = self._ensure_three_channels(recon)
        if recon_rgb.dtype != np.uint8:
            recon_rgb = np.clip(recon_rgb * 255.0, 0, 255).astype(np.uint8)
        recon_rgb = cv2.resize(recon_rgb, (array.shape[1], array.shape[0]))
        return np.concatenate([array, recon_rgb], axis=1)

    @staticmethod
    def _find_pca_wrapper(env: gym.Env) -> PCAObservationWrapper | None:
        current = env
        while current is not None:
            if isinstance(current, PCAObservationWrapper):
                return current
            current = getattr(current, "env", None)
        return None

    @staticmethod
    def _ensure_three_channels(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return np.stack([image] * 3, axis=-1)
        if image.shape[2] == 1:
            return np.repeat(image, 3, axis=2)
        return image
