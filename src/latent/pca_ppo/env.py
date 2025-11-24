from __future__ import annotations

import pickle
from pathlib import Path
from collections import deque
from typing import Callable

import cv2
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

from environment.carracing import OffRoadPenaltyWrapper
from latent.greyscale import GreyscalePreset


def _load_pca_model(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


class PCAObservationWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        *,
        pca_model_path: Path,
        crop_ratio: float,
        target_height: int,
        target_width: int,
        num_stack: int,
        frame_skip: int = 0,
        greyscale_preset: GreyscalePreset | None = None,
    ) -> None:
        super().__init__(env)
        self.pca_model_path = Path(pca_model_path)
        self.greyscale_preset = greyscale_preset
        if self.greyscale_preset is not None:
            self.crop_ratio = float(self.greyscale_preset.crop_ratio)
            self.target_height = int(self.greyscale_preset.output_height)
            self.target_width = int(self.greyscale_preset.output_width)
            self.channel_count = 1
        else:
            self.crop_ratio = crop_ratio
            self.target_height = target_height
            self.target_width = target_width
            self.channel_count = 3
        self.num_stack = num_stack
        self.frame_skip = max(0, frame_skip)
        self._stride = self.frame_skip + 1
        self._history_length = self._stride * (self.num_stack - 1) + 1

        self._pca = _load_pca_model(self.pca_model_path)
        self.latent_dim = int(getattr(self._pca, "n_components", len(self._pca.components_)))

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.latent_dim * self.num_stack,),
            dtype=np.float32,
        )
        self._latent_history: deque[np.ndarray] = deque(maxlen=self._history_length)
        self._latent_stack = self._latent_history

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        projected, offroad, ratio = self._process(obs)
        info = dict(info)
        info["offroad"] = offroad
        info["offroad_ratio"] = ratio
        self._latent_history.clear()
        for _ in range(self._history_length):
            self._latent_history.append(projected.copy())
        return self._get_stacked_latent(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        projected, offroad, ratio = self._process(obs)
        info = dict(info)
        info["offroad"] = offroad
        info["offroad_ratio"] = ratio
        self._latent_history.append(projected.copy())
        return self._get_stacked_latent(), reward, terminated, truncated, info

    def _get_stacked_latent(self) -> np.ndarray:
        if not self._latent_history:
            raise RuntimeError("Latent stack is empty")
        if len(self._latent_history) < self._history_length:
            raise RuntimeError("Latent history buffer underflow")
        selected = [
            np.array(self._latent_history[-1 - i * self._stride], copy=False)
            for i in range(self.num_stack)
        ]
        selected.reverse()
        stacked = np.concatenate(selected, axis=0)
        return stacked.astype(np.float32, copy=False)

    def _process(self, frame: np.ndarray) -> tuple[np.ndarray, bool, float]:
        offroad, ratio = self._detect_offroad(frame)
        latent = self._project_frame(frame)
        return latent, offroad, ratio

    def _project_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.greyscale_preset is not None:
            processed = self.greyscale_preset.apply(frame, normalize=True, keepdims=True)
        else:
            processed = self._process_rgb_frame(frame)
        flat = processed.reshape(1, -1)
        latent = self._pca.transform(flat).astype(np.float32)
        return latent.squeeze(0)

    def _process_rgb_frame(self, frame: np.ndarray) -> np.ndarray:
        height = frame.shape[0]
        crop_pixels = int(round(height * max(0.0, self.crop_ratio)))
        if crop_pixels > 0 and crop_pixels < height:
            frame = frame[: height - crop_pixels, :, :]

        resized = cv2.resize(
            frame,
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_AREA,
        )
        normalized = resized.astype(np.float32) / 255.0
        return normalized

    def reconstruct_from_latent(self, latent: np.ndarray) -> np.ndarray:
        latent = np.asarray(latent, dtype=np.float32)
        if latent.shape[0] > self.latent_dim:
            latent = latent[-self.latent_dim :]
        latent = latent.reshape(1, -1)
        reconstruction = self._pca.inverse_transform(latent)
        reconstruction = reconstruction.reshape(self.target_height, self.target_width, self.channel_count)
        reconstruction = np.clip(reconstruction, 0.0, 1.0)
        if reconstruction.shape[2] == 1:
            reconstruction = reconstruction[..., 0]
        return reconstruction

    def _detect_offroad(self, frame: np.ndarray) -> tuple[bool, float]:
        frame_norm = frame.astype(np.float32) / 255.0
        red = frame_norm[..., 0]
        green = frame_norm[..., 1]
        blue = frame_norm[..., 2]

        green_dominant = (green > 0.35) & (green > red + 0.1) & (green > blue + 0.1)
        offroad_ratio = float(np.mean(green_dominant))

        road_like = (
            (np.abs(red - green) < 0.05)
            & (np.abs(green - blue) < 0.05)
            & (green < 0.8)
        )
        road_ratio = float(np.mean(road_like))

        offroad = offroad_ratio > 0.3 and road_ratio < 0.2
        return offroad, offroad_ratio


def _make_env(
    env_id: str,
    seed: int,
    render_mode: str | None,
    *,
    pca_model_path: Path,
    crop_ratio: float,
    target_height: int,
    target_width: int,
    num_stack: int,
    frame_skip_between_frames: int = 0,
    offroad_penalty: float | None,
    max_offroad_seconds: float,
    continuous: bool = True,
    greyscale_preset: GreyscalePreset | None = None,
) -> Callable[[], gym.Env]:
    def thunk() -> gym.Env:
        env = gym.make(
            env_id,
            render_mode=render_mode,
            continuous=continuous,
            lap_complete_percent=0.95,
            domain_randomize=False,
        )
        env = PCAObservationWrapper(
            env,
            pca_model_path=pca_model_path,
            crop_ratio=crop_ratio,
            target_height=target_height,
            target_width=target_width,
            num_stack=num_stack,
            frame_skip=frame_skip_between_frames,
            greyscale_preset=greyscale_preset,
        )
        env = OffRoadPenaltyWrapper(
            env,
            max_offroad_seconds=max_offroad_seconds,
            penalty=offroad_penalty,
        )
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env

    return thunk


def create_pca_vector_env(
    env_id: str,
    num_envs: int,
    seed: int,
    *,
    pca_model_path: Path,
    crop_ratio: float,
    target_height: int,
    target_width: int,
    num_stack: int,
    frame_skip_between_frames: int = 0,
    offroad_penalty: float | None,
    max_offroad_seconds: float,
    continuous: bool = True,
    greyscale_preset: GreyscalePreset | None = None,
) -> SyncVectorEnv | AsyncVectorEnv:
    env_fns = [
        _make_env(
            env_id,
            seed + i,
            render_mode=None,
            pca_model_path=pca_model_path,
            crop_ratio=crop_ratio,
            target_height=target_height,
            target_width=target_width,
            num_stack=num_stack,
            frame_skip_between_frames=frame_skip_between_frames,
            offroad_penalty=offroad_penalty,
            max_offroad_seconds=max_offroad_seconds,
            continuous=continuous,
            greyscale_preset=greyscale_preset,
        )
        for i in range(num_envs)
    ]
    if num_envs > 1:
        return AsyncVectorEnv(env_fns)
    return SyncVectorEnv(env_fns)


def create_pca_single_env(
    env_id: str,
    seed: int,
    render_mode: str | None,
    *,
    pca_model_path: Path,
    crop_ratio: float,
    target_height: int,
    target_width: int,
    num_stack: int,
    frame_skip_between_frames: int = 0,
    offroad_penalty: float | None,
    max_offroad_seconds: float,
    continuous: bool,
    greyscale_preset: GreyscalePreset | None = None,
) -> gym.Env:
    env_fn = _make_env(
        env_id,
        seed,
        render_mode=render_mode,
        pca_model_path=pca_model_path,
        crop_ratio=crop_ratio,
        target_height=target_height,
        target_width=target_width,
        num_stack=num_stack,
        frame_skip_between_frames=frame_skip_between_frames,
        offroad_penalty=offroad_penalty,
        max_offroad_seconds=max_offroad_seconds,
        continuous=continuous,
        greyscale_preset=greyscale_preset,
    )
    return env_fn()
