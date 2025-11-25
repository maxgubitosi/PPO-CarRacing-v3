from __future__ import annotations

from collections import deque
from typing import Callable

import cv2
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv


class CarRacingPreprocess(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        original_space = self.env.observation_space
        if not isinstance(original_space, Box) or len(original_space.shape) != 3:
            raise ValueError("CarRacingPreprocess expects an RGB Box observation")

        h, w, c = original_space.shape
        if c != 3:
            raise ValueError("Expected RGB observations")

        self.crop_ratio = 0.13
        self.crop_pixels = int(round(h * self.crop_ratio))
        self.cropped_height = h - self.crop_pixels
        if self.cropped_height <= 0:
            raise ValueError("Crop removes entire image")

        self.target_height = self.cropped_height // 2
        self.target_width = w // 2
        if self.target_height == 0 or self.target_width == 0:
            raise ValueError("Target resize dimensions must be positive")

        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(1, self.target_height, self.target_width),
            dtype=np.float32,
        )

        self._last_offroad_ratio: float = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        processed, offroad = self._process(obs)
        info = dict(info)
        info["offroad"] = offroad
        info["offroad_ratio"] = self._last_offroad_ratio
        return processed, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        processed, offroad = self._process(obs)
        info = dict(info)
        info["offroad"] = offroad
        info["offroad_ratio"] = self._last_offroad_ratio
        return processed, reward, terminated, truncated, info

    def _process(self, frame: np.ndarray) -> tuple[np.ndarray, bool]:
        offroad = self._detect_offroad(frame)
        if self.crop_pixels > 0:
            frame = frame[: self.cropped_height, :, :]
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        downsampled = gray[::2, ::2]
        normalized = downsampled.astype(np.float32) / 255.0
        return normalized[np.newaxis, ...], offroad

    def _detect_offroad(self, frame: np.ndarray) -> bool:
        frame_norm = frame.astype(np.float32) / 255.0
        red = frame_norm[..., 0]
        green = frame_norm[..., 1]
        blue = frame_norm[..., 2]

        green_dominant = (green > 0.35) & (green > red + 0.1) & (green > blue + 0.1)
        self._last_offroad_ratio = float(np.mean(green_dominant))

        road_like = (
            (np.abs(red - green) < 0.05)
            & (np.abs(green - blue) < 0.05)
            & (green < 0.8)
        )
        road_ratio = float(np.mean(road_like))

        return self._last_offroad_ratio > 0.3 and road_ratio < 0.2


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, num_stack: int, frame_skip: int = 0) -> None:
        super().__init__(env)
        if num_stack < 1:
            raise ValueError("num_stack must be at least 1")
        self.num_stack = num_stack
        self.frame_skip = max(0, frame_skip)
        self._stride = self.frame_skip + 1
        self._history_length = self._stride * (num_stack - 1) + 1
        self.frames: deque[np.ndarray] = deque(maxlen=self._history_length)

        obs_space = self.env.observation_space
        if not isinstance(obs_space, Box):
            raise ValueError("FrameStackWrapper requires a Box observation space")

        low = np.repeat(obs_space.low, num_stack, axis=0)
        high = np.repeat(obs_space.high, num_stack, axis=0)
        self.observation_space = Box(low=low, high=high, dtype=obs_space.dtype)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.frames.clear()
        for _ in range(self._history_length):
            self.frames.append(np.array(obs, copy=True))
        return self._get_observation(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(np.array(obs, copy=True))
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        assert len(self.frames) >= self._history_length, "FrameStackWrapper history buffer underflow"
        selected = [
            np.array(self.frames[-1 - i * self._stride], copy=False)
            for i in range(self.num_stack)
        ]
        selected.reverse()
        return np.concatenate(selected, axis=0)


class OffRoadPenaltyWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        max_offroad_seconds: float = 2.0,
        penalty: float | None =  None,
    ) -> None:
        super().__init__(env)
        dt = getattr(self.env.unwrapped, "dt", None)
        if dt is None or dt <= 0:
            dt = 0.02
        self.offroad_limit = max(1, int(round(max_offroad_seconds / dt)))
        self._offroad_steps = 0
        self.penalty = penalty

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._offroad_steps = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        offroad = bool(info.get("offroad", False))
        if offroad:
            self._offroad_steps += 1
        else:
            self._offroad_steps = 0

        truncated_due_to_offroad = False
        if self._offroad_steps >= self.offroad_limit:
            truncated = True
            truncated_due_to_offroad = True
            info = dict(info)
            info["offroad_truncated"] = True
            self._offroad_steps = 0

        if self.penalty is not None and truncated_due_to_offroad:
            reward -= abs(self.penalty)

        if self.penalty is not None and offroad and not truncated_due_to_offroad:
            reward -= abs(self.penalty) * 0.1

        if terminated or truncated:
            self._offroad_steps = 0

        return obs, reward, terminated, truncated, info


def _make_env(
    env_id: str,
    seed: int,
    render_mode: str | None,
    *,
    offroad_penalty: float | None,
    max_offroad_seconds: float = 2.0,
    continuous: bool = True,
    frame_skip_between_frames: int = 0,
    num_stack: int = 4,
) -> Callable[[], gym.Env]:
    """
    Crea un environment CarRacing-v3.
    
    Args:
        continuous: Si True, usa acciones continuas Box(3,). 
                   Si False, usa acciones discretas Discrete(5) nativas del environment.
    """
    def thunk() -> gym.Env:
        # Crear environment con el parámetro continuous nativo
        env = gym.make(
            env_id, 
            render_mode=render_mode,
            continuous=continuous,
            lap_complete_percent=0.95,
            domain_randomize=False,
        )
        
        env = CarRacingPreprocess(env)
        env = FrameStackWrapper(
            env,
            num_stack=num_stack,
            frame_skip=frame_skip_between_frames,
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


def create_vector_env(
    env_id: str,
    num_envs: int,
    seed: int,
    render_mode: str | None = None,
    *,
    offroad_penalty: float | None = None,
    max_offroad_seconds: float = 2.0,
    continuous: bool = True,
    frame_skip_between_frames: int = 0,
    num_stack: int = 4,
) -> gym.Env:
    env_fns = [
        _make_env(
            env_id,
            seed + idx,
            render_mode if render_mode and idx == 0 else None,
            offroad_penalty=offroad_penalty,
            max_offroad_seconds=max_offroad_seconds,
            continuous=continuous,
            frame_skip_between_frames=frame_skip_between_frames,
            num_stack=num_stack,
        )
        for idx in range(num_envs)
    ]
    if num_envs == 1:
        return SyncVectorEnv(env_fns)
    return AsyncVectorEnv(env_fns, shared_memory=False)


def create_single_env(
    env_id: str,
    seed: int,
    render_mode: str | None = None,
    *,
    offroad_penalty: float | None = None,
    max_offroad_seconds: float = 2.0,
    continuous: bool = True,
    frame_skip_between_frames: int = 0,
    num_stack: int = 4,
) -> gym.Env:
    return _make_env(
        env_id,
        seed,
        render_mode,
        offroad_penalty=offroad_penalty,
        max_offroad_seconds=max_offroad_seconds,
        continuous=continuous,
        frame_skip_between_frames=frame_skip_between_frames,
        num_stack=num_stack,
    )()


__all__ = [
    "create_vector_env",
    "create_single_env",
    "CarRacingPreprocess",
    "FrameStackWrapper",
    "OffRoadPenaltyWrapper",
    "OffRoadTruncationWrapper"
]
