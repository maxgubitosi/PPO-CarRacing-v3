from __future__ import annotations

from typing import Callable

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv


class CarRacingPreprocess(ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        h, w, c = self.observation_space.shape
        if c != 3:
            raise ValueError("Expected RGB observations")

        self.target_height = h // 4
        self.target_width = w // 4
        if self.target_height == 0 or self.target_width == 0:
            raise ValueError("Target resize dimensions must be positive")

        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(c, self.target_height, self.target_width),
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        obs = observation.astype(np.float32) / 255.0
        resized = cv2.resize(
            obs,
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_AREA,
        )
        return np.transpose(resized, (2, 0, 1))


def _make_env(env_id: str, seed: int, render_mode: str | None) -> Callable[[], gym.Env]:
    def thunk() -> gym.Env:
        env = gym.make(env_id, render_mode=render_mode)
        env = CarRacingPreprocess(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env

    return thunk


def create_vector_env(env_id: str, num_envs: int, seed: int, render_mode: str | None = None) -> gym.Env:
    env_fns = [_make_env(env_id, seed + idx, render_mode if render_mode and idx == 0 else None) for idx in range(num_envs)]
    if num_envs == 1:
        return SyncVectorEnv(env_fns)
    return AsyncVectorEnv(env_fns, shared_memory=False)


def create_single_env(env_id: str, seed: int, render_mode: str | None = None) -> gym.Env:
    return _make_env(env_id, seed, render_mode)()


__all__ = ["create_vector_env", "create_single_env", "CarRacingPreprocess"]

