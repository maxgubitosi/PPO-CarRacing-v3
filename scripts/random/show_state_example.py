from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import gymnasium as gym
import numpy as np

from environment import CarRacingPreprocess, FrameStackWrapper, OffRoadPenaltyWrapper


def main() -> None:
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    env = CarRacingPreprocess(env)
    env = FrameStackWrapper(env, num_stack=4)
    env = OffRoadPenaltyWrapper(env)

    obs, info = env.reset()

    dt = getattr(env.unwrapped.unwrapped.unwrapped, "dt", 0.02)
    total_steps = int(round(3.0 / dt))
    accelerate_steps = (int(round(2.0 / dt)), int(round(3.0 / dt)))

    current_obs = obs
    for step in range(1, total_steps + 1):
        if accelerate_steps[0] <= step <= accelerate_steps[1]:
            action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        current_obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            current_obs, info = env.reset()

    print("Observation shape after warm-up:", current_obs.shape)
    print("Off-road flag:", info.get("offroad"))

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    fig.suptitle("Stacked grayscale frames (most recent on the right)")

    for idx in range(4):
        frame = current_obs[idx]
        axes[idx].imshow(frame, cmap="gray", vmin=0.0, vmax=1.0)
        axes[idx].axis("off")
        axes[idx].set_title(f"Frame {idx + 1}")

    plt.tight_layout()
    plt.show()

    env.close()


if __name__ == "__main__":
    main()

