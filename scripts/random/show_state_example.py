from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize stacked frames for CarRacing preprocessing.")
    parser.add_argument("--env-id", type=str, default="CarRacing-v3")
    parser.add_argument("--num-stack", type=int, default=4, help="Number of frames stacked together.")
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=0,
        help="Frames skipped between stacked frames (e.g., 4 means each channel is 5 frames apart).",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=1.5,
        help="Seconds to roll the environment before capturing the observation.",
    )
    parser.add_argument(
        "--accelerate-seconds",
        type=float,
        default=2.0,
        help="Seconds spent accelerating during warmup (rest is coasting).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = gym.make(args.env_id, render_mode="rgb_array")
    env = CarRacingPreprocess(env)
    env = FrameStackWrapper(env, num_stack=args.num_stack, frame_skip=args.frame_skip)
    env = OffRoadPenaltyWrapper(env)

    obs, info = env.reset()

    dt = getattr(env.unwrapped.unwrapped.unwrapped, "dt", 0.02)
    total_steps = int(round(args.warmup_seconds / dt))
    accel_seconds = max(0.0, min(args.accelerate_seconds, args.warmup_seconds))
    coast_seconds = args.warmup_seconds - accel_seconds
    accel_start = max(1, int(round(coast_seconds / dt)))
    accel_end = total_steps

    current_obs = obs
    for step in range(1, total_steps + 1):
        if accel_start <= step <= accel_end:
            action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        current_obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            current_obs, info = env.reset()

    channel, height, width = current_obs.shape
    assert channel == args.num_stack, "Unexpected observation shape from FrameStackWrapper"

    stride_frames = args.frame_skip + 1
    temporal_gap = stride_frames * dt

    print("Observation shape after warm-up:", current_obs.shape)
    print("Frame gap between stacked channels: %.3f s (~%d simulator frames)" % (temporal_gap, stride_frames))
    print("Off-road flag:", info.get("offroad"))

    fig, axes = plt.subplots(1, args.num_stack, figsize=(3 * args.num_stack, 3))
    if args.num_stack == 1:
        axes = [axes]
    fig.suptitle(
        f"Stacked grayscale frames (rightmost = most recent)\n"
        f"num_stack={args.num_stack}, frame_skip={args.frame_skip}"
    )

    for idx in range(args.num_stack):
        frame = current_obs[idx]
        axes[idx].imshow(frame, cmap="gray", vmin=0.0, vmax=1.0)
        axes[idx].axis("off")
        axes[idx].set_title(f"Frame {idx + 1}\nΔt={(args.num_stack-idx-1)*temporal_gap:.2f}s")

    plt.tight_layout()
    plt.show()

    env.close()


if __name__ == "__main__":
    main()
