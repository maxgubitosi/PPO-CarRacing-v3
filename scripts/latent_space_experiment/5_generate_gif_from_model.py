from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import imageio.v2 as imageio
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from latent.pca_ppo import PCAPPOConfig  # noqa: E402
from latent.pca_ppo.agent import PCAPPOAgent  # noqa: E402
from latent.pca_ppo.env import create_pca_single_env  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a policy gif from a PCA PPO checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the checkpoint produced by 4_train_pca_ppo_agent.py (e.g. pca_ppo_update_900.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/latent_space_experiment/videos/ppo_latent/policy_sample.gif"),
        help="Where to save the generated GIF.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Seed for the evaluation environment.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum number of steps to record for the episode.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device on which to run inference.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for the resulting GIF.",
    )
    return parser.parse_args()


def resolve_device(choice: str) -> str:
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dict_to_config(config_dict: Dict[str, Any]) -> PCAPPOConfig:
    converted = dict(config_dict)
    path_fields = {
        "log_root",
        "checkpoint_root",
        "video_root",
        "pca_model_path",
    }
    for field in path_fields:
        if field in converted and converted[field] is not None:
            converted[field] = Path(converted[field])
    return PCAPPOConfig(**converted)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[PCAPPOConfig, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "agent" not in checkpoint or "config" not in checkpoint:
        raise ValueError("Checkpoint missing required keys 'agent' and 'config'.")
    config = _dict_to_config(checkpoint["config"])
    return config, checkpoint["agent"]


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device_str = resolve_device(args.device)
    device = torch.device(device_str)

    config, agent_state = load_checkpoint(checkpoint_path, device)
    config.device = device_str

    env = create_pca_single_env(
        config.env_id,
        args.seed,
        render_mode="rgb_array",
        pca_model_path=(ROOT_DIR / config.pca_model_path).resolve(),
        crop_ratio=config.crop_ratio,
        target_height=config.resize_height,
        target_width=config.resize_width,
        num_stack=config.num_stack,
        offroad_penalty=config.offroad_penalty,
        max_offroad_seconds=config.max_offroad_seconds,
    )

    obs_space = env.observation_space
    action_space = env.action_space
    agent = PCAPPOAgent(obs_space, action_space, config)
    agent.load_state_dict(agent_state)

    frames = []
    obs, _ = env.reset(seed=args.seed)
    frame = env.render()
    if frame is not None:
        frames.append(_prepare_frame(frame))

    done = False
    steps = 0

    while not done and steps < args.max_steps:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, _ = agent.act_deterministic(obs_tensor)
        obs, _, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
        done = terminated or truncated
        frame = env.render()
        if frame is not None:
            frames.append(_prepare_frame(frame))
        steps += 1

    env.close()

    if not frames:
        raise RuntimeError("No frames captured; ensure render_mode='rgb_array' is supported.")

    imageio.mimsave(output_path, frames, duration=1 / args.fps)
    print(f"Saved GIF to {output_path}")


def _prepare_frame(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        if array.max(initial=0.0) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


if __name__ == "__main__":
    main()
