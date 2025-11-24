from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import imageio.v2 as imageio
import numpy as np
import torch
import cv2
try:
    from torch.serialization import add_safe_globals
except ImportError:  # pragma: no cover - compatibility
    add_safe_globals = None

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from latent.pca_ppo import PCAPPOConfig  # noqa: E402
from latent.pca_ppo.agent import PCAPPOAgent  # noqa: E402
from latent.pca_ppo.env import PCAObservationWrapper, create_pca_single_env  # noqa: E402
from latent.greyscale import load_greyscale_preset  # noqa: E402


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
    parser.add_argument(
        "--compare-reconstruction",
        action="store_true",
        help="If set, concatenates the PCA reconstruction next to the raw frame.",
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
        "greyscale_presets_path",
    }
    for field in path_fields:
        if field in converted and converted[field] is not None:
            converted[field] = Path(converted[field])
    return PCAPPOConfig(**converted)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[PCAPPOConfig, Dict[str, Any]]:
    if add_safe_globals is not None:
        add_safe_globals([Path])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    greyscale_preset = None
    if config.greyscale_label and config.greyscale_presets_path:
        preset_path = (ROOT_DIR / config.greyscale_presets_path).resolve()
        greyscale_preset = load_greyscale_preset(preset_path, config.greyscale_label)

    env = create_pca_single_env(
        config.env_id,
        args.seed,
        render_mode="rgb_array",
        pca_model_path=(ROOT_DIR / config.pca_model_path).resolve(),
        crop_ratio=config.crop_ratio,
        target_height=config.resize_height,
        target_width=config.resize_width,
        num_stack=config.num_stack,
        frame_skip_between_frames=config.frame_skip,
        offroad_penalty=config.offroad_penalty,
        max_offroad_seconds=config.max_offroad_seconds,
        continuous=config.continuous,
        greyscale_preset=greyscale_preset,
    )

    obs_space = env.observation_space
    action_space = env.action_space
    agent = PCAPPOAgent(obs_space, action_space, config)
    agent.load_state_dict(agent_state)

    frames = []
    obs, _ = env.reset(seed=args.seed)
    frame = env.render()
    if frame is not None:
        frames.append(_prepare_frame(frame, env, args.compare_reconstruction))

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
            frames.append(_prepare_frame(frame, env, args.compare_reconstruction))
        steps += 1

    env.close()

    if not frames:
        raise RuntimeError("No frames captured; ensure render_mode='rgb_array' is supported.")

    imageio.mimsave(output_path, frames, duration=1 / args.fps)
    print(f"Saved GIF to {output_path}")


def _prepare_frame(frame: np.ndarray, env: Any, compare: bool) -> np.ndarray:
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        if array.max(initial=0.0) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)

    if not compare:
        return array

    wrapper = _find_pca_wrapper(env)
    if wrapper is None:
        return array

    latent_stack = getattr(wrapper, "_latent_stack", None)
    if not latent_stack:
        return array

    latest_latent = latent_stack[-1]
    recon = wrapper.reconstruct_from_latent(latest_latent)
    recon_rgb = (recon * 255.0).clip(0, 255).astype(np.uint8)
    recon_rgb = _ensure_three_channels(recon_rgb)
    recon_rgb = cv2.resize(recon_rgb, (array.shape[1], array.shape[0]))
    combined = np.concatenate([array, recon_rgb], axis=1)
    return combined


def _find_pca_wrapper(env: Any) -> PCAObservationWrapper | None:
    current = env
    while current is not None:
        if isinstance(current, PCAObservationWrapper):
            return current
        current = getattr(current, "env", None)
    return None


def _ensure_three_channels(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1)
    if image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    return image


if __name__ == "__main__":
    main()
