from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

try:
    from torch.serialization import add_safe_globals
except ImportError:  # pragma: no cover
    add_safe_globals = None

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from environment import create_single_env  # noqa: E402
from ppo_clip import PPOConfig  # noqa: E402
from ppo_clip.agent import PPOClipAgent  # noqa: E402
from utils import resolve_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a policy GIF from a standard PPO checkpoint, optionally with feature activation maps."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint produced by scripts/training/train_ppo_clip.py (e.g. models/ppo_clip/ppo_clip_update_900.pt).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/random/videos/ppo_clip/policy_sample.gif"),
        help="Where to save the generated GIF.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Seed for the evaluation environment.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum number of steps to record.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run the policy on.",
    )
    parser.add_argument("--fps", type=int, default=15, help="Frames per second for the resulting GIF.")
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=2,
        help="Record one frame every N environment steps (default: 2) to keep GIF size manageable.",
    )
    parser.add_argument(
        "--feature-activation-map",
        action="store_true",
        help="If set, append a heatmap visualization of the first convolution's activations for each stacked frame.",
    )
    return parser.parse_args()


def _dict_to_config(config_dict: Dict[str, Any]) -> PPOConfig:
    converted = dict(config_dict)
    path_fields = {"log_root", "checkpoint_root", "video_root"}
    for field in path_fields:
        if field in converted and converted[field] is not None:
            converted[field] = Path(converted[field])
    return PPOConfig(**converted)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[PPOConfig, Dict[str, Any]]:
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

    env = create_single_env(
        config.env_id,
        args.seed,
        render_mode="rgb_array",
        offroad_penalty=config.offroad_penalty,
        max_offroad_seconds=config.max_offroad_seconds,
    )

    obs_space = env.observation_space
    action_space = env.action_space
    agent = PPOClipAgent(obs_space, action_space, config)
    agent.load_state_dict(agent_state)

    frames: List[np.ndarray] = []
    frame_skip = max(1, args.frame_skip)

    obs, _ = env.reset(seed=args.seed)
    frame = env.render()
    if frame is not None:
        frames.append(_prepare_frame(frame, obs, agent, args.feature_activation_map, device))

    done = False
    steps = 0
    frame_counter = 0
    while not done and steps < args.max_steps:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, _ = agent.act_deterministic(obs_tensor)
        obs, _, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
        done = terminated or truncated
        frame = env.render()
        frame_counter += 1
        if frame is not None and frame_counter % frame_skip == 0:
            frames.append(_prepare_frame(frame, obs, agent, args.feature_activation_map, device))
        steps += 1

    env.close()

    if not frames:
        raise RuntimeError("No frames captured; ensure render_mode='rgb_array' is supported.")

    imageio.mimsave(output_path, frames, duration=1 / args.fps)
    print(f"Saved GIF to {output_path}")


def _prepare_frame(
    frame: np.ndarray,
    obs_stack: np.ndarray,
    agent: PPOClipAgent,
    show_feature_map: bool,
    device: torch.device,
) -> np.ndarray:
    frame_array = np.asarray(frame)
    if frame_array.dtype != np.uint8:
        if frame_array.max(initial=0.0) <= 1.0:
            frame_array = frame_array * 255.0
        frame_array = np.clip(frame_array, 0, 255).astype(np.uint8)

    if not show_feature_map:
        return frame_array

    heatmap_strip = _generate_activation_strip(obs_stack, agent, frame_array.shape[:2], device)
    combined = np.concatenate([frame_array, heatmap_strip], axis=1)
    return combined


def _generate_activation_strip(
    obs_stack: np.ndarray,
    agent: PPOClipAgent,
    target_hw: tuple[int, int],
    device: torch.device,
) -> np.ndarray:
    obs_tensor = torch.as_tensor(obs_stack, dtype=torch.float32, device=device).unsqueeze(0)
    conv_layer = agent.network.feature_extractor[0]
    weight = conv_layer.weight  # (out_channels, in_channels, kH, kW)
    stride = conv_layer.stride
    padding = conv_layer.padding

    height, width = target_hw
    last_channel = obs_tensor[:, -1:, :, :]
    last_weight = weight[:, -1:, :, :]
    activation = F.conv2d(last_channel, last_weight, bias=None, stride=stride, padding=padding)
    activation = torch.relu(activation)
    activation = activation.mean(dim=1, keepdim=True)
    activation = F.interpolate(activation, size=(height, width), mode="bilinear", align_corners=False)
    activation_np = activation.squeeze().detach().cpu().numpy()
    norm = (activation_np - activation_np.min()) / (activation_np.max() - activation_np.min() + 1e-8)
    heat = (norm * 255.0).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return heat_color


if __name__ == "__main__":
    main()
