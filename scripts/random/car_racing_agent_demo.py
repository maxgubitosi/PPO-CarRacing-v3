from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    from torch.serialization import add_safe_globals
except ImportError:
    add_safe_globals = None

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_CHECKPOINT = Path(
    "/Users/gaborgorondi/Documents/udesa/9_rl/RL-TPs/TPF/PPO-CarRacing-v3/results/models/"
    "ppo_clip/ppo_clip_20251124-161508/ppo_clip_update_14600.pt"
)

from environment import create_single_env
from ppo_clip import PPOConfig
from ppo_clip.agent import PPOClipAgent
from utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a pretrained PPO agent play CarRacing-v3.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to a PPO checkpoint (.pt).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device preference for inference.",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Episodes to run.")
    parser.add_argument("--seed", type=int, default=1234, help="Base seed for the environment.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on steps per episode.")
    return parser.parse_args()


def _dict_to_config(config_dict: dict) -> PPOConfig:
    converted = dict(config_dict)
    for field in ("log_root", "checkpoint_root", "video_root"):
        value = converted.get(field)
        if value is not None:
            converted[field] = Path(value)
    return PPOConfig(**converted)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[PPOConfig, dict]:
    if add_safe_globals is not None:
        add_safe_globals([Path])
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if "config" not in checkpoint or "agent" not in checkpoint:
        raise ValueError("Checkpoint missing required keys 'config' and 'agent'.")
    return _dict_to_config(checkpoint["config"]), checkpoint["agent"]


def build_env(config: PPOConfig, seed: int):
    return create_single_env(
        config.env_id,
        seed,
        render_mode="human",
        offroad_penalty=config.offroad_penalty,
        max_offroad_seconds=config.max_offroad_seconds,
        continuous=config.continuous,
        frame_skip_between_frames=config.frame_skip,
        num_stack=config.num_stack,
        steering_constraint=config.steering_constraint,
    )


def play_episode(
    env,
    agent: PPOClipAgent,
    device: torch.device,
    *,
    seed: int | None,
    max_steps: int | None,
) -> tuple[float, int]:
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    done = False
    cap = max_steps if max_steps and max_steps > 0 else None

    while not done and (cap is None or steps < cap):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, _ = agent.act_deterministic(obs_tensor)
        obs, reward, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
        env.render()
        total_reward += float(reward)
        steps += 1
        done = terminated or truncated

    return total_reward, steps


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    device_str = resolve_device(args.device)
    device = torch.device(device_str)

    config, agent_state = load_checkpoint(checkpoint_path, device)
    config.device = device_str

    env = build_env(config, args.seed)
    agent = PPOClipAgent(env.observation_space, env.action_space, config)
    agent.load_state_dict(agent_state)
    agent.network.eval()

    episodes = max(1, int(args.episodes))
    max_steps = args.max_steps if args.max_steps and args.max_steps > 0 else None

    try:
        for idx in range(episodes):
            episode_seed = args.seed + idx if args.seed is not None else None
            reward, steps = play_episode(env, agent, device, seed=episode_seed, max_steps=max_steps)
            print(f"Episode {idx + 1}/{episodes} – reward={reward:.2f}, steps={steps}")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()

