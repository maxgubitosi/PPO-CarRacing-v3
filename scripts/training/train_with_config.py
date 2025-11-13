"""Train PPO agent with configuration from YAML file."""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import yaml

warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from ppo_clip import PPOConfig, PPOTrainer


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO-Clip on CarRacing-v3 with YAML config")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (e.g., configs/ppo_discrete_sota.yaml)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Override: Path to checkpoint (.pt) to resume from",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override: Total timesteps to train",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override: Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default=None,
        help="Override: Device to use",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Load configuration from YAML
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    yaml_config = load_config(config_path)
    print(f"Loaded configuration from: {config_path}")
    print(f"Config: {yaml_config}")
    
    # Override with command-line arguments if provided
    if args.resume is not None:
        yaml_config["resume"] = args.resume
    if args.total_timesteps is not None:
        yaml_config["total_timesteps"] = args.total_timesteps
    if args.seed is not None:
        yaml_config["seed"] = args.seed
    if args.device is not None:
        yaml_config["device"] = args.device
    
    # Convert learning_rate to float if it's a string (YAML sometimes parses scientific notation as string)
    if isinstance(yaml_config["learning_rate"], str):
        yaml_config["learning_rate"] = float(yaml_config["learning_rate"])
    
    # Handle device selection
    device = yaml_config.get("device", "auto")
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    # Create PPO config
    config = PPOConfig(
        total_timesteps=yaml_config["total_timesteps"],
        num_envs=yaml_config["num_envs"],
        num_steps=yaml_config["num_steps"],
        num_stack=yaml_config["num_stack"],
        frame_skip=yaml_config["frame_skip"],
        num_minibatches=yaml_config["num_minibatches"],
        update_epochs=yaml_config["update_epochs"],
        gamma=yaml_config["gamma"],
        gae_lambda=yaml_config["gae_lambda"],
        clip_coef=yaml_config["clip_coef"],
        ent_coef=yaml_config["ent_coef"],
        value_coef=yaml_config["value_coef"],
        learning_rate=yaml_config["learning_rate"],
        max_grad_norm=yaml_config["max_grad_norm"],
        target_kl=yaml_config["target_kl"],
        seed=yaml_config["seed"],
        device=device,
        torch_deterministic=yaml_config["torch_deterministic"],
        track_eval=not yaml_config["no_eval"],
        eval_episodes=yaml_config["eval_episodes"],
        eval_interval=yaml_config["eval_interval"],
        save_interval=yaml_config["save_interval"],
        video_interval_minutes=None if yaml_config["no_video"] else yaml_config["video_interval_minutes"],
        max_video_steps=yaml_config["max_video_steps"],
        max_offroad_seconds=yaml_config["max_offroad_seconds"],
        offroad_penalty=yaml_config["offroad_penalty"],
        continuous=not yaml_config["discrete"],
        reward_shaping=yaml_config["reward_shaping"],
    )
    
    # Create trainer
    trainer = PPOTrainer(config)
    
    # Resume from checkpoint if specified
    if yaml_config.get("resume"):
        checkpoint_path = Path(yaml_config["resume"])
        print(f"Resuming from checkpoint: {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path)
    
    # Train
    print("\n" + "="*80)
    print("Starting training with configuration:")
    print("="*80)
    print(f"  Total timesteps: {config.total_timesteps:,}")
    print(f"  Num environments: {config.num_envs}")
    print(f"  Action space: {'Discrete (5 actions)' if not config.continuous else 'Continuous (Box(3,))'}")
    print(f"  Reward shaping: {config.reward_shaping}")
    print(f"  Device: {config.device}")
    print(f"  Seed: {config.seed}")
    print("="*80 + "\n")
    
    trainer.train()


if __name__ == "__main__":
    main()
