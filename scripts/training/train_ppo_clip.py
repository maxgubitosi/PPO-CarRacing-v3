from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from ppo_clip import PPOConfig, PPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO-Clip on CarRacing-v3")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=512)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--update-epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--torch-deterministic", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--video-interval-minutes", type=float, default=10.0)
    parser.add_argument("--max-video-steps", type=int, default=1000)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--max-offroad-seconds", type=float, default=2.0)
    parser.add_argument("--offroad-penalty", type=float, default=None, nargs="?")
    parser.add_argument("--resume" , type=str, default=None, help="Checkpoint (.pt) path to resume from")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    config = PPOConfig(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        value_coef=args.value_coef,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        seed=args.seed,
        device=device,
        torch_deterministic=args.torch_deterministic,
        track_eval=not args.no_eval,
        eval_episodes=args.eval_episodes,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        video_interval_minutes=None if args.no_video else args.video_interval_minutes,
        max_video_steps=args.max_video_steps,
        max_offroad_seconds=args.max_offroad_seconds,
        offroad_penalty=args.offroad_penalty,
    )

    trainer = PPOTrainer(config)
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))
    trainer.train()


if __name__ == "__main__":
    main()

