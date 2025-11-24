from __future__ import annotations

import argparse
import sys
from pathlib import Path
import warnings

import torch

warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from latent.pca_ppo import PCAPPOConfig, PCAPPOTrainer  # noqa: E402
from latent.greyscale import load_greyscale_preset  # noqa: E402
from latent.paths import GREYSCALE_PRESETS_PATH  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO agent on PCA-transformed CarRacing observations.")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--torch-deterministic", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--video-interval-minutes", type=float, default=None)
    parser.add_argument("--max-video-steps", type=int, default=1000)
    parser.add_argument("--max-offroad-seconds", type=float, default=2.0)
    parser.add_argument("--offroad-penalty", type=float, default=None, nargs="?")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint (.pt) path to resume from")
    parser.add_argument("--pca-model-path", type=Path, default=Path("scripts/latent_space_experiment/models/pca/dim_012/pca_model.pkl"))
    parser.add_argument("--num-stack", type=int, default=4)
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Number of environment frames to skip between stacked PCA observations",
    )
    parser.add_argument("--crop-ratio", type=float, default=0.13)
    parser.add_argument("--resize-height", type=int, default=48)
    parser.add_argument("--resize-width", type=int, default=48)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("scripts/latent_space_experiment/tensorboard_logs/ppo_latent"),
        help="Directory (relative to repo root) where TensorBoard logs for the latent PPO run are stored.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("scripts/latent_space_experiment/models/pca_ppo_runs"),
        help="Directory for saving PCA PPO checkpoints.",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path("scripts/latent_space_experiment/videos/ppo_latent"),
        help="Directory for saving PCA PPO policy videos.",
    )
    parser.add_argument(
        "--discrete",
        action="store_true",
        help="Use the discrete (5-action) CarRacing action space instead of the continuous steering/brake/gas.",
    )
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument(
        "--greyscale-presets-path",
        type=Path,
        default=GREYSCALE_PRESETS_PATH,
        help="JSONL file with greyscale presets for preprocessing.",
    )
    parser.add_argument(
        "--greyscale-label",
        type=str,
        default="veryheavy-medium",
        help="Preset label to use for preprocessing (empty string to disable).",
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


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    greyscale_preset = None
    if args.greyscale_label:
        preset_path = Path(args.greyscale_presets_path).expanduser().resolve()
        greyscale_preset = load_greyscale_preset(preset_path, args.greyscale_label)
        args.crop_ratio = greyscale_preset.crop_ratio
        args.resize_height = greyscale_preset.output_height
        args.resize_width = greyscale_preset.output_width
        print(
            f"Using greyscale preset '{args.greyscale_label}' "
            f"({args.resize_height}x{args.resize_width}) for PCA PPO training."
        )

    config = PCAPPOConfig(
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
        video_interval_minutes=args.video_interval_minutes,
        max_video_steps=args.max_video_steps,
        max_offroad_seconds=args.max_offroad_seconds,
        offroad_penalty=args.offroad_penalty,
        pca_model_path=args.pca_model_path,
        num_stack=args.num_stack,
        frame_skip=args.frame_skip,
        crop_ratio=args.crop_ratio,
        resize_height=args.resize_height,
        resize_width=args.resize_width,
        ridge_lambda=args.ridge_lambda,
        log_root=args.log_root,
        checkpoint_root=args.checkpoint_root,
        video_root=args.video_root,
        continuous=not args.discrete,
        greyscale_presets_path=args.greyscale_presets_path if greyscale_preset else None,
        greyscale_label=args.greyscale_label if greyscale_preset else None,
    )

    trainer = PCAPPOTrainer(config)
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))
    trainer.train()


if __name__ == "__main__":
    main()
