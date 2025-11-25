from __future__ import annotations

import argparse
import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import yaml

from latent.pca_ppo import PCAPPOConfig, PCAPPOTrainer  # noqa: E402
from latent.greyscale import load_greyscale_preset  # noqa: E402
from latent.paths import GREYSCALE_PRESETS_PATH  # noqa: E402
from utils import resolve_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PCA PPO agent.")
    parser.add_argument("--config", type=str, required=True, help="YAML config path (e.g., configs/pca_ppo_config.yaml)")
    parser.add_argument("--resume", type=str, default=None, help="Override checkpoint path to resume from")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "mps"], default=None)
    return parser.parse_args()


def load_yaml_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    yaml_cfg = load_yaml_config(config_path)
    if args.resume is not None:
        yaml_cfg["resume"] = args.resume
    if args.device is not None:
        yaml_cfg["device"] = args.device

    device = resolve_device(yaml_cfg.get("device", "auto"))

    greyscale_preset = None
    greyscale_label = yaml_cfg.get("greyscale_label")
    greyscale_path = yaml_cfg.get("greyscale_presets_path")
    if greyscale_label:
        preset_path = Path(greyscale_path or GREYSCALE_PRESETS_PATH).expanduser().resolve()
        greyscale_preset = load_greyscale_preset(preset_path, greyscale_label)
        yaml_cfg["crop_ratio"] = greyscale_preset.crop_ratio
        yaml_cfg["resize_height"] = greyscale_preset.output_height
        yaml_cfg["resize_width"] = greyscale_preset.output_width
        yaml_cfg["greyscale_presets_path"] = greyscale_path or GREYSCALE_PRESETS_PATH

    resume_path = yaml_cfg.get("resume")

    config = PCAPPOConfig(
        total_timesteps=yaml_cfg["total_timesteps"],
        num_envs=yaml_cfg["num_envs"],
        num_steps=yaml_cfg["num_steps"],
        num_minibatches=yaml_cfg["num_minibatches"],
        update_epochs=yaml_cfg["update_epochs"],
        gamma=yaml_cfg["gamma"],
        gae_lambda=yaml_cfg["gae_lambda"],
        clip_coef=yaml_cfg["clip_coef"],
        ent_coef=yaml_cfg["ent_coef"],
        value_coef=yaml_cfg["value_coef"],
        learning_rate=yaml_cfg["learning_rate"],
        max_grad_norm=yaml_cfg["max_grad_norm"],
        target_kl=yaml_cfg["target_kl"],
        seed=yaml_cfg["seed"],
        device=device,
        torch_deterministic=yaml_cfg["torch_deterministic"],
        track_eval=not yaml_cfg.get("no_eval", False),
        eval_episodes=yaml_cfg["eval_episodes"],
        eval_interval=yaml_cfg["eval_interval"],
        save_interval=yaml_cfg["save_interval"],
        video_interval_minutes=yaml_cfg["video_interval_minutes"],
        max_video_steps=yaml_cfg["max_video_steps"],
        max_offroad_seconds=yaml_cfg["max_offroad_seconds"],
        offroad_penalty=yaml_cfg["offroad_penalty"],
        pca_model_path=yaml_cfg["pca_model_path"],
        num_stack=yaml_cfg["num_stack"],
        frame_skip=yaml_cfg["frame_skip"],
        crop_ratio=yaml_cfg["crop_ratio"],
        resize_height=yaml_cfg["resize_height"],
        resize_width=yaml_cfg["resize_width"],
        ridge_lambda=yaml_cfg["ridge_lambda"],
        log_root=Path(yaml_cfg["log_root"]),
        checkpoint_root=Path(yaml_cfg["checkpoint_root"]),
        video_root=Path(yaml_cfg["video_root"]),
        continuous=not yaml_cfg.get("discrete", False),
        greyscale_presets_path=yaml_cfg.get("greyscale_presets_path"),
        greyscale_label=greyscale_label if greyscale_preset else None,
        use_lr_scheduler=yaml_cfg.get("use_lr_scheduler", False),
        lr_end=yaml_cfg.get("lr_end", 1e-6),
        reward_shaping=yaml_cfg.get("reward_shaping", False),
        verbose=yaml_cfg.get("verbose", False),
        latent_hidden_dim=yaml_cfg.get("latent_hidden_dim"),
        compare_reconstruction=yaml_cfg.get("compare_reconstruction", True),
    )

    trainer = PCAPPOTrainer(config)
    if resume_path:
        trainer.load_checkpoint(Path(resume_path))
    trainer.train()


if __name__ == "__main__":
    main()
