#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Iterable

import torch

warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ppo_clip import PPOConfig, PPOTrainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile PPO training speed for multiple devices and env counts."
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["cpu", "mps", "cuda"],
        help="Ordered list of devices to test. Devices that are unavailable are skipped.",
    )
    parser.add_argument(
        "--env-grid",
        "--num-envs",
        dest="env_grid",
        type=int,
        nargs="+",
        default=[4, 8, 16, 24],
        help="Parallel environment counts to sweep for each device.",
    )
    parser.add_argument("--total-timesteps", type=int, default=17_000, help="Timesteps per run.")
    parser.add_argument("--num-steps", type=int, default=512, help="Steps collected per rollout.")
    parser.add_argument("--num-stack", type=int, default=4, help="Frames stacked per observation.")
    parser.add_argument("--frame-skip", type=int, default=0, help="Frames skipped between stacks.")
    parser.add_argument("--num-minibatches", type=int, default=8, help="PPO minibatches per update.")
    parser.add_argument("--update-epochs", type=int, default=8, help="Gradient epochs per update.")
    parser.add_argument("--seed", type=int, default=42, help="Training seed.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate to reuse for each run.",
    )
    parser.add_argument(
        "--discrete",
        action="store_true",
        help="Use discrete action space (matches --discrete flag from train_ppo_clip).",
    )
    return parser.parse_args()


def device_is_available(name: str) -> bool:
    name = name.lower()
    if name == "cpu":
        return True
    if name == "cuda":
        return torch.cuda.is_available()
    if name == "mps":
        return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    return False


def format_seconds(value: float) -> str:
    return f"{value:.2f}s"


def summarize_profile(entries: list[dict[str, float]]) -> dict[str, float]:
    if not entries:
        return {
            "rollout_wall": 0.0,
            "rollout_cpu": 0.0,
            "update_wall": 0.0,
            "update_cpu": 0.0,
            "updates": 0,
        }

    def total(key: str) -> float:
        return sum(entry.get(key, 0.0) for entry in entries)

    summary = {
        "rollout_wall": total("rollout_wall_s"),
        "rollout_cpu": total("rollout_cpu_s"),
        "update_wall": total("update_wall_s"),
        "update_cpu": total("update_cpu_s"),
        "updates": len(entries),
        "per_update_rollout_wall": total("rollout_wall_s") / len(entries),
        "per_update_update_wall": total("update_wall_s") / len(entries),
    }
    summary["total_profiled_wall"] = summary["rollout_wall"] + summary["update_wall"]
    summary["total_profiled_cpu"] = summary["rollout_cpu"] + summary["update_cpu"]
    return summary


def build_config(args: argparse.Namespace, device: str, num_envs: int) -> PPOConfig:
    profile_root = Path("results/device_profiles") / device
    base_config = PPOConfig(
        total_timesteps=args.total_timesteps,
        num_envs=num_envs,
        num_steps=args.num_steps,
        num_stack=args.num_stack,
        frame_skip=args.frame_skip,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=device,
        track_eval=False,
        video_interval_minutes=None,
        save_interval=10**9,
        eval_interval=10**9,
        continuous=not args.discrete,
        collect_timing_metrics=True,
        write_artifacts=False,
        log_root=profile_root / "tensorboard",
        checkpoint_root=profile_root / "checkpoints",
        video_root=profile_root / "videos",
    )
    return base_config


def profile_device(args: argparse.Namespace, device: str, num_envs: int) -> dict[str, float | int | str]:
    print(f"\n=== Profiling device: {device} | envs: {num_envs} ===")
    config = build_config(args, device, num_envs)
    trainer = PPOTrainer(config)

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    trainer.train()
    wall_total = time.perf_counter() - wall_start
    cpu_total = time.process_time() - cpu_start
    actual_steps = getattr(trainer, "completed_steps", args.total_timesteps)

    profile_stats = summarize_profile(trainer.profile_history)
    profiled_wall = profile_stats.get("total_profiled_wall", 0.0) or 1e-9
    rollout_share = profile_stats.get("rollout_wall", 0.0) / profiled_wall
    update_share = profile_stats.get("update_wall", 0.0) / profiled_wall

    steps_per_second = (actual_steps / wall_total) if wall_total else 0.0
    wall_per_100k = wall_total * (100_000 / actual_steps) if actual_steps else 0.0

    result = {
        "device": device,
        "num_envs": num_envs,
        "wall_time": wall_total,
        "cpu_time": cpu_total,
        "steps": actual_steps,
        "steps_per_second": steps_per_second,
        "rollout_wall": profile_stats.get("rollout_wall", 0.0),
        "rollout_cpu": profile_stats.get("rollout_cpu", 0.0),
        "update_wall": profile_stats.get("update_wall", 0.0),
        "update_cpu": profile_stats.get("update_cpu", 0.0),
        "rollout_share": rollout_share,
        "update_share": update_share,
        "updates": profile_stats.get("updates", 0),
        "wall_100k": wall_per_100k,
    }

    print(f"Completed in {format_seconds(wall_total)} ({steps_per_second:.1f} steps/s)")
    print(
        f"  Exploration rollout: {format_seconds(result['rollout_wall'])} "
        f"({rollout_share * 100:.1f}% wall, CPU {format_seconds(result['rollout_cpu'])})"
    )
    print(
        f"  Model updates     : {format_seconds(result['update_wall'])} "
        f"({update_share * 100:.1f}% wall, CPU {format_seconds(result['update_cpu'])})"
    )
    return result


def main() -> None:
    args = parse_args()
    requested_devices: Iterable[str] = (device.lower() for device in args.devices)
    devices = [device for device in requested_devices if device_is_available(device)]

    if not devices:
        print("No requested devices are available on this machine.")
        sys.exit(1)

    env_counts = sorted({env for env in args.env_grid if env > 0})
    if not env_counts:
        print("No valid environment counts were provided.")
        sys.exit(1)

    skipped = [device for device in set(args.devices) if device.lower() not in devices]
    for device in skipped:
        if not device_is_available(device.lower()):
            print(f"Skipping {device}: device not available.")

    results = []
    for device in devices:
        for num_envs in env_counts:
            try:
                results.append(profile_device(args, device, num_envs))
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # pragma: no cover - just diagnostic output
                print(f"Profiling failed on {device} (envs={num_envs}): {exc}")

    if not results:
        print("No profiling runs completed.")
        return

    print("\n=== Aggregated comparison ===")
    slowest_wall_norm = max(result["wall_100k"] for result in results)
    header = (
        f"{'Device':<6} {'Envs':>5} {'Wall(s)':>10} {'Wall(% better)':>15} {'Wall_100k(s)':>14} "
        f"{'Steps/s':>10} {'Rollout%':>10} {'Update%':>10}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        rel_pct = (result["wall_100k"] / slowest_wall_norm * 100) if slowest_wall_norm else 0.0
        print(
            f"{result['device']:<6} "
            f"{result['num_envs']:>5d} "
            f"{result['wall_time']:>10.2f} "
            f"{rel_pct:>14.1f}% "
            f"{result['wall_100k']:>14.2f} "
            f"{result['steps_per_second']:>10.1f} "
            f"{result['rollout_share'] * 100:>9.1f}% "
            f"{result['update_share'] * 100:>9.1f}%"
        )


if __name__ == "__main__":
    main()
