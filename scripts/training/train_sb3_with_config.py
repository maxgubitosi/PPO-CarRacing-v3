"""Train SB3 PPO using a YAML configuration (for comparison with custom PPO implementation)."""
from __future__ import annotations

import sys
import time
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import yaml

warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from torch.utils.tensorboard import SummaryWriter

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from environment.carracing import (
    CarRacingPreprocess,
    FrameStackWrapper,
    OffRoadPenaltyWrapper,
    SteeringConstraintWrapper,
)
from utils import resolve_device, set_seed


ACTION_NAMES = ["Do Nothing", "Right", "Left", "Gas", "Brake"]


class TrainingMetricsCallback(BaseCallback):
    def __init__(
        self,
        writer: SummaryWriter,
        steering_constraint: str | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.writer = writer
        self.steering_constraint = steering_constraint
        self._update_counter = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_info = info["episode"]
                self.writer.add_scalar("rollout/episode_return", ep_info["r"], self.num_timesteps)
                self.writer.add_scalar("rollout/episode_length", ep_info["l"], self.num_timesteps)
        return True

    def _on_rollout_end(self) -> None:
        self._update_counter += 1
        self._log_losses()
        self._log_charts()
        self._log_action_distribution()

    def _log_losses(self) -> None:
        logger = self.model.logger
        if logger is None:
            return
        name_map = logger.name_to_value
        if "train/loss" in name_map:
            self.writer.add_scalar("losses/loss", name_map["train/loss"], self.num_timesteps)
        if "train/policy_gradient_loss" in name_map:
            self.writer.add_scalar("losses/policy_loss", name_map["train/policy_gradient_loss"], self.num_timesteps)
        if "train/value_loss" in name_map:
            self.writer.add_scalar("losses/value_loss", name_map["train/value_loss"], self.num_timesteps)
        if "train/entropy_loss" in name_map:
            self.writer.add_scalar("losses/entropy", -name_map["train/entropy_loss"], self.num_timesteps)
        if "train/approx_kl" in name_map:
            self.writer.add_scalar("losses/approx_kl", name_map["train/approx_kl"], self.num_timesteps)

    def _log_charts(self) -> None:
        lr = self.model.lr_schedule(self.model._current_progress_remaining)
        self.writer.add_scalar("charts/learning_rate", lr, self.num_timesteps)
        self.writer.add_scalar("charts/update", self._update_counter, self.num_timesteps)

    def _log_action_distribution(self) -> None:
        rollout_buffer = self.model.rollout_buffer
        actions = rollout_buffer.actions.flatten().astype(int)
        action_space = self.model.action_space
        
        if hasattr(action_space, "n"):
            num_actions = action_space.n
            action_counts = np.bincount(actions, minlength=num_actions)
            total = action_counts.sum() or 1
            action_freq = action_counts / total
            
            names = self._resolve_action_names(num_actions)
            for name, freq in zip(names, action_freq):
                self.writer.add_scalar(f"actions/discrete/{name}", freq, self.num_timesteps)
        else:
            actions_2d = rollout_buffer.actions.reshape(-1, rollout_buffer.actions.shape[-1])
            steering = actions_2d[:, 0]
            gas = actions_2d[:, 1]
            brake = actions_2d[:, 2]
            self.writer.add_scalar("actions/continuous/steering_mean", float(np.mean(steering)), self.num_timesteps)
            self.writer.add_scalar("actions/continuous/steering_std", float(np.std(steering)), self.num_timesteps)
            self.writer.add_scalar("actions/continuous/gas_mean", float(np.mean(gas)), self.num_timesteps)
            self.writer.add_scalar("actions/continuous/gas_std", float(np.std(gas)), self.num_timesteps)
            self.writer.add_scalar("actions/continuous/brake_mean", float(np.mean(brake)), self.num_timesteps)
            self.writer.add_scalar("actions/continuous/brake_std", float(np.std(brake)), self.num_timesteps)
            self.writer.add_histogram("actions/continuous/steering_hist", steering, self.num_timesteps)
            self.writer.add_histogram("actions/continuous/gas_hist", gas, self.num_timesteps)
            self.writer.add_histogram("actions/continuous/brake_hist", brake, self.num_timesteps)

    def _resolve_action_names(self, num_actions: int) -> list[str]:
        constraint = (self.steering_constraint or "").strip().lower()
        if constraint == "only_left":
            removed = 1
        elif constraint == "only_right":
            removed = 2
        else:
            removed = None
        if removed is not None:
            filtered = [name for idx, name in enumerate(ACTION_NAMES) if idx != removed]
        else:
            filtered = ACTION_NAMES
        return filtered[:num_actions]


class EvalVideoCallback(BaseCallback):
    def __init__(
        self,
        writer: SummaryWriter,
        eval_env_fn,
        eval_episodes: int = 5,
        eval_interval: int = 50,
        video_interval_minutes: float | None = 10.0,
        max_video_steps: int = 1000,
        video_dir: Path | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.writer = writer
        self.eval_env_fn = eval_env_fn
        self.eval_episodes = eval_episodes
        self.eval_interval = eval_interval
        self.video_interval_seconds = video_interval_minutes * 60 if video_interval_minutes else None
        self.max_video_steps = max_video_steps
        self.video_dir = video_dir
        self._last_video_time = time.perf_counter()
        self._update_counter = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self._update_counter += 1

        if self._update_counter % self.eval_interval == 0:
            self._evaluate()

        if self.video_interval_seconds is not None:
            now = time.perf_counter()
            if now - self._last_video_time >= self.video_interval_seconds:
                self._record_video()
                self._last_video_time = now

    def _evaluate(self) -> None:
        eval_env = self.eval_env_fn()
        returns = []
        lengths = []
        deaths = []

        for _ in range(self.eval_episodes):
            obs, _ = eval_env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            died = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
                if reward <= -99.0:
                    died = True

            returns.append(total_reward)
            lengths.append(steps)
            deaths.append(1 if died else 0)

        eval_env.close()

        self.writer.add_scalar("eval/return_mean", float(np.mean(returns)), self.num_timesteps)
        self.writer.add_scalar("eval/return_std", float(np.std(returns)), self.num_timesteps)
        self.writer.add_scalar("eval/return_max", float(np.max(returns)), self.num_timesteps)
        self.writer.add_scalar("eval/return_min", float(np.min(returns)), self.num_timesteps)
        self.writer.add_scalar("eval/episode_length", float(np.mean(lengths)), self.num_timesteps)
        self.writer.add_scalar("eval/death_rate", float(np.mean(deaths)), self.num_timesteps)

    def _record_video(self) -> None:
        if self.video_dir is None:
            return

        env = self.eval_env_fn(render_mode="rgb_array")
        frames = []
        obs, _ = env.reset()
        frame = env.render()
        if frame is not None:
            frames.append(self._prepare_frame(frame))

        done = False
        steps = 0

        while not done and steps < self.max_video_steps:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            frame = env.render()
            if frame is not None:
                frames.append(self._prepare_frame(frame))
            steps += 1

        env.close()

        if not frames:
            return

        video_array = np.stack(frames)
        video_path = self.video_dir / f"policy_step_{self.num_timesteps}.gif"
        imageio.mimsave(video_path, video_array, duration=1 / 30)

        video_tensor = torch.from_numpy(video_array).permute(0, 3, 1, 2).unsqueeze(0)
        self.writer.add_video("policy/sample", video_tensor, self.num_timesteps, fps=30)

    @staticmethod
    def _prepare_frame(frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame * 255.0 if frame.max() <= 1.0 else frame, 0, 255)
            frame = frame.astype(np.uint8)
        return frame


class CheckpointCallback(BaseCallback):
    def __init__(self, save_interval: int, checkpoint_dir: Path, verbose: int = 0):
        super().__init__(verbose)
        self.save_interval = save_interval
        self.checkpoint_dir = checkpoint_dir
        self._update_counter = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self._update_counter += 1
        if self._update_counter % self.save_interval == 0:
            path = self.checkpoint_dir / f"sb3_ppo_update_{self._update_counter}.zip"
            self.model.save(path)


def make_env(
    env_id: str,
    seed: int,
    render_mode: str | None = None,
    offroad_penalty: float | None = None,
    max_offroad_seconds: float = 2.0,
    continuous: bool = True,
    frame_skip: int = 0,
    num_stack: int = 4,
    steering_constraint: str | None = None,
):
    def thunk():
        env = gym.make(
            env_id,
            render_mode=render_mode,
            continuous=continuous,
            lap_complete_percent=0.95,
            domain_randomize=False,
        )
        env = CarRacingPreprocess(env)
        env = FrameStackWrapper(env, num_stack=num_stack, frame_skip=frame_skip)
        env = OffRoadPenaltyWrapper(env, max_offroad_seconds=max_offroad_seconds, penalty=offroad_penalty)
        if steering_constraint:
            env = SteeringConstraintWrapper(env, steering_constraint)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env
    return thunk


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SB3 PPO on CarRacing-v3 with YAML config")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (e.g., configs/ppo_config.yaml)",
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

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    cfg = load_config(config_path)
    print(f"Loaded configuration from: {config_path}")
    print(f"Config: {cfg}")

    if args.total_timesteps is not None:
        cfg["total_timesteps"] = args.total_timesteps
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device is not None:
        cfg["device"] = args.device

    if isinstance(cfg.get("learning_rate"), str):
        cfg["learning_rate"] = float(cfg["learning_rate"])

    device = resolve_device(cfg.get("device", "auto"))
    set_seed(cfg["seed"], deterministic=cfg.get("torch_deterministic", False))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"sb3_ppo_clip_{timestamp}"

    log_dir = ROOT_DIR / "results" / "tensorboard_logs" / "sb3_ppo_clip" / run_name
    checkpoint_dir = ROOT_DIR / "results" / "models" / "sb3_ppo_clip" / run_name
    video_dir = ROOT_DIR / "results" / "videos" / "sb3_ppo_clip" / run_name

    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    no_video = cfg.get("no_video", False)
    if not no_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    continuous = not cfg.get("discrete", False)
    steering_constraint = cfg.get("steering_constraint")

    env_kwargs = dict(
        env_id="CarRacing-v3",
        offroad_penalty=cfg.get("offroad_penalty"),
        max_offroad_seconds=cfg.get("max_offroad_seconds", 2.0),
        continuous=continuous,
        frame_skip=cfg.get("frame_skip", 0),
        num_stack=cfg.get("num_stack", 4),
        steering_constraint=steering_constraint,
    )

    num_envs = cfg["num_envs"]
    env_fns = [make_env(seed=cfg["seed"] + i, **env_kwargs) for i in range(num_envs)]
    if num_envs == 1:
        env = DummyVecEnv(env_fns)
    else:
        env = SubprocVecEnv(env_fns)
    env = VecMonitor(env)

    num_steps = cfg["num_steps"]
    batch_size = num_envs * num_steps
    minibatch_size = batch_size // cfg["num_minibatches"]

    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=cfg["learning_rate"],
        n_steps=num_steps,
        batch_size=minibatch_size,
        n_epochs=cfg["update_epochs"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_coef"],
        ent_coef=cfg["ent_coef"],
        vf_coef=cfg["value_coef"],
        max_grad_norm=cfg["max_grad_norm"],
        target_kl=cfg.get("target_kl"),
        seed=cfg["seed"],
        device=device,
        verbose=0,
        policy_kwargs={"normalize_images": False},
    )

    writer = SummaryWriter(log_dir=str(log_dir))
    config_str = "\n".join(f"{k}: {v}" for k, v in cfg.items())
    writer.add_text("config", config_str)

    def eval_env_fn(render_mode: str | None = None):
        return make_env(seed=cfg["seed"] + 1000, render_mode=render_mode, **env_kwargs)()

    callbacks = [
        TrainingMetricsCallback(writer, steering_constraint=steering_constraint),
    ]

    no_eval = cfg.get("no_eval", False)
    if not no_eval:
        callbacks.append(
            EvalVideoCallback(
                writer=writer,
                eval_env_fn=eval_env_fn,
                eval_episodes=cfg.get("eval_episodes", 5),
                eval_interval=cfg.get("eval_interval", 50),
                video_interval_minutes=None if no_video else cfg.get("video_interval_minutes", 10.0),
                max_video_steps=cfg.get("max_video_steps", 1000),
                video_dir=video_dir if not no_video else None,
            )
        )

    callbacks.append(CheckpointCallback(cfg.get("save_interval", 50), checkpoint_dir))

    action_space = env.action_space
    if hasattr(action_space, "n"):
        action_desc = f"Discrete ({action_space.n} actions)"
    elif hasattr(action_space, "shape"):
        shape = tuple(int(dim) for dim in action_space.shape)
        action_desc = f"Box{shape}"
    else:
        action_desc = str(action_space)

    print("\n" + "=" * 80)
    print("Starting SB3 PPO training with configuration:")
    print("=" * 80)
    print(f"  Total timesteps: {cfg['total_timesteps']:,}")
    print(f"  Num environments: {num_envs}")
    print(f"  Action space: {action_desc}")
    print(f"  Reward shaping: {cfg.get('reward_shaping', False)}")
    print(f"  Device: {device}")
    print(f"  Seed: {cfg['seed']}")
    print(f"  Log dir: {log_dir}")
    print("=" * 80 + "\n")

    model.learn(
        total_timesteps=cfg["total_timesteps"],
        callback=callbacks,
        progress_bar=True,
    )

    model.save(checkpoint_dir / "sb3_ppo_final.zip")
    env.close()
    writer.close()


if __name__ == "__main__":
    main()
