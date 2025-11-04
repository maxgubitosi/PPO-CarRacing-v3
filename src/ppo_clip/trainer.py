from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict

import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

try:
    from torch.serialization import add_safe_globals
except ImportError:  # pragma: no cover - compatibility for older torch
    add_safe_globals = None

from environment import create_single_env, create_vector_env
from utils import set_seed

from .agent import PPOClipAgent
from .config import PPOConfig
from .rollout_buffer import RolloutBuffer


class PPOTrainer:
    def __init__(self, config: PPOConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.project_root = Path(__file__).resolve().parents[2]

        set_seed(config.seed, deterministic=config.torch_deterministic)

        self.env = create_vector_env(
            config.env_id,
            config.num_envs,
            config.seed,
            offroad_penalty=config.offroad_penalty,
            max_offroad_seconds=config.max_offroad_seconds,
            action_wrapper_name=config.action_wrapper,
        )
        self.eval_env_seeds = [config.seed + 10 + i for i in range(config.eval_episodes)]
        self.eval_env_index = 0
        self.eval_env = (
            create_single_env(
                config.env_id,
                self.eval_env_seeds[self.eval_env_index],
                render_mode=None,
                offroad_penalty=config.offroad_penalty,
                max_offroad_seconds=config.max_offroad_seconds,
                action_wrapper_name=config.action_wrapper,
            )
            if config.track_eval
            else None
        )

        obs_space = self.env.single_observation_space
        action_space = self.env.single_action_space

        self.agent = PPOClipAgent(obs_space, action_space, config)

        # Detectar si el espacio de acción es discreto
        from gymnasium.spaces import Discrete
        is_discrete = isinstance(action_space, Discrete)
        action_dim = action_space.n if is_discrete else int(np.prod(action_space.shape))

        self.buffer = RolloutBuffer(
            num_steps=config.num_steps,
            num_envs=config.num_envs,
            obs_shape=obs_space.shape,
            action_dim=action_dim,
            device=self.device,
            is_discrete=is_discrete,
        )

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_name = f"ppo_clip_{timestamp}"
        self.log_dir = (self.project_root / config.log_root / self.run_name).resolve()
        self.checkpoint_dir = (self.project_root / config.checkpoint_root / self.run_name).resolve()
        self.video_dir = (self.project_root / config.video_root / self.run_name).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if config.video_interval_minutes is not None:
            self.video_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.writer.add_text("config", str(asdict(config)))

        self.video_interval_seconds = (
            config.video_interval_minutes * 60 if config.video_interval_minutes is not None else None
        )
        self._last_video_time = time.perf_counter()
        self._video_seed_offset = 1000
        self._resume_step = 0
        self._resume_update = 0

    def train(self) -> None:
        obs, _ = self.env.reset(seed=self.config.seed)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        next_done = torch.zeros(self.config.num_envs, device=self.device)

        global_step = self._resume_step

        for update in range(self._resume_update + 1, self.config.num_updates + 1):
            for step in range(self.config.num_steps):
                with torch.no_grad():
                    sample = self.agent.sample(obs)

                actions = sample["action"].cpu().numpy()
                next_obs, rewards, terminated, truncated, infos = self.env.step(actions)

                dones = np.logical_or(terminated, truncated)

                reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
                done_tensor = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

                self.buffer.add(
                    obs,
                    sample["raw_action"].detach(),
                    sample["log_prob"].detach(),
                    reward_tensor,
                    done_tensor,
                    sample["value"].detach(),
                )

                obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
                next_done = done_tensor
                global_step += self.config.num_envs

                self._log_rollout_infos(infos, global_step)

            with torch.no_grad():
                _, last_values = self.agent.network.get_dist_and_value(obs)

            self.buffer.compute_returns_and_advantages(
                last_values=last_values.detach(),
                last_dones=next_done,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
            )

            metrics = defaultdict(list)
            for epoch in range(self.config.update_epochs):
                for batch in self.buffer.get(self.config.minibatch_size):
                    stats = self.agent.update(batch)
                    metrics["loss"].append(stats.loss.item())
                    metrics["policy_loss"].append(stats.policy_loss.item())
                    metrics["value_loss"].append(stats.value_loss.item())
                    metrics["entropy"].append(stats.entropy.item())
                    metrics["approx_kl"].append(stats.approx_kl.item())

                if (
                    self.config.target_kl
                    and metrics["approx_kl"]
                    and metrics["approx_kl"][-1] > self.config.target_kl
                ):
                    break

            self.buffer.reset()

            self._log_update_metrics(metrics, global_step, update)

            if self.config.track_eval and update % self.config.eval_interval == 0:
                self._evaluate(global_step)

            if update % self.config.save_interval == 0:
                self._save_checkpoint(update, global_step)

            if self.video_interval_seconds is not None:
                now = time.perf_counter()
                if now - self._last_video_time >= self.video_interval_seconds:
                    self._record_video(global_step)
                    self._last_video_time = now

        self.env.close()
        if self.eval_env:
            self.eval_env.close()
        self.writer.close()

    def _log_rollout_infos(self, infos: Dict[str, np.ndarray], global_step: int) -> None:
        if "episode" not in infos:
            return

        for episode_info in infos["episode"]:
            if episode_info is None:
                continue
            self.writer.add_scalar("rollout/episode_return", episode_info["r"], global_step)
            self.writer.add_scalar("rollout/episode_length", episode_info["l"], global_step)

    def _log_update_metrics(self, metrics: Dict[str, list[float]], global_step: int, update: int) -> None:
        for key, values in metrics.items():
            if not values:
                continue
            mean_value = float(np.mean(values))
            self.writer.add_scalar(f"losses/{key}", mean_value, global_step)

        self.writer.add_scalar("charts/learning_rate", self.agent.optimizer.param_groups[0]["lr"], global_step)
        self.writer.add_scalar("charts/update", update, global_step)

    def _evaluate(self, global_step: int) -> None:
        if self.eval_env is None:
            return

        returns = []
        lengths = []

        for idx in range(self.config.eval_episodes):
            seed = self.eval_env_seeds[self.eval_env_index]
            self.eval_env_index = (self.eval_env_index + 1) % len(self.eval_env_seeds)

            obs, _ = self.eval_env.reset(seed=seed)
            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action, _ = self.agent.act_deterministic(obs_tensor)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action.squeeze(0).cpu().numpy())
                done = terminated or truncated
                total_reward += reward
                steps += 1

            returns.append(total_reward)
            lengths.append(steps)

        self.writer.add_scalar("eval/return_mean", float(np.mean(returns)), global_step)
        self.writer.add_scalar("eval/return_std", float(np.std(returns)), global_step)
        self.writer.add_scalar("eval/episode_length", float(np.mean(lengths)), global_step)

    def _save_checkpoint(self, update: int, global_step: int) -> None:
        checkpoint_path = self.checkpoint_dir / f"ppo_clip_update_{update}.pt"
        torch.save(
            {
                "agent": self.agent.state_dict(),
                "config": asdict(self.config),
                "global_step": global_step,
                "update": update,
            },
            checkpoint_path,
        )

    def _record_video(self, global_step: int) -> None:
        env_seed = self.eval_env_seeds[self.eval_env_index]
        self.eval_env_index = (self.eval_env_index + 1) % len(self.eval_env_seeds)
        env = create_single_env(
            self.config.env_id,
            env_seed,
            render_mode="rgb_array",
            offroad_penalty=self.config.offroad_penalty,
            max_offroad_seconds=self.config.max_offroad_seconds,
            action_wrapper_name=self.config.action_wrapper,
        )

        frames = []
        obs, _ = env.reset()
        frame = env.render()
        if frame is not None:
            frames.append(self._prepare_frame(frame))

        done = False
        steps = 0

        while not done and steps < self.config.max_video_steps:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action, _ = self.agent.act_deterministic(obs_tensor)
            obs, _, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
            done = terminated or truncated
            frame = env.render()
            if frame is not None:
                frames.append(self._prepare_frame(frame))
            steps += 1

        env.close()

        if not frames:
            return

        video_array = np.stack(frames)
        video_path = self.video_dir / f"policy_step_{global_step}.gif"
        imageio.mimsave(video_path, video_array, duration=1 / 30)

        video_tensor = torch.from_numpy(video_array).permute(0, 3, 1, 2).unsqueeze(0)
        self.writer.add_video("policy/sample", video_tensor, global_step, fps=30)

    @staticmethod
    def _prepare_frame(frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame * 255.0 if frame.max() <= 1.0 else frame, 0, 255)
            frame = frame.astype(np.uint8)
        return frame

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        if add_safe_globals is not None:
            add_safe_globals([PPOConfig])

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if "agent" in checkpoint:
            agent_state = checkpoint["agent"]
            self._resume_step = int(checkpoint.get("global_step", 0))
            self._resume_update = int(checkpoint.get("update", 0))
        else:
            agent_state = checkpoint
            self._resume_step = 0
            self._resume_update = 0

        if agent_state is None:
            raise ValueError("Checkpoint missing agent state")

        # Backwards compatibility: legacy checkpoints may store config inside agent state
        if "config" in agent_state:
            agent_state = {k: v for k, v in agent_state.items() if k in {"model", "optimizer"}}

        self.agent.load_state_dict(agent_state)

        # Ensure writer resumes with existing logs
        self.writer.add_text("resume", f"Resumed from {checkpoint_path}")

