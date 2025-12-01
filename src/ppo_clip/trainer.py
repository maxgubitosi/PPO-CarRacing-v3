from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict

import imageio.v2 as imageio
import numpy as np
import torch
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    from torch.serialization import add_safe_globals
except ImportError:
    add_safe_globals = None

from environment import create_single_env, create_vector_env
from utils import set_seed

from .agent import PPOClipAgent
from .config import PPOConfig
from .rollout_buffer import RolloutBuffer


class _NullSummaryWriter:
    def __getattr__(self, name):
        def noop(*args, **kwargs):
            return None
        return noop


class PPOTrainer:
    def __init__(
        self,
        config: PPOConfig,
        *,
        vector_env_builder: Callable[[PPOConfig], gym.Env] | None = None,
        single_env_builder: Callable[[PPOConfig, int, str | None], gym.Env] | None = None,
        frame_transform: Callable[[np.ndarray, gym.Env], np.ndarray] | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.project_root = Path(__file__).resolve().parents[2]
        self._vector_env_builder = vector_env_builder
        self._single_env_builder = single_env_builder
        self._frame_transform = frame_transform

        set_seed(config.seed, deterministic=config.torch_deterministic)

        if self._vector_env_builder:
            self.env = self._vector_env_builder(config)
        else:
            self.env = create_vector_env(
                config.env_id,
                config.num_envs,
                config.seed,
                offroad_penalty=config.offroad_penalty,
                max_offroad_seconds=config.max_offroad_seconds,
                continuous=config.continuous,
                frame_skip_between_frames=config.frame_skip,
                num_stack=config.num_stack,
                steering_constraint=config.steering_constraint,
            )
        self.eval_env_seeds = [config.seed + 10 + i for i in range(config.eval_episodes)]
        self.eval_env_index = 0
        self.eval_env = (
            self._build_single_env(self.eval_env_seeds[self.eval_env_index], render_mode=None)
            if config.track_eval
            else None
        )

        obs_space = self.env.single_observation_space
        action_space = self.env.single_action_space

        self.agent = PPOClipAgent(obs_space, action_space, config)

        self._resume_step = 0
        self._resume_update = 0
        
        # Núm total de updates para la corrida actual
        self.num_updates = config.total_timesteps // (config.num_steps * config.num_envs)
        
        self.lr_scheduler = None

        # Detectar tipo de espacio de acción (discreto o continuo)
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
        self.write_artifacts = getattr(config, "write_artifacts", True)
        if self.write_artifacts:
            self.log_dir = (self.project_root / config.log_root / self.run_name).resolve()
            self.checkpoint_dir = (self.project_root / config.checkpoint_root / self.run_name).resolve()
            self.video_dir = (self.project_root / config.video_root / self.run_name).resolve()
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if config.video_interval_minutes is not None:
                self.video_dir.mkdir(parents=True, exist_ok=True)

            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            self.writer.add_text("config", str(asdict(config)))
        else:
            self.log_dir = None
            self.checkpoint_dir = None
            self.video_dir = None
            self.writer = _NullSummaryWriter()

        if self.write_artifacts and config.video_interval_minutes is not None:
            self.video_interval_seconds = config.video_interval_minutes * 60
        else:
            self.video_interval_seconds = None
        self._last_video_time = time.perf_counter()
        self._video_seed_offset = 1000
        self.collect_timing_metrics = config.collect_timing_metrics
        self.profile_history: list[dict[str, float]] = []
        self.completed_steps = 0
        
        # Crear scheduler (se recrea al reanudar entrenamiento)
        self._create_lr_scheduler()

    def _create_lr_scheduler(self) -> None:
        """Create or recreate the learning rate scheduler based on current config and resume state."""
        if not self.config.use_lr_scheduler:
            self.lr_scheduler = None
            return
        
        remaining_updates = self.num_updates - self._resume_update
        
        # Restablecer cualquier scheduler previo para evitar heredar estado
        if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
            del self.lr_scheduler
            self.lr_scheduler = None
        
        # Forzar que el optimizer use la tasa actual como base
        for param_group in self.agent.optimizer.param_groups:
            param_group['initial_lr'] = self.config.learning_rate
            param_group['lr'] = self.config.learning_rate
        
        if self.config.verbose:
            print(f"\n[DEBUG _create_lr_scheduler]")
            print(f"  config.learning_rate: {self.config.learning_rate:.6f}")
            print(f"  config.lr_end: {self.config.lr_end:.6f}")
            print(f"  num_updates: {self.num_updates}")
            print(f"  _resume_update: {self._resume_update}")
            print(f"  remaining_updates: {remaining_updates}")
            print(f"  LR antes del scheduler: {self.agent.optimizer.param_groups[0]['lr']:.6f}")
        
        def lr_lambda(update_relative):
            if remaining_updates <= 0:
                return self.config.lr_end / self.config.learning_rate
            progress = update_relative / remaining_updates
            return 1.0 - progress * (1.0 - self.config.lr_end / self.config.learning_rate)
        
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.agent.optimizer, 
            lr_lambda=lr_lambda
        )
        
        if self.config.verbose:
            print(f"  LR tras crear scheduler: {self.agent.optimizer.param_groups[0]['lr']:.6f}")
            print(f"  Lambda inicial: {lr_lambda(0):.6f}")
            print(f"  LR esperado: {self.config.learning_rate * lr_lambda(0):.6f}\n")

    def train(self) -> None:
        """
        Ejecuta el bucle principal de entrenamiento PPO.

        1) Recolecta rollouts. (trayectorias)
        2) Calcula ventajas y returns. (GAE)
        3) Actualiza la política usando los rollouts. 
        4) Loggea métricas y guarda checkpoints periódicamente. 
        5) Evalúa la política periódicamente. 
        """
        obs, _ = self.env.reset(seed=self.config.seed)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        next_done = torch.zeros(self.config.num_envs, device=self.device)

        global_step = self._resume_step
        
        # Barra de progreso
        pbar = tqdm(
            total=self.config.total_timesteps,
            initial=self._resume_step,
            desc="Training",
            unit="steps"
        )

        for update in range(self._resume_update + 1, self.config.num_updates + 1):
            profile_entry: dict[str, float] | None = None
            if self.collect_timing_metrics:
                profile_entry = {
                    "update": float(update),
                    "rollout_timesteps": float(self.config.batch_size),
                }
                rollout_wall_start = time.perf_counter()
                rollout_cpu_start = time.process_time()

            # 1)Recolectar trayectorias
            for step in range(self.config.num_steps):
                with torch.no_grad():
                    sample = self.agent.sample(obs)

                actions = sample["action"].cpu().numpy()
                next_obs, rewards, terminated, truncated, infos = self.env.step(actions)

                # Reward shaping opcional (limitar rewards positivos)
                if self.config.reward_shaping:
                    rewards = np.clip(rewards, a_min=None, a_max=1.0)

                dones = np.logical_or(terminated, truncated)

                reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
                done_tensor = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

                # Almacenar transición en el buffer
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
                
                pbar.update(self.config.num_envs)
                self._log_rollout_infos(infos, global_step)

            if profile_entry is not None:
                profile_entry["global_step"] = float(global_step)
                profile_entry["rollout_wall_s"] = time.perf_counter() - rollout_wall_start
                profile_entry["rollout_cpu_s"] = time.process_time() - rollout_cpu_start
                update_wall_start = time.perf_counter()
                update_cpu_start = time.process_time()
            else:
                update_wall_start = 0.0
                update_cpu_start = 0.0
            
            # 2) GAE: calcular ventajas y returns
            with torch.no_grad():
                _, last_values = self.agent.network.get_dist_and_value(obs)

            self.buffer.compute_returns_and_advantages(
                last_values=last_values.detach(),
                last_dones=next_done,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
            )
            metrics = defaultdict(list)
            
            # 3) Actualizar política usando los rollouts
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

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # 4) Loggear métricas
            self._log_update_metrics(metrics, global_step, update)

            if profile_entry is not None:
                profile_entry["update_wall_s"] = time.perf_counter() - update_wall_start
                profile_entry["update_cpu_s"] = time.process_time() - update_cpu_start
                profile_entry["total_wall_s"] = profile_entry["rollout_wall_s"] + profile_entry["update_wall_s"]
                profile_entry["total_cpu_s"] = profile_entry["rollout_cpu_s"] + profile_entry["update_cpu_s"]
                self.profile_history.append(profile_entry)

            # 5) Evaluar política periódicamente (segun param)
            if self.config.track_eval and update % self.config.eval_interval == 0:
                self._evaluate(global_step)

            if self.write_artifacts and update % self.config.save_interval == 0:
                self._save_checkpoint(update, global_step)

            if self.video_interval_seconds is not None:
                now = time.perf_counter()
                if now - self._last_video_time >= self.video_interval_seconds:
                    self._record_video(global_step)
                    self._last_video_time = now

        pbar.close()
        self.env.close()
        if self.eval_env:
            self.eval_env.close()
        self.writer.close()
        self.completed_steps = global_step

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
        
        # Loggear distribución de acciones del buffer
        self._log_action_distribution(global_step)

    def _log_action_distribution(self, global_step: int) -> None:

        actions = self.buffer.actions.cpu().numpy()  # (num_steps, num_envs, action_dim)
        
        # Aplanar para obtener todas las acciones del rollout
        actions_flat = actions.reshape(-1, actions.shape[-1])
        
        if self.buffer.is_discrete:
            # Para acciones discretas: contar frecuencia de cada acción
            flattened = actions_flat.flatten().astype(int)
            action_space = getattr(self.env, "single_action_space", None)
            num_actions = int(getattr(action_space, "n", flattened.max(initial=0) + 1))
            action_counts = np.bincount(flattened, minlength=num_actions)
            total = action_counts.sum() or 1
            action_freq = action_counts / total

            action_names = self._resolve_discrete_action_names(num_actions)
            for name, freq in zip(action_names, action_freq):
                self.writer.add_scalar(f"actions/discrete/{name}", freq, global_step)
        else:
            # Para acciones continuas: estadísticas de steering, gas, brake
            steering = actions_flat[:, 0]
            gas = actions_flat[:, 1]
            brake = actions_flat[:, 2]
            
            # Loggear medias y desviaciones estándar
            self.writer.add_scalar("actions/continuous/steering_mean", float(np.mean(steering)), global_step)
            self.writer.add_scalar("actions/continuous/steering_std", float(np.std(steering)), global_step)
            self.writer.add_scalar("actions/continuous/gas_mean", float(np.mean(gas)), global_step)
            self.writer.add_scalar("actions/continuous/gas_std", float(np.std(gas)), global_step)
            self.writer.add_scalar("actions/continuous/brake_mean", float(np.mean(brake)), global_step)
            self.writer.add_scalar("actions/continuous/brake_std", float(np.std(brake)), global_step)
            
            # loggear histogramas
            self.writer.add_histogram("actions/continuous/steering_hist", steering, global_step)
            self.writer.add_histogram("actions/continuous/gas_hist", gas, global_step)
            self.writer.add_histogram("actions/continuous/brake_hist", brake, global_step)

    def _resolve_discrete_action_names(self, num_actions: int) -> list[str]:
        canonical = ["Do Nothing", "Right", "Left", "Gas", "Brake"]
        constraint = (self.config.steering_constraint or "").strip().lower()
        if constraint == "only_left":
            removed = 1  # eliminar "Right"
        elif constraint == "only_right":
            removed = 2  # eliminar "Left"
        else:
            removed = None

        if removed is not None:
            filtered = [name for idx, name in enumerate(canonical) if idx != removed]
        else:
            filtered = canonical
        return filtered[:num_actions]

    def _evaluate(self, global_step: int) -> None:
        if self.eval_env is None:
            return

        returns = []
        lengths = []
        deaths = []  # Episodios que terminaron por alejarse demasiado de la pista

        for idx in range(self.config.eval_episodes):
            seed = self.eval_env_seeds[self.eval_env_index]
            self.eval_env_index = (self.eval_env_index + 1) % len(self.eval_env_seeds)

            obs, _ = self.eval_env.reset(seed=seed)
            done = False
            total_reward = 0.0
            steps = 0
            died = False

            while not done:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action, _ = self.agent.act_deterministic(obs_tensor)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action.squeeze(0).cpu().numpy())
                done = terminated or truncated
                total_reward += reward
                steps += 1
                
                # Detectar muerte: reward de -100 indica que se fue demasiado de la pista
                if reward <= -99.0:
                    died = True

            returns.append(total_reward)
            lengths.append(steps)
            deaths.append(1 if died else 0)

        self.writer.add_scalar("eval/return_mean", float(np.mean(returns)), global_step)
        self.writer.add_scalar("eval/return_std", float(np.std(returns)), global_step)
        self.writer.add_scalar("eval/return_max", float(np.max(returns)), global_step)
        self.writer.add_scalar("eval/return_min", float(np.min(returns)), global_step)
        self.writer.add_scalar("eval/episode_length", float(np.mean(lengths)), global_step)
        self.writer.add_scalar("eval/death_rate", float(np.mean(deaths)), global_step)  # % de episodios que murieron
        
        if self.config.verbose:
            print(
                f"Eval @ {global_step}: "
                f"return={np.mean(returns):.2f}±{np.std(returns):.2f} "
                f"(max={np.max(returns):.2f}, min={np.min(returns):.2f}), "
                f"length={np.mean(lengths):.0f}, "
                f"death_rate={np.mean(deaths)*100:.0f}%"
            )

    def _save_checkpoint(self, update: int, global_step: int) -> None:
        if not (self.write_artifacts and self.checkpoint_dir):
            return
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
        if not (self.write_artifacts and self.video_dir):
            return
        env_seed = self.eval_env_seeds[self.eval_env_index]
        self.eval_env_index = (self.eval_env_index + 1) % len(self.eval_env_seeds)
        env = self._build_single_env(env_seed, render_mode="rgb_array")

        frames = []
        obs, _ = env.reset()
        frame = env.render()
        if frame is not None:
            if self._frame_transform is not None:
                frame = self._frame_transform(frame, env)
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
                if self._frame_transform is not None:
                    frame = self._frame_transform(frame, env)
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

        # Compatibilidad hacia atrás: checkpoints viejos pueden incluir config en estado del agente
        if "config" in agent_state:
            agent_state = {k: v for k, v in agent_state.items() if k in {"model", "optimizer"}}

        self.agent.load_state_dict(agent_state)
        
        # Sobrescribir el LR del optimizador con el valor actual del config
        for param_group in self.agent.optimizer.param_groups:
            param_group['lr'] = self.config.learning_rate
        
        if self.config.verbose:
            print(f"\n[Checkpoint Loaded]")
            print(f"  Resumed from update {self._resume_update}, step {self._resume_step:,}")
            print(f"  Optimizer LR was set to: {self.config.learning_rate:.6f}")

        # Recalcular updates según los nuevos timesteps totales
        self.num_updates = self.config.total_timesteps // (self.config.num_steps * self.config.num_envs)
        remaining_updates = self.num_updates - self._resume_update
        total_steps_remaining = remaining_updates * self.config.num_steps * self.config.num_envs
        
        # Recrear el scheduler de LR con los updates restantes
        self._create_lr_scheduler()
        
        current_lr = self.agent.optimizer.param_groups[0]["lr"]
        
        if self.config.verbose:
            print(f"\n{'='*70}")
            print(f"LR Scheduler Configuration:")
            print(f"  Checkpoint: Update {self._resume_update}, Step {self._resume_step:,}")
            print(f"  Target: Update {self.num_updates}, Step {self.config.total_timesteps:,}")
            print(f"  Remaining: {remaining_updates} updates ({total_steps_remaining:,} steps)")
            print(f"  LR Schedule: {current_lr:.6f} → {self.config.lr_end:.6f}")
            print(f"  LR will decay linearly over the next {total_steps_remaining:,} steps")
            print(f"{'='*70}\n")

        self.writer.add_text("resume", f"Resumed from {checkpoint_path}")

    def _build_single_env(self, seed: int, render_mode: str | None) -> gym.Env:
        if self._single_env_builder:
            return self._single_env_builder(self.config, seed, render_mode)
        return create_single_env(
            self.config.env_id,
            seed,
            render_mode,
            offroad_penalty=self.config.offroad_penalty,
            max_offroad_seconds=self.config.max_offroad_seconds,
            continuous=self.config.continuous,
            frame_skip_between_frames=self.config.frame_skip,
            num_stack=self.config.num_stack,
            steering_constraint=self.config.steering_constraint,
        )
