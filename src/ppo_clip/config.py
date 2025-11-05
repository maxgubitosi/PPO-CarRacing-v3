from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PPOConfig:
    env_id: str = "CarRacing-v3"
    total_timesteps: int = 1_000_000
    num_envs: int = 4
    num_steps: int = 512
    num_stack: int = 4
    frame_skip: int = 0
    num_minibatches: int = 8
    update_epochs: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    value_coef: float = 0.5
    learning_rate: float = 5e-5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    seed: int = 42
    device: str = "cuda"
    torch_deterministic: bool = False
    track_eval: bool = True
    eval_episodes: int = 5
    log_interval: int = 10
    eval_interval: int = 50
    save_interval: int = 50
    log_root: Path = field(default_factory=lambda: Path("tensorboard_logs/ppo_clip"))
    checkpoint_root: Path = field(default_factory=lambda: Path("models/ppo_clip"))
    video_root: Path = field(default_factory=lambda: Path("videos/ppo_clip"))
    video_interval_minutes: float | None = 10.0
    max_video_steps: int = 1000
    max_offroad_seconds: float = 2.0
    offroad_penalty: float | None = None
    continuous: bool = True  # True para acciones continuas, False para discretas (5 acciones)

    def __post_init__(self) -> None:
        if self.num_stack <= 0:
            raise ValueError("num_stack must be positive")

        if self.frame_skip < 0:
            raise ValueError("frame_skip must be non-negative")

        if self.video_interval_minutes is not None and self.video_interval_minutes <= 0:
            raise ValueError("video_interval_minutes must be positive or None")

        if self.max_video_steps <= 0:
            raise ValueError("max_video_steps must be positive")

        if self.max_offroad_seconds <= 0:
            raise ValueError("max_offroad_seconds must be positive")

        try:
            import torch

            if self.device == "cuda" and not torch.cuda.is_available():
                self.device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
            elif self.device == "mps":
                if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
                    self.device = "cpu"
        except Exception:
            self.device = "cpu"

        if self.batch_size % self.num_minibatches != 0:
            raise ValueError("batch_size must be divisible by num_minibatches")

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def num_updates(self) -> int:
        updates = self.total_timesteps // self.batch_size
        if updates == 0:
            raise ValueError("total_timesteps is too small for the provided rollout setup")
        return updates

