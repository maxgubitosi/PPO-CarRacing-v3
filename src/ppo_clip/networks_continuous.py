from __future__ import annotations

import numpy as np
import torch
from gymnasium.spaces import Box
from torch import nn
from torch.distributions import Normal


class ContinuousActorCritic(nn.Module):
    """Actor-Critic para espacios de acción continuos (Box)."""
    
    def __init__(self, observation_space: Box, action_space: Box) -> None:
        super().__init__()
        if len(observation_space.shape) != 3:
            raise ValueError("CarRacing observations must be images with channel dimension")

        if not isinstance(action_space, Box):
            raise TypeError("PPO-Clip actor-critic requires continuous action space")

        c, h, w = observation_space.shape
        action_dim = int(np.prod(action_space.shape))

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            feature_dim = self.feature_extractor(dummy).shape[-1]

        hidden_dim = 128

        self.actor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.register_buffer("action_low", torch.from_numpy(action_space.low).float())
        self.register_buffer("action_high", torch.from_numpy(action_space.high).float())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(obs)

    def get_dist_and_value(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        features = self.forward(obs)
        mean = self.actor(features)
        log_std = self.log_std.clamp(-5, 2)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        value = self.critic(features).squeeze(-1)
        return dist, value

    def scale_action(self, action: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.min(action, self.action_high), self.action_low)

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.get_dist_and_value(obs)
        raw_action = dist.rsample()
        log_prob = dist.log_prob(raw_action).sum(-1)
        scaled_action = self.scale_action(raw_action)
        return scaled_action, log_prob, value, raw_action

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.get_dist_and_value(obs)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value

    def act_deterministic(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist, value = self.get_dist_and_value(obs)
        mean_action = dist.mean
        scaled_action = self.scale_action(mean_action)
        return scaled_action, value

