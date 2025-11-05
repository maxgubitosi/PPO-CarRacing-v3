from __future__ import annotations

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from torch import nn
from torch.distributions import Categorical, Normal


class LatentActorCritic(nn.Module):
    def __init__(self, observation_space: Box, action_space) -> None:
        super().__init__()
        if len(observation_space.shape) != 1:
            raise ValueError("LatentActorCritic expects 1D latent observations.")

        self.is_discrete = isinstance(action_space, Discrete)

        latent_dim = int(np.prod(observation_space.shape))

        if self.is_discrete:
            action_dim = action_space.n
            self.actor = nn.Linear(latent_dim, action_dim)
            self.log_std = None
            self.register_buffer("action_low", torch.tensor(0.0))
            self.register_buffer("action_high", torch.tensor(float(action_dim - 1)))
        elif isinstance(action_space, Box):
            action_dim = int(np.prod(action_space.shape))
            self.actor = nn.Linear(latent_dim, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
            self.register_buffer("action_low", torch.from_numpy(action_space.low).float())
            self.register_buffer("action_high", torch.from_numpy(action_space.high).float())
        else:
            raise TypeError("LatentActorCritic supports Box (continuous) or Discrete action spaces.")

        self.action_dim = action_dim
        self.critic = nn.Linear(latent_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return obs

    def get_dist_and_value(self, obs: torch.Tensor):
        features = self.forward(obs)
        value = self.critic(features).squeeze(-1)
        if self.is_discrete:
            logits = self.actor(features)
            dist = Categorical(logits=logits)
        else:
            mean = self.actor(features)
            log_std = self.log_std.clamp(-5, 2)
            std = torch.exp(log_std)
            dist = Normal(mean, std)
        return dist, value

    def scale_action(self, action: torch.Tensor) -> torch.Tensor:
        if self.is_discrete:
            return action.long()
        return torch.max(torch.min(action, self.action_high), self.action_low)

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.get_dist_and_value(obs)
        if self.is_discrete:
            action = dist.sample().long()
            log_prob = dist.log_prob(action)
            return action, log_prob, value, action
        raw_action = dist.rsample()
        log_prob = dist.log_prob(raw_action).sum(-1)
        scaled_action = self.scale_action(raw_action)
        return scaled_action, log_prob, value, raw_action

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.get_dist_and_value(obs)
        if self.is_discrete:
            actions = actions.long()
            if actions.dim() > 1:
                actions = actions.squeeze(-1)
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()
        else:
            log_prob = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value

    def act_deterministic(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist, value = self.get_dist_and_value(obs)
        if self.is_discrete:
            action = torch.argmax(dist.probs, dim=-1)
            return action, value
        mean_action = dist.mean
        scaled_action = self.scale_action(mean_action)
        return scaled_action, value
