from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn.functional as F
from gymnasium.spaces import Box, Discrete, Space

from .config import PPOConfig
from .networks_factory import create_actor_critic
from .rollout_buffer import RolloutBatch


@dataclass
class UpdateStats:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor


class PPOClipAgent:
    def __init__(self, observation_space: Space, action_space: Space, config: PPOConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.network = create_actor_critic(
            observation_space,
            action_space,
            latent_hidden_dim=config.latent_hidden_dim,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
            eps=1e-5,
            weight_decay=config.weight_decay,
        )
        self.is_discrete = isinstance(action_space, Discrete)

    def sample(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Muestrea una acción de la política."""
        result = self.network.act(obs)
        if len(result) == 4:
            action, log_prob, value, raw_action = result
        else:
            action, log_prob, value = result
            raw_action = action
        return {
            "action": action,
            "log_prob": log_prob,
            "value": value,
            "raw_action": raw_action,
        }

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.network.evaluate_actions(obs, actions)

    def update(self, batch: RolloutBatch) -> UpdateStats:
        advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std(unbiased=False) + 1e-8)

        log_prob, entropy, values = self.evaluate(batch.observations, batch.actions)
        ratio = (log_prob - batch.log_probs).exp()
        unclipped = advantages * ratio
        clipped = advantages * torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef)
        policy_loss = -torch.min(unclipped, clipped).mean()

        value_loss = F.mse_loss(batch.returns, values)
        loss = policy_loss + self.config.value_coef * value_loss - self.config.ent_coef * entropy.mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            approx_kl = ((batch.log_probs - log_prob).mean()).abs()

        return UpdateStats(
            loss=loss.detach(),
            policy_loss=policy_loss.detach(),
            value_loss=value_loss.detach(),
            entropy=entropy.mean().detach(),
            approx_kl=approx_kl,
        )

    def act_deterministic(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.network.act_deterministic(obs)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.network.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])

