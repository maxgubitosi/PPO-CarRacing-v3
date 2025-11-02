from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor


class RolloutBuffer:
    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_shape: tuple[int, ...],
        action_dim: int,
        device: torch.device,
    ) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        buffer_shape = (num_steps, num_envs)

        self.observations = torch.zeros(buffer_shape + obs_shape, device=device)
        self.actions = torch.zeros(buffer_shape + (action_dim,), device=device)
        self.log_probs = torch.zeros(buffer_shape, device=device)
        self.rewards = torch.zeros(buffer_shape, device=device)
        self.dones = torch.zeros(buffer_shape, device=device)
        self.values = torch.zeros(buffer_shape, device=device)
        self.advantages = torch.zeros(buffer_shape, device=device)
        self.returns = torch.zeros(buffer_shape, device=device)

        self._step = 0

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        if self._step >= self.num_steps:
            raise IndexError("RolloutBuffer is full")

        self.observations[self._step].copy_(obs)
        self.actions[self._step].copy_(action)
        self.log_probs[self._step].copy_(log_prob)
        self.rewards[self._step].copy_(reward)
        self.dones[self._step].copy_(done)
        self.values[self._step].copy_(value)

        self._step += 1

    def compute_returns_and_advantages(
        self,
        last_values: torch.Tensor,
        last_dones: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        last_advantage = torch.zeros(self.num_envs, device=self.device)

        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]

            delta = (
                self.rewards[step]
                + gamma * next_values * next_non_terminal
                - self.values[step]
            )
            last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
            self.advantages[step] = last_advantage

        self.returns = self.advantages + self.values

    def get(self, minibatch_size: int) -> Iterator[RolloutBatch]:
        num_samples = self.num_steps * self.num_envs
        indices = torch.randperm(num_samples, device=self.device)

        observations = self.observations.reshape(num_samples, *self.observations.shape[2:])
        actions = self.actions.reshape(num_samples, -1)
        log_probs = self.log_probs.reshape(num_samples)
        advantages = self.advantages.reshape(num_samples)
        returns = self.returns.reshape(num_samples)
        values = self.values.reshape(num_samples)

        for start in range(0, num_samples, minibatch_size):
            end = start + minibatch_size
            batch_idx = indices[start:end]
            yield RolloutBatch(
                observations=observations[batch_idx],
                actions=actions[batch_idx],
                log_probs=log_probs[batch_idx],
                advantages=advantages[batch_idx],
                returns=returns[batch_idx],
                values=values[batch_idx],
            )

    def reset(self) -> None:
        self._step = 0

