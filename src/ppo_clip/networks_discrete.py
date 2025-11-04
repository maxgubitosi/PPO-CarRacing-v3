"""
Red Actor-Critic para espacios de acción discretos (Discrete).
"""

from __future__ import annotations

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from torch import nn
from torch.distributions import Categorical


class DiscreteActorCritic(nn.Module):
    """Actor-Critic para espacios de acción discretos (Discrete)."""
    
    def __init__(self, observation_space: Box, action_space: Discrete) -> None:
        super().__init__()
        if len(observation_space.shape) != 3:
            raise ValueError("CarRacing observations must be images with channel dimension")

        if not isinstance(action_space, Discrete):
            raise TypeError("DiscreteActorCritic requires Discrete action space")

        c, h, w = observation_space.shape
        num_actions = action_space.n

        # Feature extractor CNN (misma arquitectura que continuous)
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

        # Actor: produce logits para cada acción discreta
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

        # Critic: produce un valor escalar
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(obs)

    def get_dist_and_value(self, obs: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        """
        Obtiene la distribución de acciones y el valor del estado.
        
        Args:
            obs: Observaciones (B, C, H, W)
        
        Returns:
            dist: Distribución categórica sobre las acciones
            value: Valor del estado (B,)
        """
        features = self.forward(obs)
        logits = self.actor(features)
        dist = Categorical(logits=logits)
        value = self.critic(features).squeeze(-1)
        return dist, value

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Muestrea una acción de la distribución.
        
        Args:
            obs: Observaciones (B, C, H, W)
        
        Returns:
            action: Acción discreta muestreada (B,)
            log_prob: Log-probabilidad de la acción (B,)
            value: Valor del estado (B,)
        """
        dist, value = self.get_dist_and_value(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evalúa acciones dadas.
        
        Args:
            obs: Observaciones (B, C, H, W)
            actions: Acciones a evaluar (B,)
        
        Returns:
            log_prob: Log-probabilidad de las acciones (B,)
            entropy: Entropía de la distribución (B,)
            value: Valor del estado (B,)
        """
        dist, value = self.get_dist_and_value(obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value

    def act_deterministic(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Selecciona la acción más probable (modo greedy).
        
        Args:
            obs: Observaciones (B, C, H, W)
        
        Returns:
            action: Acción con mayor probabilidad (B,)
            value: Valor del estado (B,)
        """
        dist, value = self.get_dist_and_value(obs)
        action = dist.probs.argmax(dim=-1)
        return action, value
