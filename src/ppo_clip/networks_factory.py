"""Factory para crear la red Actor-Critic apropiada."""

from gymnasium.spaces import Box, Discrete, Space

from .networks_continuous import ContinuousActorCritic
from .networks_discrete import DiscreteActorCritic
from .networks_latent import LatentActorCritic


def create_actor_critic(
    observation_space: Space,
    action_space: Space,
    *,
    latent_hidden_dim: int | None = None,
):
   
    obs_dims = len(observation_space.shape)

    if obs_dims == 3:
        if isinstance(action_space, Box):
            return ContinuousActorCritic(observation_space, action_space)
        if isinstance(action_space, Discrete):
            return DiscreteActorCritic(observation_space, action_space)
    elif obs_dims == 1:
        return LatentActorCritic(observation_space, action_space, hidden_dim=latent_hidden_dim)

    raise ValueError(
        f"Espacio de observación no soportado: shape={observation_space.shape}, "
        f"action_space={type(action_space)}"
    )
