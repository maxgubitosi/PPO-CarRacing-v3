"""
Factory para crear arquitecturas Actor-Critic según el tipo de espacio de acción.
"""

from gymnasium.spaces import Box, Discrete, Space

from .networks_continuous import ContinuousActorCritic
from .networks_discrete import DiscreteActorCritic


def create_actor_critic(observation_space: Space, action_space: Space):
    """
    Crea una red Actor-Critic apropiada según el tipo de espacio de acción.
    
    Args:
        observation_space: Espacio de observaciones del environment
        action_space: Espacio de acciones del environment (Box o Discrete)
    
    Returns:
        ActorCritic network apropiada para el espacio de acción
    
    Raises:
        ValueError: Si el action_space no es Box ni Discrete
    """
    if isinstance(action_space, Box):
        return ContinuousActorCritic(observation_space, action_space)
    elif isinstance(action_space, Discrete):
        return DiscreteActorCritic(observation_space, action_space)
    else:
        raise ValueError(
            f"Unsupported action space type: {type(action_space)}. "
            f"Supported types: Box (continuous), Discrete"
        )
