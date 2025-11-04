"""
Clase base para wrappers de espacio de acción.
"""

from abc import ABC, abstractmethod
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Space


class ActionSpaceWrapper(gym.ActionWrapper, ABC):
    """
    Clase base abstracta para wrappers de espacio de acción.
    
    Los wrappers concretos deben:
    1. Definir el nuevo action_space en __init__
    2. Implementar action() para convertir la acción del agente a la acción del environment
    3. Opcionalmente implementar reverse_action() para logging/debugging
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Los subclasses deben definir self.action_space aquí
    
    @abstractmethod
    def action(self, action):
        """
        Convierte la acción del agente al formato que espera el environment.
        
        Args:
            action: Acción producida por el agente (formato del action_space wrapper)
        
        Returns:
            Acción en el formato original del environment
        """
        pass
    
    def reverse_action(self, action):
        """
        Convierte una acción del environment al formato del agente (opcional).
        Útil para logging o debugging.
        
        Args:
            action: Acción en el formato original del environment
        
        Returns:
            Acción en el formato del action_space wrapper
        """
        raise NotImplementedError("reverse_action not implemented for this wrapper")


class IdentityActionWrapper(ActionSpaceWrapper):
    """
    Wrapper identidad que no modifica el espacio de acción.
    Útil como caso base o para testing.
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Mantener el action_space original
        self.action_space = env.action_space
    
    def action(self, action):
        """No modifica la acción."""
        return action
    
    def reverse_action(self, action):
        """No modifica la acción."""
        return action
