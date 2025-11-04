"""
Wrapper para discretizar el espacio de acción en 5 acciones básicas.

Convierte el espacio de acción continuo Box(3,) a Discrete(5) con las siguientes acciones:
- 0: Turn left (steer=-1.0, gas=0.0, brake=0.0)
- 1: Turn right (steer=+1.0, gas=0.0, brake=0.0)
- 2: Brake (steer=0.0, gas=0.0, brake=0.8)
- 3: Accelerate (steer=0.0, gas=1.0, brake=0.0)
- 4: Do nothing (steer=0.0, gas=0.0, brake=0.0)
"""

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete

from .base import ActionSpaceWrapper


class Discrete5ActionWrapper(ActionSpaceWrapper):
    """
    Discretiza el espacio de acción continuo en 5 acciones básicas.
    
    Mapeo de acciones discretas a continuas:
    - 0: Turn left     → [-1.0, 0.0, 0.0]
    - 1: Turn right    → [+1.0, 0.0, 0.0]
    - 2: Brake         → [ 0.0, 0.0, 0.8]
    - 3: Accelerate    → [ 0.0, 1.0, 0.0]
    - 4: Do nothing    → [ 0.0, 0.0, 0.0]
    """
    
    # Mapeo de acciones discretas a continuas [steering, gas, brake]
    ACTION_MAP = {
        0: np.array([-1.0, 0.0, 0.0], dtype=np.float32),  # Turn left
        1: np.array([+1.0, 0.0, 0.0], dtype=np.float32),  # Turn right
        2: np.array([0.0, 0.0, 0.8], dtype=np.float32),   # Brake
        3: np.array([0.0, 1.0, 0.0], dtype=np.float32),   # Accelerate
        4: np.array([0.0, 0.0, 0.0], dtype=np.float32),   # Do nothing
    }
    
    # Nombres descriptivos para cada acción (útil para logging)
    ACTION_NAMES = {
        0: "turn_left",
        1: "turn_right",
        2: "brake",
        3: "accelerate",
        4: "do_nothing",
    }
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Cambiar el action_space a Discrete(5)
        self.action_space = Discrete(5)
    
    def action(self, action: int) -> np.ndarray:
        """
        Convierte la acción discreta del agente a la acción continua del environment.
        
        Args:
            action: Entero en [0, 4] representando la acción discreta
        
        Returns:
            Array numpy [steering, gas, brake] en formato continuo
        """
        if not isinstance(action, (int, np.integer)):
            action = int(action)
        
        if action not in self.ACTION_MAP:
            raise ValueError(f"Invalid action {action}. Must be in [0, 4].")
        
        return self.ACTION_MAP[action].copy()
    
    def get_action_name(self, action: int) -> str:
        """
        Obtiene el nombre descriptivo de una acción.
        
        Args:
            action: Entero en [0, 4]
        
        Returns:
            Nombre de la acción (ej: "turn_left", "accelerate")
        """
        return self.ACTION_NAMES.get(action, f"unknown_{action}")
