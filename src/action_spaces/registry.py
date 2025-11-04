"""
Registry para action space wrappers.

Permite registrar, buscar y listar wrappers de espacio de acción disponibles.
"""

from typing import Callable, Dict, Type
import gymnasium as gym

from .base import ActionSpaceWrapper, IdentityActionWrapper
from .discrete_5 import Discrete5ActionWrapper


# Registry de wrappers disponibles
_ACTION_WRAPPER_REGISTRY: Dict[str, Type[ActionSpaceWrapper]] = {}


def register_action_wrapper(name: str, wrapper_class: Type[ActionSpaceWrapper]) -> None:
    """
    Registra un nuevo action space wrapper.
    
    Args:
        name: Identificador único para el wrapper (ej: "discrete_5")
        wrapper_class: Clase del wrapper que hereda de ActionSpaceWrapper
    """
    if not issubclass(wrapper_class, ActionSpaceWrapper):
        raise TypeError(f"{wrapper_class} must inherit from ActionSpaceWrapper")
    
    if name in _ACTION_WRAPPER_REGISTRY:
        raise ValueError(f"Action wrapper '{name}' already registered")
    
    _ACTION_WRAPPER_REGISTRY[name] = wrapper_class


def get_action_wrapper(name: str) -> Type[ActionSpaceWrapper]:
    """
    Obtiene un action space wrapper por su nombre.
    
    Args:
        name: Identificador del wrapper
    
    Returns:
        Clase del wrapper
    
    Raises:
        ValueError: Si el wrapper no está registrado
    """
    if name not in _ACTION_WRAPPER_REGISTRY:
        available = list(_ACTION_WRAPPER_REGISTRY.keys())
        raise ValueError(
            f"Action wrapper '{name}' not found. "
            f"Available wrappers: {available}"
        )
    
    return _ACTION_WRAPPER_REGISTRY[name]


def list_action_wrappers() -> list[str]:
    """
    Lista todos los action space wrappers registrados.
    
    Returns:
        Lista de nombres de wrappers disponibles
    """
    return sorted(_ACTION_WRAPPER_REGISTRY.keys())


# Registrar wrappers built-in
register_action_wrapper("continuous", IdentityActionWrapper)
register_action_wrapper("discrete_5", Discrete5ActionWrapper)
