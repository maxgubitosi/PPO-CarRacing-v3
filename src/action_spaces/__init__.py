"""
Action space wrappers para CarRacing-v3.

Permite discretizar o modificar el espacio de acciones de forma modular.
"""

from .base import ActionSpaceWrapper
from .discrete_5 import Discrete5ActionWrapper
from .registry import get_action_wrapper, list_action_wrappers, register_action_wrapper

__all__ = [
    "ActionSpaceWrapper",
    "Discrete5ActionWrapper",
    "get_action_wrapper",
    "list_action_wrappers",
    "register_action_wrapper",
]
