"""Adaptive memory architecture implementation."""

from .encoder import FactEncoder, EncoderOutput
from .memory import AssociativeMemory, MemoryEntry
from .router import Router, RoutingResult
from .expert import Expert, ExpertManager
from .optimizer import MemoryParameterBudgeter, BudgetConfig
from .system import AdaptiveMemorySystem, AdaptiveMemoryConfig

__all__ = [
    "FactEncoder",
    "EncoderOutput",
    "AssociativeMemory",
    "MemoryEntry",
    "Router",
    "RoutingResult",
    "Expert",
    "ExpertManager",
    "MemoryParameterBudgeter",
    "BudgetConfig",
    "AdaptiveMemorySystem",
    "AdaptiveMemoryConfig",
]
