"""Routing module for the adaptive memory architecture."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .encoder import Vector
from .expert import Expert, ExpertManager
from .memory import AssociativeMemory


@dataclass
class RoutingResult:
    """Result from the router describing selected experts and novelty."""

    novelty: float
    selected_experts: Sequence[Expert]


class Router:
    """Routes facts and queries to appropriate experts."""

    def __init__(
        self,
        expert_manager: ExpertManager,
        memory: AssociativeMemory,
        grow_threshold: float = 0.3,
        top_k: int = 2,
    ) -> None:
        self._expert_manager = expert_manager
        self._memory = memory
        self.grow_threshold = grow_threshold
        self._top_k = top_k

    @property
    def experts(self) -> Sequence[Expert]:
        return list(self._expert_manager.experts())

    def register_expert(self, expert: Expert) -> None:
        self._expert_manager.register(expert)

    def route(self, key: Vector, top_k: int | None = None) -> RoutingResult:
        """Route the key to matching experts and compute novelty."""

        novelty = max(0.0, min(1.0, 1.0 - self._memory.best_similarity(key)))
        experts = self._expert_manager.experts()
        if not experts:
            return RoutingResult(novelty=max(novelty, 1e-3), selected_experts=[])
        similarities = [expert.similarity_to_region(key) for expert in experts]
        gated = self._gate(similarities, top_k=top_k)
        selected = [experts[i] for i in gated]
        return RoutingResult(novelty=novelty, selected_experts=selected)

    def _gate(self, similarities: Sequence[float], top_k: int | None = None) -> List[int]:
        if not similarities:
            return []
        top_k = min(top_k or self._top_k, len(similarities))
        weights = self._softmax(similarities)
        ranked = sorted(range(len(similarities)), key=lambda idx: weights[idx], reverse=True)
        return ranked[:top_k]

    @staticmethod
    def _softmax(similarities: Sequence[float]) -> List[float]:
        if not similarities:
            return []
        max_sim = max(similarities)
        exps = [math.exp(sim - max_sim) for sim in similarities]
        total = sum(exps) + 1e-12
        return [value / total for value in exps]


__all__: Tuple[str, ...] = ("Router", "RoutingResult")
