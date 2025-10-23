"""Expert modules for the adaptive memory architecture."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

from .encoder import Vector
from .memory import MemoryEntry


@dataclass
class Expert:
    """Base expert that models a semantic region."""

    identifier: str
    centroid: Vector
    bandwidth: float = 0.2
    adapters: List[Vector] = field(default_factory=list)
    usage_count: int = 0

    def similarity_to_region(self, key: Vector) -> float:
        """Compute similarity between the key and expert region."""
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(key, self.centroid)))
        similarity = math.exp(-(distance ** 2) / (2 * self.bandwidth ** 2 + 1e-12))
        return float(similarity)

    def process(self, entries: Sequence[Tuple[MemoryEntry, float]]) -> dict:
        """Aggregate memory entries to produce a response."""
        if not entries:
            return {"confidence": 0.0}
        aggregated: dict[str, float] = {}
        total_weight = 0.0
        for entry, score in entries:
            weight = score * entry.meta.get("trust", 1.0)
            for key, value in entry.value.items():
                aggregated[key] = aggregated.get(key, 0.0) + value * weight
            total_weight += weight
        if total_weight > 0:
            for key in list(aggregated.keys()):
                aggregated[key] /= total_weight
        aggregated["confidence"] = min(1.0, total_weight)
        self.usage_count += 1
        return aggregated

    def adapt(self, entries: Iterable[MemoryEntry]) -> None:
        """Adapt expert parameters using entries."""
        vectors = [entry.key for entry in entries]
        if not vectors:
            return
        centroid = [0.0] * len(vectors[0])
        for vector in vectors:
            for i, value in enumerate(vector):
                centroid[i] += value
        centroid = [value / len(vectors) for value in centroid]
        norm = math.sqrt(sum(value * value for value in centroid)) + 1e-12
        self.centroid = [value / norm for value in centroid]


class ExpertManager:
    """Manages expert lifecycle."""

    def __init__(self, bandwidth: float = 0.2) -> None:
        self.bandwidth = bandwidth
        self._experts: List[Expert] = []
        self._counter = 0

    def experts(self) -> Sequence[Expert]:
        return list(self._experts)

    def spawn(self, centroid: Vector) -> Expert:
        identifier = f"expert_{self._counter:03d}"
        self._counter += 1
        expert = Expert(identifier=identifier, centroid=list(centroid), bandwidth=self.bandwidth)
        self._experts.append(expert)
        return expert

    def register(self, expert: Expert) -> None:
        """Register an externally constructed expert."""
        if any(existing.identifier == expert.identifier for existing in self._experts):
            return
        self._experts.append(expert)

    def consolidate(self, expert: Expert, entries: Iterable[MemoryEntry]) -> None:
        expert.adapt(entries)


__all__: Tuple[str, ...] = ("Expert", "ExpertManager")
