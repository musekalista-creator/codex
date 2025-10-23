"""Associative memory implementation."""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

from .encoder import Vector


@dataclass
class MemoryEntry:
    """Single entry stored in the associative memory."""

    key: Vector = field(compare=False, repr=False)
    value: Dict[str, float] = field(compare=False)
    meta: Dict[str, float] = field(default_factory=dict, compare=False)
    timestamp: float = field(default_factory=time.time)


class AssociativeMemory:
    """Append-only memory supporting cosine similarity retrieval."""

    def __init__(self, max_entries: int | None = None) -> None:
        self.max_entries = max_entries
        self._entries: List[MemoryEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    def write(self, entry: MemoryEntry) -> None:
        """Append an entry to memory, evicting oldest if necessary."""

        self._entries.append(entry)
        if self.max_entries is not None and len(self._entries) > self.max_entries:
            self._entries.pop(0)

    def bulk_write(self, entries: Iterable[MemoryEntry]) -> None:
        for entry in entries:
            self.write(entry)

    def read(self, query: Vector, top_k: int = 5) -> Sequence[Tuple[MemoryEntry, float]]:
        """Retrieve the top-k entries most similar to the query."""
        if not self._entries:
            return []
        heap: List[Tuple[float, MemoryEntry]] = []
        for entry in self._entries:
            similarity = self._cosine_similarity(query, entry.key)
            if len(heap) < top_k:
                heapq.heappush(heap, (similarity, entry))
            else:
                if similarity > heap[0][0]:
                    heapq.heapreplace(heap, (similarity, entry))
        results = sorted(heap, key=lambda item: item[0], reverse=True)
        scored_entries = [(entry, score) for score, entry in results]
        now = time.time()
        for entry, score in scored_entries:
            entry.meta["usage_count"] = entry.meta.get("usage_count", 0.0) + 1.0
            entry.meta["last_access"] = now
            entry.meta["confidence"] = max(entry.meta.get("confidence", 0.0), score)
        return scored_entries

    def best_similarity(self, query: Vector) -> float:
        """Return the highest similarity between the query and stored keys."""

        if not self._entries:
            return 0.0
        return max(self._cosine_similarity(query, entry.key) for entry in self._entries)

    @staticmethod
    def _cosine_similarity(a: Vector, b: Vector) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a)) + 1e-12
        norm_b = math.sqrt(sum(y * y for y in b)) + 1e-12
        return float(dot / (norm_a * norm_b))


__all__: Tuple[str, ...] = ("AssociativeMemory", "MemoryEntry")
