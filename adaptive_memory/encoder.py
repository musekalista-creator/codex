"""Encoding module for the adaptive memory architecture."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple


Vector = List[float]


def _normalize_text(tokens: Iterable[str]) -> Iterator[str]:
    """Lower-case and strip whitespace for stable hashing."""

    for token in tokens:
        cleaned = token.strip().lower()
        if cleaned:
            yield cleaned


@dataclass
class EncoderOutput:
    """Container for encoder outputs."""

    key: Vector
    value: Dict[str, float]
    meta: Dict[str, float]


class FactEncoder:
    """Simple fact encoder producing key-value representations."""

    def __init__(self, dimension: int = 128, random_state: int | None = None) -> None:
        if dimension <= 0:
            raise ValueError("encoder dimension must be positive")
        self.dimension = dimension
        self.random = random.Random(random_state)

    def encode_fact(self, fact_tokens: Iterable[str], payload: Dict[str, float]) -> EncoderOutput:
        """Encode a fact represented as tokens and numeric payload."""

        key = self._encode_tokens(fact_tokens)
        payload_copy = dict(payload)
        trust = payload_copy.pop("trust", 1.0)
        meta = {
            "usage_count": 0.0,
            "trust": trust,
        }
        return EncoderOutput(key=key, value=payload_copy, meta=meta)

    def encode_query(self, query_tokens: Iterable[str]) -> Vector:
        """Encode a query using the same key space as facts."""

        return self._encode_tokens(query_tokens)

    def _encode_tokens(self, tokens: Iterable[str]) -> Vector:
        normalized_tokens = list(_normalize_text(tokens))
        if not normalized_tokens:
            return self._random_vector()
        token_vectors = [self._encode_token(token) for token in normalized_tokens]
        bigram_vectors = [
            self._encode_token(f"{a}|{b}")
            for a, b in zip(normalized_tokens[:-1], normalized_tokens[1:])
        ]
        all_vectors = token_vectors + bigram_vectors
        return self._aggregate_vectors(all_vectors)

    def _aggregate_vectors(self, vectors: Sequence[Vector]) -> Vector:
        summed = [0.0] * self.dimension
        for vector in vectors:
            for i, value in enumerate(vector):
                summed[i] += value
        key = [value / max(len(vectors), 1) for value in summed]
        return self._normalize(key)

    def _encode_token(self, token: str) -> Vector:
        seed = abs(hash(token)) % (2**32)
        token_rng = random.Random(seed)
        return [token_rng.gauss(0.0, 1.0) for _ in range(self.dimension)]

    def _random_vector(self) -> Vector:
        return self._normalize([self.random.gauss(0.0, 1.0) for _ in range(self.dimension)])

    @staticmethod
    def _normalize(vector: Vector) -> Vector:
        norm = math.sqrt(sum(value * value for value in vector)) + 1e-12
        return [value / norm for value in vector]


__all__: Tuple[str, ...] = ("FactEncoder", "EncoderOutput")
