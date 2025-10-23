"""High-level adaptive memory system orchestration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Sequence, Tuple

from .encoder import EncoderOutput, FactEncoder
from .expert import ExpertManager, Expert
from .memory import AssociativeMemory, MemoryEntry
from .optimizer import BudgetConfig, MemoryParameterBudgeter
from .router import Router


@dataclass
class AdaptiveMemoryConfig:
    """Configuration for building the adaptive memory system."""

    encoder_dim: int = 128
    grow_threshold: float = 0.3
    max_memory_entries: int | None = None
    expert_bandwidth: float = 0.2
    router_top_k: int = 2
    budget_config: BudgetConfig = field(default_factory=BudgetConfig)


class AdaptiveMemorySystem:
    """Coordinates encoding, routing, memory, and experts."""

    def __init__(self, config: AdaptiveMemoryConfig | None = None) -> None:
        self.config = config or AdaptiveMemoryConfig()
        self.encoder = FactEncoder(dimension=self.config.encoder_dim)
        self.memory = AssociativeMemory(max_entries=self.config.max_memory_entries)
        self.expert_manager = ExpertManager(bandwidth=self.config.expert_bandwidth)
        self.router = Router(
            expert_manager=self.expert_manager,
            memory=self.memory,
            grow_threshold=self.config.grow_threshold,
            top_k=self.config.router_top_k,
        )
        self.budgeter = MemoryParameterBudgeter(config=self.config.budget_config)

    def remember(self, fact_tokens: Iterable[str], payload: Dict[str, float]) -> MemoryEntry:
        """Encode and store a fact, creating experts if necessary."""
        encoded: EncoderOutput = self.encoder.encode_fact(fact_tokens, payload)
        routing = self.router.route(encoded.key)
        selected_experts: Sequence[Expert]
        if routing.novelty > self.config.grow_threshold:
            new_expert = self.expert_manager.spawn(encoded.key)
            selected_experts = [new_expert]
        else:
            selected_experts = routing.selected_experts
        entry = MemoryEntry(key=encoded.key, value=encoded.value, meta=encoded.meta)
        self.memory.write(entry)
        for expert in selected_experts:
            expert.adapters.append(encoded.key)
        return entry

    def recall(self, query_tokens: Iterable[str], top_k: int = 5) -> Dict[str, float]:
        """Retrieve facts relevant to the query using experts."""
        query_key = self.encoder.encode_query(query_tokens)
        routing = self.router.route(query_key)
        neighbors = self.memory.read(query_key, top_k=top_k)
        if not routing.selected_experts:
            return self._aggregate_neighbors(neighbors)
        responses = [expert.process(neighbors) for expert in routing.selected_experts]
        return self._merge_responses(responses)

    def evaluate_budget(
        self,
        prediction_loss: float,
        parameter_bits: float,
        memory_bits: float,
        utility: float,
    ) -> float:
        return self.budgeter.objective(prediction_loss, parameter_bits, memory_bits, utility)

    @staticmethod
    def _aggregate_neighbors(neighbors: Sequence[Tuple[MemoryEntry, float]]) -> Dict[str, float]:
        if not neighbors:
            return {}
        total_weight = sum(score for _, score in neighbors) + 1e-12
        aggregated: Dict[str, float] = {}
        for entry, score in neighbors:
            for key, value in entry.value.items():
                aggregated[key] = aggregated.get(key, 0.0) + value * score
        for key in aggregated:
            aggregated[key] /= total_weight
        aggregated["confidence"] = min(1.0, total_weight)
        return aggregated

    @staticmethod
    def _merge_responses(responses: Iterable[Dict[str, float]]) -> Dict[str, float]:
        merged: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        confidence = 0.0
        for response in responses:
            confidence = max(confidence, response.get("confidence", 0.0))
            for key, value in response.items():
                if key == "confidence":
                    continue
                merged[key] = merged.get(key, 0.0) + value
                counts[key] = counts.get(key, 0) + 1
        for key, total in list(merged.items()):
            merged[key] = total / max(counts.get(key, 1), 1)
        if merged or confidence:
            merged["confidence"] = confidence
        return merged


__all__: Tuple[str, ...] = ("AdaptiveMemorySystem", "AdaptiveMemoryConfig")
