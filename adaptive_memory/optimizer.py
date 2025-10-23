"""Budget optimizer for balancing memory and parameters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class BudgetConfig:
    """Configuration parameters for the memory budget."""

    lambda_params: float = 1.0
    lambda_memory: float = 1.0
    beta_utility: float = 1.0


class MemoryParameterBudgeter:
    """Simple optimizer computing a scalar objective for budgeting."""

    def __init__(self, config: BudgetConfig | None = None) -> None:
        self.config = config or BudgetConfig()

    def objective(
        self,
        prediction_loss: float,
        parameter_bits: float,
        memory_bits: float,
        utility: float,
    ) -> float:
        cfg = self.config
        return (
            prediction_loss
            + cfg.lambda_params * parameter_bits
            + cfg.lambda_memory * memory_bits
            - cfg.beta_utility * utility
        )


__all__: Tuple[str, ...] = ("MemoryParameterBudgeter", "BudgetConfig")
