from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Callable, Sequence


@dataclass
class HistoricalBootstrapSampler[StateT, ExogenousT]:
    """Resamples observed exogenous records.

    block_size=1 samples independent historical records.
    block_size>1 samples consecutive blocks to preserve temporal correlation.
    """

    history: Sequence[ExogenousT]
    block_size: int = 1
    seed: int | None = None

    def __post_init__(self) -> None:
        if not self.history:
            raise ValueError("history must contain at least one exogenous record")
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        self._rng = Random(self.seed)
        self._block: list[ExogenousT] = []

    def reset(self, replication: int) -> None:
        self._rng = Random(None if self.seed is None else self.seed + replication)
        self._block = []

    def sample(self, state: StateT, t: int) -> ExogenousT:
        del state
        if self.block_size == 1:
            return self._rng.choice(list(self.history))

        if not self._block:
            max_start = max(0, len(self.history) - self.block_size)
            start = self._rng.randint(0, max_start)
            self._block = list(self.history[start : start + self.block_size])
        return self._block.pop(0)


@dataclass
class ScenarioSampler[StateT, ExogenousT]:
    """Wraps a callable for custom, forecast, or stress-test uncertainty."""

    fn: Callable[..., ExogenousT]

    def reset(self, replication: int) -> None:
        self.replication = replication

    def sample(self, state: StateT, t: int) -> ExogenousT:
        return self.fn(state=state, t=t, replication=getattr(self, "replication", 0))
