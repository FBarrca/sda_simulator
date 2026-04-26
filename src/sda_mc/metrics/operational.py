from __future__ import annotations

from typing import Callable

from sda_mc.core.types import Trajectory

from .base import MetricSpec
from .risk import Tail


def operational_metric[StateT, DecisionT, ExogenousT](
    name: str,
    fn: Callable[[Trajectory[StateT, DecisionT, ExogenousT]], float],
    *,
    higher_is_better: bool | None = None,
    tail: Tail | None = None,
    tail_alpha: float = 0.95,
) -> MetricSpec[StateT, DecisionT, ExogenousT]:
    return MetricSpec(
        name=name,
        fn=fn,
        higher_is_better=higher_is_better,
        tail=tail,
        tail_alpha=tail_alpha,
    )
