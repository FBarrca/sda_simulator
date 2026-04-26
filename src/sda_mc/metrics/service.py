from __future__ import annotations

from typing import Callable

from sda_mc.core.types import Trajectory

from .base import MetricSpec
from .risk import Tail


def total_reward[StateT, DecisionT, ExogenousT](
    name: str = "total_reward",
    *,
    tail: Tail | None = "lower",
    tail_alpha: float = 0.95,
) -> MetricSpec[StateT, DecisionT, ExogenousT]:
    return MetricSpec(
        name=name,
        fn=lambda trajectory: trajectory.total_reward,
        higher_is_better=True,
        tail=tail,
        tail_alpha=tail_alpha,
    )


def service_metric[StateT, DecisionT, ExogenousT](
    name: str,
    fn: Callable[[Trajectory[StateT, DecisionT, ExogenousT]], float],
    *,
    tail: Tail | None = None,
    tail_alpha: float = 0.95,
) -> MetricSpec[StateT, DecisionT, ExogenousT]:
    return MetricSpec(
        name=name,
        fn=fn,
        higher_is_better=True,
        tail=tail,
        tail_alpha=tail_alpha,
    )
