from __future__ import annotations

from typing import Callable

from sda_mc.core.types import Trajectory

from .base import MetricSpec
from .risk import Tail


def total_reward_cost[StateT, DecisionT, ExogenousT](
    name: str = "total_cost",
    *,
    transform: Callable[[float], float] = lambda reward: -reward,
    tail: Tail | None = "upper",
    tail_alpha: float = 0.95,
) -> MetricSpec[StateT, DecisionT, ExogenousT]:
    """Build a cost metric from trajectory total reward."""

    return MetricSpec(
        name=name,
        fn=lambda trajectory: transform(trajectory.total_reward),
        higher_is_better=False,
        tail=tail,
        tail_alpha=tail_alpha,
    )


def sum_step_cost[StateT, DecisionT, ExogenousT](
    name: str,
    fn: Callable[[Trajectory[StateT, DecisionT, ExogenousT]], float],
    *,
    tail: Tail | None = "upper",
    tail_alpha: float = 0.95,
) -> MetricSpec[StateT, DecisionT, ExogenousT]:
    return MetricSpec(
        name=name,
        fn=fn,
        higher_is_better=False,
        tail=tail,
        tail_alpha=tail_alpha,
    )
