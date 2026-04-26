from __future__ import annotations

from typing import Callable

from sda_mc.core.types import Trajectory

from .base import MetricSpec


def violation_count[StateT, DecisionT, ExogenousT](
    name: str,
    fn: Callable[[Trajectory[StateT, DecisionT, ExogenousT]], float],
) -> MetricSpec[StateT, DecisionT, ExogenousT]:
    return MetricSpec(name=name, fn=fn, higher_is_better=False, tail="upper")


def feasibility_rate[StateT, DecisionT, ExogenousT](
    name: str,
    fn: Callable[[Trajectory[StateT, DecisionT, ExogenousT]], float],
) -> MetricSpec[StateT, DecisionT, ExogenousT]:
    return MetricSpec(name=name, fn=fn, higher_is_better=True, tail="lower")
