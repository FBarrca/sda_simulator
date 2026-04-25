from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean, stdev

from .types import Trajectory


@dataclass(frozen=True)
class Summary:
    n: int
    mean: float
    std: float
    stderr: float
    ci95_low: float
    ci95_high: float


def summarize_rewards(trajectories: list[Trajectory]) -> Summary:
    values = [tr.total_reward for tr in trajectories]
    if not values:
        raise ValueError("no trajectories to summarize")
    mu = mean(values)
    sd = stdev(values) if len(values) > 1 else 0.0
    se = sd / sqrt(len(values)) if len(values) > 1 else 0.0
    return Summary(
        n=len(values),
        mean=mu,
        std=sd,
        stderr=se,
        ci95_low=mu - 1.96 * se,
        ci95_high=mu + 1.96 * se,
    )
