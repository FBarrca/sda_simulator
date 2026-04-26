from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean, stdev
from typing import Sequence


@dataclass(frozen=True)
class StatisticalSummary:
    """Replication-level summary for one scalar metric."""

    n: int
    mean: float
    std: float
    stderr: float
    ci95_low: float
    ci95_high: float
    min: float
    max: float
    sum: float


def summarize(values: Sequence[float]) -> StatisticalSummary:
    """Summarize scalar values with a normal-approximation 95% interval."""

    if not values:
        raise ValueError("no values to summarize")
    numeric = [float(value) for value in values]
    mu = mean(numeric)
    sd = stdev(numeric) if len(numeric) > 1 else 0.0
    se = sd / sqrt(len(numeric)) if len(numeric) > 1 else 0.0
    return StatisticalSummary(
        n=len(numeric),
        mean=mu,
        std=sd,
        stderr=se,
        ci95_low=mu - 1.96 * se,
        ci95_high=mu + 1.96 * se,
        min=min(numeric),
        max=max(numeric),
        sum=sum(numeric),
    )
