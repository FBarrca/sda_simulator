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

    @property
    def ci_half_width(self) -> float:
        """Half-width of the 95% confidence interval (mean ± ci_half_width)."""
        return (self.ci95_high - self.ci95_low) / 2.0


def format_ci(summary: StatisticalSummary, decimals: int = 2) -> str:
    """Format a summary as 'mean ± half_width  (n=N)' for display."""
    fmt = f".{decimals}f"
    return f"{summary.mean:{fmt}} ± {summary.ci_half_width:{fmt}}  (n={summary.n})"


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
