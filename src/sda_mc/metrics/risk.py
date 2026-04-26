from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from statistics import mean
from typing import Literal, Sequence


Tail = Literal["upper", "lower"]


@dataclass(frozen=True)
class TailRisk:
    """Tail-risk metrics for a scalar sample-path KPI."""

    alpha: float
    tail: Tail
    var: float
    cvar: float


def _validate(values: Sequence[float], alpha: float) -> list[float]:
    if not values:
        raise ValueError("no values to summarize")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    return [float(value) for value in values]


def value_at_risk(values: Sequence[float], alpha: float = 0.95, *, tail: Tail = "upper") -> float:
    """Return empirical VaR for an upper cost tail or lower reward tail."""

    numeric = sorted(_validate(values, alpha))
    if tail == "upper":
        index = min(len(numeric) - 1, max(0, ceil(alpha * len(numeric)) - 1))
        return numeric[index]
    if tail == "lower":
        index = min(len(numeric) - 1, max(0, ceil((1.0 - alpha) * len(numeric)) - 1))
        return numeric[index]
    raise ValueError("tail must be 'upper' or 'lower'")


def cvar(values: Sequence[float], alpha: float = 0.95, *, tail: Tail = "upper") -> float:
    """Return empirical conditional VaR over the worst tail observations."""

    numeric = _validate(values, alpha)
    tail_size = max(1, ceil((1.0 - alpha) * len(numeric)))
    if tail == "upper":
        tail_values = sorted(numeric, reverse=True)[:tail_size]
    elif tail == "lower":
        tail_values = sorted(numeric)[:tail_size]
    else:
        raise ValueError("tail must be 'upper' or 'lower'")
    return mean(tail_values)


def tail_risk(values: Sequence[float], alpha: float = 0.95, *, tail: Tail = "upper") -> TailRisk:
    return TailRisk(
        alpha=alpha,
        tail=tail,
        var=value_at_risk(values, alpha, tail=tail),
        cvar=cvar(values, alpha, tail=tail),
    )
