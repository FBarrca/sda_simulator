from __future__ import annotations

from statistics import mean as _mean, stdev as _stdev
from typing import Callable, Sequence

from .statistical import StatisticalSummary, summarize


type Reducer = Callable[[Sequence[float]], float]


def require_values(values: Sequence[float]) -> list[float]:
    if not values:
        raise ValueError("no values to summarize")
    return [float(value) for value in values]


def count(values: Sequence[float]) -> float:
    return float(len(require_values(values)))


def mean(values: Sequence[float]) -> float:
    return _mean(require_values(values))


def std(values: Sequence[float]) -> float:
    numeric = require_values(values)
    return _stdev(numeric) if len(numeric) > 1 else 0.0


def stderr(values: Sequence[float]) -> float:
    numeric = require_values(values)
    return std(numeric) / (len(numeric) ** 0.5) if len(numeric) > 1 else 0.0


def minimum(values: Sequence[float]) -> float:
    return min(require_values(values))


def maximum(values: Sequence[float]) -> float:
    return max(require_values(values))


def total(values: Sequence[float]) -> float:
    return sum(require_values(values))


def ci95_low(values: Sequence[float]) -> float:
    return summarize(values).ci95_low


def ci95_high(values: Sequence[float]) -> float:
    return summarize(values).ci95_high


def standard_summary(values: Sequence[float]) -> StatisticalSummary:
    return summarize(values)
