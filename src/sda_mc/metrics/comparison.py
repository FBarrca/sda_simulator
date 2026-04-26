from __future__ import annotations

from dataclasses import dataclass

from .base import MetricReport


@dataclass(frozen=True)
class MetricComparison:
    metric_name: str
    baseline_mean: float
    candidate_mean: float
    delta: float
    percent_delta: float | None


def compare_reports(
    baseline: MetricReport,
    candidate: MetricReport,
    metric_names: list[str] | None = None,
) -> list[MetricComparison]:
    names = metric_names or list(baseline.aggregates)
    comparisons: list[MetricComparison] = []
    for name in names:
        baseline_mean = baseline.aggregates[name].mean
        candidate_mean = candidate.aggregates[name].mean
        delta = candidate_mean - baseline_mean
        percent_delta = None if baseline_mean == 0 else delta / abs(baseline_mean)
        comparisons.append(
            MetricComparison(
                metric_name=name,
                baseline_mean=baseline_mean,
                candidate_mean=candidate_mean,
                delta=delta,
                percent_delta=percent_delta,
            )
        )
    return comparisons
