from __future__ import annotations

from dataclasses import dataclass

from .base import MetricReport


@dataclass(frozen=True)
class MetricComparison:
    metric_name: str
    baseline_mean: float
    candidate_mean: float
    baseline_ci_hw: float
    candidate_ci_hw: float
    delta: float
    percent_delta: float | None
    significant: bool
    """True when the 95% CIs do not overlap — a conservative significance signal."""


def compare_reports(
    baseline: MetricReport,
    candidate: MetricReport,
    metric_names: list[str] | None = None,
) -> list[MetricComparison]:
    names = metric_names or list(baseline.aggregates)
    comparisons: list[MetricComparison] = []
    for name in names:
        b = baseline.aggregates[name]
        c = candidate.aggregates[name]
        delta = c.mean - b.mean
        percent_delta = None if b.mean == 0 else delta / abs(b.mean)
        significant = c.ci95_low > b.ci95_high or c.ci95_high < b.ci95_low
        comparisons.append(
            MetricComparison(
                metric_name=name,
                baseline_mean=b.mean,
                candidate_mean=c.mean,
                baseline_ci_hw=b.ci_half_width,
                candidate_ci_hw=c.ci_half_width,
                delta=delta,
                percent_delta=percent_delta,
                significant=significant,
            )
        )
    return comparisons
