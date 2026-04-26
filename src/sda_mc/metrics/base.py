from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from sda_mc.core.types import Trajectory

from .risk import Tail, TailRisk, tail_risk
from .statistical import StatisticalSummary, summarize


type MetricFn[StateT, DecisionT, ExogenousT] = Callable[
    [Trajectory[StateT, DecisionT, ExogenousT]], float
]


@dataclass(frozen=True)
class ExperimentMetadata:
    """Metadata that identifies and describes one simulation experiment."""

    name: str
    policy_name: str | None = None
    horizon: int | None = None
    replications: int | None = None
    seed: int | None = None
    tags: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricSpec[StateT, DecisionT, ExogenousT]:
    """Defines one scalar metric computed for each trajectory."""

    name: str
    fn: MetricFn[StateT, DecisionT, ExogenousT]
    description: str = ""
    unit: str | None = None
    higher_is_better: bool | None = None
    tail: Tail | None = None
    tail_alpha: float = 0.95

    def evaluate(self, trajectory: Trajectory[StateT, DecisionT, ExogenousT]) -> float:
        return float(self.fn(trajectory))


@dataclass(frozen=True)
class SampleMetricResult:
    replication: int
    policy_name: str
    values: dict[str, float]


@dataclass(frozen=True)
class MetricReport:
    metadata: ExperimentMetadata | None
    sample_paths: list[SampleMetricResult]
    aggregates: dict[str, StatisticalSummary]
    tail_risk: dict[str, TailRisk]

    def values(self, metric_name: str) -> list[float]:
        try:
            return [sample.values[metric_name] for sample in self.sample_paths]
        except KeyError as exc:
            raise KeyError(f"unknown metric: {metric_name}") from exc


class MetricSet[StateT, DecisionT, ExogenousT]:
    """Ordered collection of named metric specs."""

    def __init__(
        self,
        specs: Iterable[MetricSpec[StateT, DecisionT, ExogenousT]] = (),
    ) -> None:
        self._specs: list[MetricSpec[StateT, DecisionT, ExogenousT]] = []
        self._names: set[str] = set()
        for spec in specs:
            self.add(spec)

    def add(self, spec: MetricSpec[StateT, DecisionT, ExogenousT]) -> None:
        if spec.name in self._names:
            raise ValueError(f"duplicate metric name: {spec.name}")
        self._specs.append(spec)
        self._names.add(spec.name)

    def __iter__(self):
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)


def evaluate_metrics[StateT, DecisionT, ExogenousT](
    trajectories: Iterable[Trajectory[StateT, DecisionT, ExogenousT]],
    metric_specs: Iterable[MetricSpec[StateT, DecisionT, ExogenousT]],
    *,
    metadata: ExperimentMetadata | None = None,
) -> MetricReport:
    """Evaluate metrics per trajectory and aggregate them across replications."""

    trajectory_list = list(trajectories)
    if not trajectory_list:
        raise ValueError("no trajectories to evaluate")

    specs = MetricSet(metric_specs)
    if len(specs) == 0:
        raise ValueError("no metric specs to evaluate")

    sample_paths = [
        SampleMetricResult(
            replication=trajectory.replication,
            policy_name=trajectory.policy_name,
            values={spec.name: spec.evaluate(trajectory) for spec in specs},
        )
        for trajectory in trajectory_list
    ]
    aggregates = {
        spec.name: summarize([sample.values[spec.name] for sample in sample_paths])
        for spec in specs
    }
    risks = {
        spec.name: tail_risk(
            [sample.values[spec.name] for sample in sample_paths],
            spec.tail_alpha,
            tail=spec.tail,
        )
        for spec in specs
        if spec.tail is not None
    }
    return MetricReport(
        metadata=metadata,
        sample_paths=sample_paths,
        aggregates=aggregates,
        tail_risk=risks,
    )
