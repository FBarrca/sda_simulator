from __future__ import annotations

from collections.abc import Iterable

from .base import MetricSpec


class MetricRegistry[StateT, DecisionT, ExogenousT]:
    """Small ordered registry for named metric specs."""

    def __init__(
        self,
        specs: Iterable[MetricSpec[StateT, DecisionT, ExogenousT]] = (),
    ) -> None:
        self._specs: dict[str, MetricSpec[StateT, DecisionT, ExogenousT]] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: MetricSpec[StateT, DecisionT, ExogenousT]) -> None:
        if spec.name in self._specs:
            raise ValueError(f"duplicate metric name: {spec.name}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> MetricSpec[StateT, DecisionT, ExogenousT]:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"unknown metric: {name}") from exc

    def values(self) -> list[MetricSpec[StateT, DecisionT, ExogenousT]]:
        return list(self._specs.values())

    def __iter__(self):
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)
