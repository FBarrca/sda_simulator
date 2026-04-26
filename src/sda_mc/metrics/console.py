from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import TracebackType
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from .base import MetricReport


@dataclass(frozen=True)
class TableColumn:
    name: str
    justify: str = "right"
    no_wrap: bool = True


class PolicyProgress(AbstractContextManager["PolicyProgress"]):
    """Rich progress wrapper for policy-level simulation loops."""

    def __init__(self, policies: Sequence[str], *, console: Console | None = None) -> None:
        self.console = console or Console()
        self._policies = policies
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}", justify="left"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
            disable=not self.console.is_terminal,
        )
        self._task_id: int | None = None

    def __enter__(self) -> "PolicyProgress":
        self._progress.__enter__()
        self._task_id = self._progress.add_task(
            "Running policies",
            total=len(self._policies),
        )
        return self

    def start_policy(self, policy_name: str) -> None:
        self._progress.update(
            self._require_task(),
            description=f"Running {policy_name}",
        )

    def finish_policy(self, policy_name: str) -> None:
        _ = policy_name
        task_id = self._require_task()
        self._progress.advance(task_id)
        if self._progress.tasks[task_id].completed >= self._progress.tasks[task_id].total:
            self._progress.update(task_id, description="Finished policies")
            return
        self._progress.update(
            task_id,
            description=f"Finished {policy_name}",
        )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        return self._progress.__exit__(exc_type, exc_value, traceback)

    def _require_task(self) -> int:
        if self._task_id is None:
            raise RuntimeError("PolicyProgress must be entered before use")
        return self._task_id


def render_metric_table(
    *,
    title: str,
    columns: Sequence[TableColumn],
    rows: Iterable[Sequence[Any]],
    console: Console | None = None,
) -> None:
    output = console or Console(width=140)
    table = Table(title=title, show_lines=False)
    for column in columns:
        table.add_column(column.name, justify=column.justify, no_wrap=column.no_wrap)
    for row in rows:
        table.add_row(*(str(value) for value in row))
    output.print(table)


def report_metric(report: MetricReport, metric_name: str, field: str = "mean") -> float:
    summary = report.aggregates[metric_name]
    return float(getattr(summary, field))
