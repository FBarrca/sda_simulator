from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
from typing import Any

from sda_mc.core.simulator import SimulatorConfig

from .base import ExperimentMetadata, MetricReport


def metadata_from_config(
    *,
    name: str,
    config: SimulatorConfig,
    policy_name: str | None = None,
    seed: int | None = None,
    tags: tuple[str, ...] = (),
    extra: dict[str, Any] | None = None,
) -> ExperimentMetadata:
    return ExperimentMetadata(
        name=name,
        policy_name=policy_name,
        horizon=config.horizon,
        replications=config.replications,
        seed=seed,
        tags=tags,
        extra=extra or {},
    )


def report_to_dict(report: MetricReport) -> dict[str, Any]:
    return {
        "metadata": asdict(report.metadata) if report.metadata is not None else None,
        "sample_paths": [asdict(sample) for sample in report.sample_paths],
        "aggregates": {
            name: asdict(summary) for name, summary in report.aggregates.items()
        },
        "tail_risk": {name: asdict(risk) for name, risk in report.tail_risk.items()},
    }


def load_env_file(path: str | Path | None = None, *, override: bool = False) -> Path | None:
    """Load simple KEY=VALUE settings from a .env file into os.environ."""

    env_path = Path(path) if path is not None else _find_env_file()
    if env_path is None or not env_path.exists():
        return None

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _clean_env_value(value.strip())
        if key and (override or key not in os.environ):
            os.environ[key] = value
    return env_path


def _find_env_file() -> Path | None:
    for parent in [Path.cwd(), *Path.cwd().parents]:
        candidate = parent / ".env"
        if candidate.exists():
            return candidate
    return None


def _clean_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def env_flag(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def wandb_enabled() -> bool:
    return env_flag("SDA_MC_WANDB")


def wandb_verbose() -> bool:
    return env_flag("SDA_MC_WANDB_VERBOSE")


def configure_wandb_console() -> dict[str, Any]:
    """Return W&B settings and environment defaults for quiet console output."""

    if wandb_verbose():
        return {}

    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("WANDB_QUIET", "true")
    os.environ.setdefault("WANDB_CONSOLE", "off")
    return {"silent": True, "console": "off"}


def default_run_name(metadata: ExperimentMetadata | None) -> str | None:
    if metadata is None:
        return None
    if metadata.policy_name:
        return f"{metadata.name}:{metadata.policy_name}"
    return metadata.name


def log_wandb_report(
    report: MetricReport,
    *,
    project: str | None = None,
    entity: str | None = None,
    mode: str | None = None,
    run_name: str | None = None,
    group: str | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """Log aggregate and tail-risk metrics to Weights & Biases when installed."""

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "Weights & Biases tracking requires installing "
            "'sda-mc-simulator[tracking]'"
        ) from exc

    metadata = report.metadata
    run_config: dict[str, Any] = {}
    if metadata is not None:
        run_config.update(asdict(metadata))
    if config:
        run_config.update(config)
    settings = configure_wandb_console()

    with wandb.init(
        project=project or os.getenv("SDA_MC_WANDB_PROJECT") or None,
        entity=entity or os.getenv("SDA_MC_WANDB_ENTITY") or None,
        name=run_name or default_run_name(metadata),
        group=group or (metadata.name if metadata is not None else None),
        config=run_config or None,
        mode=mode or os.getenv("SDA_MC_WANDB_MODE") or None,
        settings=settings or None,
    ):
        payload: dict[str, float] = {}
        for name, summary in report.aggregates.items():
            payload[f"{name}/mean"] = summary.mean
            payload[f"{name}/std"] = summary.std
            payload[f"{name}/ci95_low"] = summary.ci95_low
            payload[f"{name}/ci95_high"] = summary.ci95_high
        for name, risk in report.tail_risk.items():
            payload[f"{name}/var{int(risk.alpha * 100)}"] = risk.var
            payload[f"{name}/cvar{int(risk.alpha * 100)}"] = risk.cvar
        wandb.log(payload)
