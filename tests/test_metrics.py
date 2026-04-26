from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from types import SimpleNamespace

import pytest

from sda_mc.core.types import Trajectory
from sda_mc.metrics import (
    ExperimentMetadata,
    MetricRegistry,
    MetricSpec,
    configure_wandb_console,
    cvar,
    default_run_name,
    env_flag,
    evaluate_metrics,
    load_env_file,
    log_wandb_report,
    summarize,
    value_at_risk,
    wandb_verbose,
)


@dataclass(frozen=True)
class State:
    value: float


def _trajectory(replication: int, total_reward: float) -> Trajectory[State, None, None]:
    return Trajectory(
        replication=replication,
        policy_name="test-policy",
        steps=[],
        total_reward=total_reward,
        final_state=State(total_reward * 2.0),
    )


def test_summarize_matches_normal_ci_behavior():
    summary = summarize([1.0, 2.0, 3.0])

    assert summary.n == 3
    assert summary.mean == pytest.approx(2.0)
    assert summary.std == pytest.approx(1.0)
    assert summary.stderr == pytest.approx(1.0 / (3.0**0.5))
    assert summary.ci95_low == pytest.approx(2.0 - 1.96 / (3.0**0.5))
    assert summary.ci95_high == pytest.approx(2.0 + 1.96 / (3.0**0.5))


def test_cvar_upper_tail_matches_cost_convention():
    values = [1.0, 10.0, 3.0, 20.0, 5.0]

    assert cvar(values, alpha=0.95, tail="upper") == pytest.approx(20.0)
    assert value_at_risk(values, alpha=0.95, tail="upper") == pytest.approx(20.0)


def test_cvar_lower_tail_supports_reward_downside():
    values = [1.0, 10.0, 3.0, 20.0, 5.0]

    assert cvar(values, alpha=0.95, tail="lower") == pytest.approx(1.0)
    assert value_at_risk(values, alpha=0.95, tail="lower") == pytest.approx(1.0)


def test_registry_rejects_duplicate_metric_names():
    metric = MetricSpec("reward", lambda trajectory: trajectory.total_reward)
    registry = MetricRegistry([metric])

    with pytest.raises(ValueError, match="duplicate metric name: reward"):
        registry.register(metric)


def test_evaluate_metrics_returns_sample_paths_aggregates_and_tail_risk():
    trajectories = [_trajectory(0, 1.0), _trajectory(1, 3.0), _trajectory(2, 5.0)]

    report = evaluate_metrics(
        trajectories,
        [
            MetricSpec("reward", lambda trajectory: trajectory.total_reward, tail="lower"),
            MetricSpec("final_value", lambda trajectory: trajectory.final_state.value),
        ],
        metadata=ExperimentMetadata(name="unit-test", replications=3),
    )

    assert report.metadata == ExperimentMetadata(name="unit-test", replications=3)
    assert [sample.replication for sample in report.sample_paths] == [0, 1, 2]
    assert report.values("reward") == [1.0, 3.0, 5.0]
    assert report.aggregates["reward"].mean == pytest.approx(3.0)
    assert report.aggregates["final_value"].mean == pytest.approx(6.0)
    assert report.tail_risk["reward"].cvar == pytest.approx(1.0)


def test_evaluate_metrics_rejects_empty_inputs():
    with pytest.raises(ValueError, match="no trajectories to evaluate"):
        evaluate_metrics([], [MetricSpec("reward", lambda trajectory: trajectory.total_reward)])

    with pytest.raises(ValueError, match="no metric specs to evaluate"):
        evaluate_metrics([_trajectory(0, 1.0)], [])


def test_default_run_name_separates_policy_runs():
    assert (
        default_run_name(ExperimentMetadata(name="logistics", policy_name="greedy"))
        == "logistics:greedy"
    )
    assert default_run_name(ExperimentMetadata(name="logistics")) == "logistics"
    assert default_run_name(None) is None


def test_load_env_file_sets_wandb_environment(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "SDA_MC_WANDB=1",
                "SDA_MC_WANDB_PROJECT='demo-project'",
                "SDA_MC_WANDB_MODE=offline",
            ]
        )
    )
    monkeypatch.delenv("SDA_MC_WANDB", raising=False)
    monkeypatch.delenv("SDA_MC_WANDB_PROJECT", raising=False)
    monkeypatch.delenv("SDA_MC_WANDB_MODE", raising=False)

    loaded = load_env_file(env_file)

    assert loaded == env_file
    assert env_flag("SDA_MC_WANDB") is True
    assert env_flag("SDA_MC_MISSING_FLAG") is False
    assert env_flag("SDA_MC_MISSING_FLAG", default=True) is True
    assert env_file.exists()
    assert env_file.read_text().startswith("SDA_MC_WANDB=1")


def test_wandb_console_defaults_to_quiet(monkeypatch):
    monkeypatch.delenv("SDA_MC_WANDB_VERBOSE", raising=False)
    monkeypatch.delenv("WANDB_SILENT", raising=False)
    monkeypatch.delenv("WANDB_QUIET", raising=False)
    monkeypatch.delenv("WANDB_CONSOLE", raising=False)

    settings = configure_wandb_console()

    assert settings == {"silent": True, "console": "off"}
    assert env_flag("WANDB_SILENT") is True
    assert env_flag("WANDB_QUIET") is True
    assert env_flag("SDA_MC_WANDB_VERBOSE") is False
    assert wandb_verbose() is False
    assert env_flag("WANDB_CONSOLE") is False


def test_wandb_verbose_leaves_console_settings_alone(monkeypatch):
    monkeypatch.setenv("SDA_MC_WANDB_VERBOSE", "1")
    monkeypatch.delenv("WANDB_SILENT", raising=False)
    monkeypatch.delenv("WANDB_QUIET", raising=False)
    monkeypatch.delenv("WANDB_CONSOLE", raising=False)

    settings = configure_wandb_console()

    assert settings == {}
    assert wandb_verbose() is True
    assert "WANDB_SILENT" not in os.environ


def test_log_wandb_report_groups_policy_runs_and_uses_quiet_settings(monkeypatch):
    captured: dict[str, object] = {}

    class FakeRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return None

    def fake_init(**kwargs):
        captured["init"] = kwargs
        return FakeRun()

    def fake_log(payload):
        captured["payload"] = payload

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=fake_init, log=fake_log),
    )
    monkeypatch.delenv("SDA_MC_WANDB_VERBOSE", raising=False)
    monkeypatch.setenv("SDA_MC_WANDB_PROJECT", "demo-project")
    report = evaluate_metrics(
        [_trajectory(0, 1.0)],
        [MetricSpec("reward", lambda trajectory: trajectory.total_reward)],
        metadata=ExperimentMetadata(name="logistics", policy_name="random"),
    )

    log_wandb_report(report)

    assert captured["init"]["project"] == "demo-project"
    assert captured["init"]["name"] == "logistics:random"
    assert captured["init"]["group"] == "logistics"
    assert captured["init"]["settings"] == {"silent": True, "console": "off"}
    assert captured["payload"]["reward/mean"] == pytest.approx(1.0)
