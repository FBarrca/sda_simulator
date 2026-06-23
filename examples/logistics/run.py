from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sda_mc import (  # noqa: E402
    HistoricalBootstrapSampler,
    MetricSpec,
    PolicyProgress,
    Simulator,
    SimulatorConfig,
    TableColumn,
    evaluate_metrics,
    metadata_from_config,
    render_metric_table,
)
from sda_mc.metrics.runtime import load_env_file, log_wandb_report, wandb_enabled  # noqa: E402

from data import initial_state, synthetic_history  # noqa: E402
from policies import GreedyPolicy, MilpPolicy, PriorityPolicy, RandomPolicy  # noqa: E402
from transition import logistics_transition, reward_completed_minus_late, reward_components  # noqa: E402


def service_value(trajectory) -> float:
    return sum(
        reward_components(step.state, step.decision, step.exogenous).service_value
        for step in trajectory.steps
    )


def late_cost(trajectory) -> float:
    return sum(
        reward_components(step.state, step.decision, step.exogenous).late_penalty
        for step in trajectory.steps
    )


def delivered_orders(trajectory) -> float:
    return float(
        sum(
            len(step.next_state.completed_orders) - len(step.state.completed_orders)
            for step in trajectory.steps
        )
    )


def final_late_orders(trajectory) -> float:
    final_state = trajectory.final_state
    return float(
        sum(1 for order in final_state.pending_orders if final_state.time > order.deadline)
    )


LOGISTICS_METRICS = [
    MetricSpec("service_value", service_value, higher_is_better=True),
    MetricSpec("late_cost", late_cost, higher_is_better=False, tail="upper"),
    MetricSpec("net_reward", lambda trajectory: trajectory.total_reward, higher_is_better=True),
    MetricSpec("delivered_orders", delivered_orders, higher_is_better=True),
    MetricSpec("final_late_orders", final_late_orders, higher_is_better=False),
]


def main() -> None:
    load_env_file(ROOT / ".env")
    history = synthetic_history(days=365)
    config = SimulatorConfig(horizon=30, replications=100)
    policies = [RandomPolicy(seed=1), GreedyPolicy(), PriorityPolicy(), MilpPolicy()]
    rows: list[list[str]] = []

    with PolicyProgress([policy.name for policy in policies]) as progress:
        for policy in policies:
            progress.start_policy(policy.name)
            sampler = HistoricalBootstrapSampler(history, block_size=7, seed=42)
            simulator = Simulator(
                transition=logistics_transition,
                sampler=sampler,
                reward_fn=reward_completed_minus_late,
            )
            trajectories = simulator.run(
                initial_state=lambda _r: initial_state(),
                policy=policy,
                config=config,
            )
            report = evaluate_metrics(
                trajectories,
                LOGISTICS_METRICS,
                metadata=metadata_from_config(
                    name="logistics",
                    config=config,
                    policy_name=policy.name,
                    seed=42,
                    extra={"domain": "logistics"},
                ),
            )
            if wandb_enabled():
                log_wandb_report(report)

            net = report.aggregates["net_reward"]
            service = report.aggregates["service_value"]
            cost = report.aggregates["late_cost"]
            delivered = report.aggregates["delivered_orders"]
            final_late = report.aggregates["final_late_orders"]
            rows.append(
                [
                    policy.name,
                    f"{service.mean:.2f}",
                    f"{cost.mean:.2f}",
                    f"{net.mean:.2f}",
                    f"({net.ci95_low:.2f}, {net.ci95_high:.2f})",
                    f"{report.tail_risk['late_cost'].cvar:.2f}",
                    f"{delivered.mean:.2f}",
                    f"{final_late.mean:.2f}",
                ]
            )
            progress.finish_policy(policy.name)

    render_metric_table(
        title="Logistics policy metrics",
        columns=[
            TableColumn("policy", justify="left"),
            TableColumn("service_mean"),
            TableColumn("late_cost_mean"),
            TableColumn("net_mean"),
            TableColumn("net_ci95"),
            TableColumn("cost_cvar95"),
            TableColumn("deliv_mean"),
            TableColumn("late_final"),
        ],
        rows=rows,
    )


if __name__ == "__main__":
    main()
