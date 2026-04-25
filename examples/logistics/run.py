from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt
from pathlib import Path
from statistics import mean, stdev
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sda_mc import HistoricalBootstrapSampler, Simulator, SimulatorConfig  # noqa: E402

from data import initial_state, synthetic_history  # noqa: E402
from policies import GreedyPolicy, MilpPolicy, PriorityPolicy, RandomPolicy  # noqa: E402
from transition import logistics_transition, reward_completed_minus_late, reward_components  # noqa: E402


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    std: float
    ci95_low: float
    ci95_high: float


def summarize(values: list[float]) -> MetricSummary:
    if not values:
        raise ValueError("no values to summarize")
    mu = mean(values)
    sd = stdev(values) if len(values) > 1 else 0.0
    se = sd / sqrt(len(values)) if len(values) > 1 else 0.0
    return MetricSummary(mu, sd, mu - 1.96 * se, mu + 1.96 * se)


def cvar_cost(values: list[float], alpha: float = 0.95) -> float:
    if not values:
        raise ValueError("no values to summarize")
    tail_size = max(1, ceil((1 - alpha) * len(values)))
    worst_costs = sorted(values, reverse=True)[:tail_size]
    return mean(worst_costs)


def trajectory_metrics(trajectories) -> dict[str, list[float]]:
    metrics = {
        "service_value": [],
        "late_cost": [],
        "net_reward": [],
        "delivered_orders": [],
        "final_late_orders": [],
    }
    for trajectory in trajectories:
        service_value = 0.0
        late_cost = 0.0
        delivered_orders = 0
        for step in trajectory.steps:
            components = reward_components(step.state, step.next_state)
            service_value += components.service_value
            late_cost += components.late_penalty
            delivered_orders += len(step.next_state.completed_orders) - len(
                step.state.completed_orders
            )

        final_state = trajectory.final_state
        final_late_orders = sum(
            1 for order in final_state.pending_orders if final_state.time > order.deadline
        )
        metrics["service_value"].append(service_value)
        metrics["late_cost"].append(late_cost)
        metrics["net_reward"].append(service_value - late_cost)
        metrics["delivered_orders"].append(float(delivered_orders))
        metrics["final_late_orders"].append(float(final_late_orders))
    return metrics


def main() -> None:
    history = synthetic_history(days=365)
    config = SimulatorConfig(horizon=30, replications=100)

    print(
        f"{'policy':24s} "
        f"{'service_mean':>12s} "
        f"{'late_cost_mean':>14s} "
        f"{'net_mean':>10s} "
        f"{'net_ci95':>23s} "
        f"{'cost_cvar95':>12s} "
        f"{'deliv_mean':>11s} "
        f"{'late_final':>10s}"
    )

    for policy in [RandomPolicy(seed=1), GreedyPolicy(), PriorityPolicy(), MilpPolicy()]:
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
        metrics = trajectory_metrics(trajectories)
        net = summarize(metrics["net_reward"])
        service = summarize(metrics["service_value"])
        late_cost = summarize(metrics["late_cost"])
        delivered = summarize(metrics["delivered_orders"])
        final_late = summarize(metrics["final_late_orders"])
        print(
            f"{policy.name:24s} "
            f"{service.mean:12.2f} "
            f"{late_cost.mean:14.2f} "
            f"{net.mean:10.2f} "
            f"({net.ci95_low:8.2f}, {net.ci95_high:8.2f}) "
            f"{cvar_cost(metrics['late_cost']):12.2f} "
            f"{delivered.mean:11.2f} "
            f"{final_late.mean:10.2f}"
        )


if __name__ == "__main__":
    main()
