from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sda_mc import (  # noqa: E402
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

from domain import OpenPurchaseOrder, State  # noqa: E402
from extract_policy_inputs import InventorySimulationData, load_inventory_simulation_data  # noqa: E402
from policy import (  # noqa: E402
    AggressiveReorderPolicy,
    AllocationOnlyPolicy,
    MilpReorderPolicy,
    NoOpPolicy,
    ReorderAllocatePolicy,
    ReorderExpediteAllocatePolicy,
)
from sampler import InventoryHistoricalSampler  # noqa: E402
from transition import inventory_transition, reward_stockout_overstock_service  # noqa: E402


def initial_state(data: InventorySimulationData) -> State:
    """Build the initial inventory and pipeline state from extracted DB data."""

    start_date = data.history[0].date
    inventory = {
        (position.material_id, position.plant_id): position.available_quantity
        for position in data.inventory.values()
    }
    pipeline = {}
    for order in data.purchase_orders:
        if order.expected_receipt_date is None:
            continue
        if order.order_date <= start_date <= order.expected_receipt_date:
            pipeline[order.order_id] = OpenPurchaseOrder(
                order_id=order.order_id,
                material_id=order.material_id,
                plant_id=order.plant_id,
                supplier_id=order.supplier_id,
                quantity_open=order.quantity,
                order_date=order.order_date,
                expected_receipt_date=order.expected_receipt_date,
            )
    return State(date=start_date, inventory=inventory, pipeline=pipeline)


INVENTORY_METRICS = [
    MetricSpec(
        "reward",
        lambda trajectory: trajectory.total_reward,
        higher_is_better=True,
        tail="lower",
    ),
    MetricSpec(
        "final_backlog",
        lambda trajectory: sum(
            order.quantity_open for order in trajectory.final_state.backlog.values()
        ),
        higher_is_better=False,
        tail="upper",
    ),
    MetricSpec(
        "final_inventory",
        lambda trajectory: sum(trajectory.final_state.inventory.values()),
        higher_is_better=None,
    ),
]


def main() -> None:
    """Run the inventory simulator for several policies."""

    load_env_file(ROOT / ".env")
    data = load_inventory_simulation_data()
    config = SimulatorConfig(horizon=60, replications=20, parallel=False)
    policies = [
        NoOpPolicy(),
        AllocationOnlyPolicy(),
        ReorderAllocatePolicy(),
        ReorderExpediteAllocatePolicy(),
        AggressiveReorderPolicy(),
        MilpReorderPolicy(),
    ]
    rows: list[list[str]] = []

    with PolicyProgress([policy.name for policy in policies]) as progress:
        for policy in policies:
            progress.start_policy(policy.name)
            sampler = InventoryHistoricalSampler(data.history, seed=42)
            simulator = Simulator(
                transition=inventory_transition,
                sampler=sampler,
                reward_fn=reward_stockout_overstock_service,
            )
            trajectories = simulator.run(
                initial_state=lambda _replication: initial_state(data),
                policy=policy,
                config=config,
            )
            report = evaluate_metrics(
                trajectories,
                INVENTORY_METRICS,
                metadata=metadata_from_config(
                    name="ocel-simulated-inventory",
                    config=config,
                    policy_name=policy.name,
                    seed=42,
                    extra={"domain": "ocel-simulated-inventory"},
                ),
            )
            if wandb_enabled():
                log_wandb_report(report)

            reward_summary = report.aggregates["reward"]
            backlog_summary = report.aggregates["final_backlog"]
            inventory_summary = report.aggregates["final_inventory"]
            reward_risk = report.tail_risk["reward"]  # lower tail = worst-case reward
            backlog_risk = report.tail_risk["final_backlog"]  # upper tail = worst-case backlog
            rows.append(
                [
                    policy.name,
                    f"{reward_summary.mean:.2f}",
                    f"({reward_summary.ci95_low:.2f}, {reward_summary.ci95_high:.2f})",
                    f"{reward_risk.cvar:.2f}",
                    f"{backlog_summary.mean:.2f}",
                    f"{backlog_risk.cvar:.2f}",
                    f"{inventory_summary.mean:.2f}",
                ]
            )
            progress.finish_policy(policy.name)

    render_metric_table(
        title=f"Inventory policy metrics | horizon={config.horizon}, replications={config.replications}",
        columns=[
            TableColumn("policy", justify="left"),
            TableColumn("reward_mean"),
            TableColumn("reward_ci95"),
            TableColumn("reward_cvar95"),
            TableColumn("backlog_mean"),
            TableColumn("backlog_cvar95"),
            TableColumn("inventory_mean"),
        ],
        rows=rows,
    )


if __name__ == "__main__":
    main()
