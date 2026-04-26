from __future__ import annotations

from datetime import date, timedelta

from domain import (
    AllocateStockAction,
    Decision,
    ExpediteAction,
    ReorderAction,
    State,
)


class NoOpPolicy:
    """Policy that takes no action.

    This is useful as a baseline to measure how much value the transition model
    gets from reorder, expedite, and allocation decisions.
    """

    name = "no_action"

    def decide(self, state: State) -> Decision:
        """Return an empty decision for the current state."""

        del state
        return Decision()


class AllocationOnlyPolicy:
    """Policy that only allocates existing stock to known backlog.

    It does not reorder or expedite. This isolates the value of allocation from
    replenishment decisions.
    """

    name = "allocation_only"

    def decide(self, state: State) -> Decision:
        """Allocate available inventory to backlog and do nothing else."""

        return Decision(allocations=tuple(allocate_backlog_by_priority(state)))


class ReorderAllocatePolicy:
    """Policy that allocates backlog and reorders low inventory positions."""

    name = "reorder_allocate"

    def __init__(
        self,
        reorder_point: float = 80.0,
        order_up_to: float = 140.0,
        lead_time_days: int = 7,
    ) -> None:
        """Create a reorder policy with fixed inventory thresholds."""

        self.reorder_point = reorder_point
        self.order_up_to = order_up_to
        self.lead_time_days = lead_time_days

    def decide(self, state: State) -> Decision:
        """Allocate known backlog and reorder items below the target position."""

        return Decision(
            reorders=tuple(
                reorder_low_items(
                    state,
                    reorder_point=self.reorder_point,
                    order_up_to=self.order_up_to,
                    lead_time_days=self.lead_time_days,
                )
            ),
            allocations=tuple(allocate_backlog_by_priority(state)),
        )


class ReorderExpediteAllocatePolicy:
    """Policy that reorders low stock, expedites pressured POs, and allocates."""

    name = "reorder_expedite_allocate"

    def __init__(
        self,
        reorder_point: float = 80.0,
        order_up_to: float = 140.0,
        lead_time_days: int = 7,
    ) -> None:
        """Create the baseline full-action policy."""

        self.reorder_point = reorder_point
        self.order_up_to = order_up_to
        self.lead_time_days = lead_time_days

    def decide(self, state: State) -> Decision:
        """Build reorder, expedite, and allocation actions from current state."""

        return Decision(
            reorders=tuple(
                reorder_low_items(
                    state,
                    reorder_point=self.reorder_point,
                    order_up_to=self.order_up_to,
                    lead_time_days=self.lead_time_days,
                )
            ),
            expedites=tuple(expedite_for_backlog(state)),
            allocations=tuple(allocate_backlog_by_priority(state)),
        )


class AggressiveReorderPolicy(ReorderExpediteAllocatePolicy):
    """Full-action policy with higher reorder targets."""

    name = "aggressive_reorder_expedite_allocate"

    def __init__(self) -> None:
        """Create a policy that carries more inventory to avoid stockouts."""

        super().__init__(reorder_point=130.0, order_up_to=220.0, lead_time_days=5)


def allocate_backlog_by_priority(state: State) -> list[AllocateStockAction]:
    """Allocate available inventory to known backlog by priority and age."""

    remaining_inventory = dict(state.inventory)
    actions = []
    for order in sorted(
        state.backlog.values(),
        key=lambda item: (-item.priority, item.due_date or "", item.arrival_date, item.order_id),
    ):
        item_key = (order.material_id, order.plant_id)
        available = remaining_inventory.get(item_key, 0.0)
        quantity = min(order.quantity_open, available)
        if quantity <= 0:
            continue
        actions.append(AllocateStockAction(order_id=order.order_id, quantity=quantity))
        remaining_inventory[item_key] = available - quantity
    return actions


def expedite_for_backlog(state: State) -> list[ExpediteAction]:
    """Expedite open purchase orders for items that currently have backlog."""

    backlog_items = {(order.material_id, order.plant_id) for order in state.backlog.values()}
    actions = []
    for order in state.pipeline.values():
        if order.expedited:
            continue
        if (order.material_id, order.plant_id) not in backlog_items:
            continue
        actions.append(
            ExpediteAction(
                order_id=order.order_id,
                new_expected_receipt_date=add_days(state.date, 1),
                expedite_cost=max(1.0, 0.03 * order.quantity_open),
            )
        )
    return actions


def reorder_low_items(
    state: State,
    *,
    reorder_point: float,
    order_up_to: float,
    lead_time_days: int,
) -> list[ReorderAction]:
    """Reorder each item whose inventory position is below `reorder_point`."""

    item_keys = set(state.inventory) | {
        (order.material_id, order.plant_id) for order in state.backlog.values()
    }
    pipeline_by_item: dict[tuple[str, str], float] = {}
    for order in state.pipeline.values():
        item_key = (order.material_id, order.plant_id)
        pipeline_by_item[item_key] = pipeline_by_item.get(item_key, 0.0) + order.quantity_open

    actions = []
    for material_id, plant_id in sorted(item_keys):
        available = state.inventory.get((material_id, plant_id), 0.0)
        pipeline = pipeline_by_item.get((material_id, plant_id), 0.0)
        inventory_position = available + pipeline
        if inventory_position >= reorder_point:
            continue
        quantity = order_up_to - inventory_position
        actions.append(
            ReorderAction(
                order_id=f"SIM-PO-{state.time}-{material_id}-{plant_id}",
                material_id=material_id,
                plant_id=plant_id,
                supplier_id="SIM-SUPPLIER",
                quantity=quantity,
                expected_receipt_date=add_days(state.date, lead_time_days),
            )
        )
    return actions


def add_days(date_text: str, days: int) -> str:
    """Return an ISO date string offset by `days`."""

    return (date.fromisoformat(date_text) + timedelta(days=days)).isoformat()
