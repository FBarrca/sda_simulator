from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from domain import (
    AllocateStockAction,
    Decision,
    ExpediteAction,
    ReorderAction,
    State,
)

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp



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


class MilpReorderPolicy:
    """Allocate + expedite, with reorders chosen by a budget-constrained MILP.

    Allocation and expediting reuse the shared greedy helpers; the distinguishing
    piece is the reorder choice. Unlike the (s, S) policies, which reorder each
    item independently up to a fixed target, this policy spends a single
    per-epoch **reorder budget** (a working-capital / receiving-dock cap) across
    items that compete for it. A 0/1 knapsack maximizes the priority- and
    urgency-weighted known demand it can protect, subject to the budget and to
    each item's uncovered shortfall.

    It is *reactive*: it reorders only to cover demand already in backlog, not to
    build proactive safety stock. If scipy is unavailable or the program is
    infeasible, it falls back to `ReorderExpediteAllocatePolicy`.
    """

    name = "milp_reorder_budget"

    def __init__(
        self,
        reorder_budget: float = 1500.0,
        lead_time_days: int = 7,
        urgency_weight: float = 0.5,
        time_limit_seconds: float = 0.25,
        mip_rel_gap: float = 0.01,
    ) -> None:
        """Create the MILP reorder policy with a fixed per-epoch order budget."""

        self.reorder_budget = reorder_budget
        self.lead_time_days = lead_time_days
        self.urgency_weight = urgency_weight
        self.time_limit_seconds = time_limit_seconds
        self.mip_rel_gap = mip_rel_gap
        self._fallback = ReorderExpediteAllocatePolicy(lead_time_days=lead_time_days)

    def decide(self, state: State) -> Decision:
        """Allocate, expedite, and reorder the budget-optimal demand to protect."""

        reorders = self._milp_reorders(state)
        if reorders is None:
            return self._fallback.decide(state)
        return Decision(
            reorders=tuple(reorders),
            expedites=tuple(expedite_for_backlog(state)),
            allocations=tuple(allocate_backlog_by_priority(state)),
        )

    def _milp_reorders(self, state: State) -> list[ReorderAction] | None:
        """Solve the knapsack and return reorder actions, or None to fall back."""

        candidates = _uncovered_shortfall_candidates(state, self.urgency_weight, self.lead_time_days)
        if not candidates:
            return []

        items = sorted({candidate.item_key for candidate in candidates})
        item_index = {item_key: index for index, item_key in enumerate(items)}

        objective = np.array([-candidate.value for candidate in candidates], dtype=float)
        weights = np.array([candidate.quantity for candidate in candidates], dtype=float)

        # Row 0: total reorder units must stay within the budget (the coupling
        # constraint). Rows 1..N: per-item reorder must not exceed that item's
        # uncovered shortfall.
        rows = [weights.tolist()]
        upper_bounds = [self.reorder_budget]
        shortfall_by_item: dict[tuple[str, str], float] = {}
        for candidate in candidates:
            shortfall_by_item[candidate.item_key] = candidate.item_shortfall
        for item_key in items:
            rows.append(
                [
                    candidate.quantity if candidate.item_key == item_key else 0.0
                    for candidate in candidates
                ]
            )
            upper_bounds.append(shortfall_by_item[item_key])

        constraints = LinearConstraint(
            np.array(rows, dtype=float),
            lb=np.full(len(rows), -np.inf),
            ub=np.array(upper_bounds, dtype=float),
        )
        result = milp(
            c=objective,
            integrality=np.ones(len(candidates)),
            bounds=Bounds(0, 1),
            constraints=constraints,
            options={"time_limit": self.time_limit_seconds, "mip_rel_gap": self.mip_rel_gap},
        )
        if result.x is None or not result.success:
            return None

        reorder_quantity_by_item: dict[tuple[str, str], float] = {}
        for selected, candidate in zip(result.x, candidates, strict=True):
            if selected < 0.5:
                continue
            reorder_quantity_by_item[candidate.item_key] = (
                reorder_quantity_by_item.get(candidate.item_key, 0.0) + candidate.quantity
            )

        del item_index  # items are addressed by key; index kept only for clarity above
        return [
            ReorderAction(
                order_id=f"SIM-PO-{state.time}-{material_id}-{plant_id}",
                material_id=material_id,
                plant_id=plant_id,
                supplier_id="SIM-SUPPLIER",
                quantity=quantity,
                expected_receipt_date=add_days(state.date, self.lead_time_days),
            )
            for (material_id, plant_id), quantity in sorted(reorder_quantity_by_item.items())
            if quantity > 0
        ]


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
        # Only controllable, simulator-created reorders respond to expediting;
        # historical POs are received on the replayed goods-receipt schedule, so
        # expediting them would charge a cost without pulling the receipt earlier.
        if not order.order_id.startswith("SIM-PO-"):
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


@dataclass(frozen=True)
class _ReorderCandidate:
    """The uncovered part of one backlog order, eligible for budget protection."""

    item_key: tuple[str, str]
    quantity: float  # uncovered units a reorder would protect (<= item shortfall)
    value: float  # priority- and urgency-weighted worth of protecting them
    item_shortfall: float  # total uncovered demand for this item across all orders


def _uncovered_shortfall_candidates(
    state: State, urgency_weight: float, lead_time_days: int
) -> list[_ReorderCandidate]:
    """Per-order uncovered demand that on-hand stock plus pipeline cannot meet.

    Within each item, existing coverage (on-hand + open pipeline) is consumed by
    the most important orders first, mirroring how greedy allocation and arriving
    POs will be spent. Each order's remaining *uncovered* quantity becomes a
    knapsack candidate weighted by priority and urgency.
    """

    pipeline_by_item: dict[tuple[str, str], float] = {}
    for order in state.pipeline.values():
        item_key = (order.material_id, order.plant_id)
        pipeline_by_item[item_key] = pipeline_by_item.get(item_key, 0.0) + order.quantity_open

    backlog_by_item: dict[tuple[str, str], list] = {}
    for order in state.backlog.values():
        backlog_by_item.setdefault((order.material_id, order.plant_id), []).append(order)

    candidates: list[_ReorderCandidate] = []
    for item_key, orders in backlog_by_item.items():
        coverage = state.inventory.get(item_key, 0.0) + pipeline_by_item.get(item_key, 0.0)
        pending: list[tuple[float, float]] = []
        for order in sorted(
            orders,
            key=lambda item: (
                -item.priority,
                -_order_urgency(state.date, item.due_date, urgency_weight, lead_time_days),
                item.due_date or "",
                item.arrival_date,
                item.order_id,
            ),
        ):
            covered = min(coverage, order.quantity_open)
            coverage -= covered
            uncovered = order.quantity_open - covered
            if uncovered <= 0:
                continue
            urgency = _order_urgency(state.date, order.due_date, urgency_weight, lead_time_days)
            pending.append((uncovered, order.priority * urgency * uncovered))

        item_shortfall = sum(quantity for quantity, _ in pending)
        for quantity, value in pending:
            candidates.append(
                _ReorderCandidate(
                    item_key=item_key,
                    quantity=quantity,
                    value=value,
                    item_shortfall=item_shortfall,
                )
            )
    return candidates


def _order_urgency(
    current_date: str, due_date: str | None, urgency_weight: float, lead_time_days: int
) -> float:
    """Urgency multiplier (>=1): higher when the order stays late after a reorder."""

    days_to_due = _days_between(current_date, due_date) if due_date else lead_time_days
    return 1.0 + urgency_weight * max(0, lead_time_days - days_to_due)


def _days_between(start: str, end: str) -> int:
    """Whole days from ISO date `start` to ISO date `end` (negative if past)."""

    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def add_days(date_text: str, days: int) -> str:
    """Return an ISO date string offset by `days`."""

    return (date.fromisoformat(date_text) + timedelta(days=days)).isoformat()
