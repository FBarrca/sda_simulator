from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from random import Random
from typing import Sequence

from domain import DemandArrival, ExogenousInfo, SupplyArrival


# Map each sales order's type to a fulfillment priority (higher = more urgent)
# and the service-level window (days from order date) used as its due date.
_ORDER_TYPE_PRIORITY = {"Urgent": 3, "Normal": 2}
_PRIORITY_SLA_DAYS = {3: 2, 2: 5, 1: 10}


def _demand_priority(order: object) -> int:
    """Derive a 1-3 priority from the order type (defaults to 1, e.g. Backorder)."""

    return _ORDER_TYPE_PRIORITY.get(getattr(order, "order_type", None), 1)


def _demand_due_date(order_date: str, priority: int) -> str:
    """Due date = order date plus a priority-dependent SLA window."""

    return (date.fromisoformat(order_date) + timedelta(days=_PRIORITY_SLA_DAYS[priority])).isoformat()


@dataclass
class InventoryHistoricalSampler:
    """Samples daily demand arrivals and supply receipts from historical days.

    `history` is a sequence of extracted daily records. On reset, the sampler
    chooses a start offset for the replication and then returns consecutive days
    with wraparound. This preserves the observed demand/supply timing within a
    sample path.
    """

    history: Sequence[object]
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate history and initialize replication-local sampling state."""

        if not self.history:
            raise ValueError("history must contain at least one daily record")
        self._rng = Random(self.seed)
        self._start = 0

    def reset(self, replication: int) -> None:
        """Start a replication at a reproducible random historical day."""

        self._rng = Random(None if self.seed is None else self.seed + replication)
        self._start = self._rng.randrange(len(self.history))

    def sample(self, state, t: int) -> ExogenousInfo:
        """Return the next daily demand/supply realization for step `t`."""

        del state
        record = self.history[(self._start + t) % len(self.history)]
        return daily_record_to_exogenous(record)


def daily_record_to_exogenous(record: object) -> ExogenousInfo:
    """Convert an extracted daily record into transition exogenous information."""

    demand_arrivals = tuple(
        DemandArrival(
            order_id=order.order_id,
            material_id=order.material_id,
            plant_id=order.plant_id,
            quantity=order.quantity,
            customer_id=order.customer_id,
            priority=priority,
            due_date=_demand_due_date(order.order_date, priority),
        )
        for order in record.demand_arrivals
        for priority in (_demand_priority(order),)
    )
    supply_arrivals = tuple(
        SupplyArrival(
            receipt_id=movement.movement_id,
            material_id=movement.material_id,
            plant_id=movement.plant_id,
            quantity=movement.quantity,
            purchase_order_id=movement.purchase_order_id,
        )
        for movement in record.supply_arrivals
    )
    return ExogenousInfo(
        date=record.date,
        demand_arrivals=demand_arrivals,
        supply_arrivals=supply_arrivals,
    )
