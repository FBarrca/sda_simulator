from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from random import Random
from typing import Mapping, Sequence

from domain import DemandArrival, ExogenousInfo, SupplyArrival


# Map each sales order's type to a fulfillment priority (higher = more urgent)
# and the service-level window (days from order date) used as its due date.
_ORDER_TYPE_PRIORITY = {"Urgent": 3, "Normal": 2}
_PRIORITY_SLA_DAYS = {3: 2, 2: 5, 1: 10}

# Supply lead-time risk on the orders the policy controls. Each value maps a
# delay in days to its probability. The reorder distribution perturbs the
# planned receipt of a freshly placed SIM-PO (mean ~1.4 days late on a 7-day
# nominal lead time); the expedite distribution is the residual an expedite
# cannot recover, so expediting usually — but not always — lands the goods on
# the requested day.
_LEAD_TIME_DELAY_WEIGHTS = {0: 0.45, 1: 0.20, 2: 0.15, 3: 0.10, 5: 0.06, 7: 0.04}
_EXPEDITE_DELAY_WEIGHTS = {0: 0.60, 1: 0.25, 2: 0.10, 3: 0.05}


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
    lead_time_delay_weights: Mapping[int, float] = field(
        default_factory=lambda: dict(_LEAD_TIME_DELAY_WEIGHTS)
    )
    expedite_delay_weights: Mapping[int, float] = field(
        default_factory=lambda: dict(_EXPEDITE_DELAY_WEIGHTS)
    )

    def __post_init__(self) -> None:
        """Validate history and initialize replication-local sampling state."""

        if not self.history:
            raise ValueError("history must contain at least one daily record")
        self._lead_time_delays = _cumulative(self.lead_time_delay_weights)
        self._expedite_delays = _cumulative(self.expedite_delay_weights)
        self._rng = Random(self.seed)
        self._start = 0

    def reset(self, replication: int) -> None:
        """Start a replication at a reproducible random historical day."""

        self._rng = Random(None if self.seed is None else self.seed + replication)
        self._start = self._rng.randrange(len(self.history))

    def sample(self, state, t: int) -> ExogenousInfo:
        """Return the next daily demand/supply realization for step `t`.

        Two supply-timing shocks are drawn every step with a fixed number of RNG
        calls, independent of the policy's decision, so every policy faces the
        identical exogenous stream for a given seed and replication.
        """

        del state
        record = self.history[(self._start + t) % len(self.history)]
        lead_time_delay_days = _weighted_draw(self._rng, self._lead_time_delays)
        expedite_delay_days = _weighted_draw(self._rng, self._expedite_delays)
        return daily_record_to_exogenous(
            record,
            lead_time_delay_days=lead_time_delay_days,
            expedite_delay_days=expedite_delay_days,
        )


def _cumulative(weights: Mapping[int, float]) -> list[tuple[float, int]]:
    """Build a sorted (cumulative_probability, value) table from a weight map."""

    items = sorted(weights.items())
    total = sum(weight for _, weight in items)
    if total <= 0:
        raise ValueError("delay weights must sum to a positive value")
    table = []
    running = 0.0
    for value, weight in items:
        running += weight / total
        table.append((running, value))
    return table


def _weighted_draw(rng: Random, table: list[tuple[float, int]]) -> int:
    """Draw one value from a cumulative table using exactly one RNG call."""

    threshold = rng.random()
    for cumulative_probability, value in table:
        if threshold <= cumulative_probability:
            return value
    return table[-1][1]


def daily_record_to_exogenous(
    record: object,
    *,
    lead_time_delay_days: int = 0,
    expedite_delay_days: int = 0,
) -> ExogenousInfo:
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
        lead_time_delay_days=lead_time_delay_days,
        expedite_delay_days=expedite_delay_days,
    )
