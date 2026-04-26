from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Sequence

from domain import DemandArrival, ExogenousInfo, SupplyArrival


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
        )
        for order in record.demand_arrivals
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
