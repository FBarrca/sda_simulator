from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DemandArrival:
    """Customer demand revealed after the policy decision for a day.

    Demand arrivals are exogenous information. The transition adds them to
    backlog so the next decision epoch can allocate available stock to them.
    """

    order_id: str
    material_id: str
    plant_id: str
    quantity: float
    customer_id: str
    priority: int = 1
    due_date: str | None = None


@dataclass(frozen=True)
class SupplyArrival:
    """Supply receipt revealed after the policy decision for a day.

    Supply arrivals increase available inventory and, when linked to a purchase
    order, reduce the open pipeline quantity for that order.
    """

    receipt_id: str
    material_id: str
    plant_id: str
    quantity: float
    purchase_order_id: str | None = None
    supplier_id: str | None = None


@dataclass
class OpenPurchaseOrder:
    """Outstanding replenishment order currently in the supply pipeline."""

    order_id: str
    material_id: str
    plant_id: str
    supplier_id: str
    quantity_open: float
    order_date: str
    expected_receipt_date: str
    expedited: bool = False


@dataclass
class BacklogOrder:
    """Unfulfilled customer demand that is known to the policy."""

    order_id: str
    material_id: str
    plant_id: str
    quantity_open: float
    customer_id: str
    arrival_date: str
    priority: int = 1
    due_date: str | None = None


@dataclass
class State:
    """Inventory system state observed by the policy at a decision epoch.

    `inventory` is keyed by `(material_id, plant_id)`. `pipeline` contains open
    purchase orders that reorder and expedite decisions can modify. `backlog`
    contains demand available for allocation decisions.
    """

    date: str
    inventory: dict[tuple[str, str], float] = field(default_factory=dict)
    pipeline: dict[str, OpenPurchaseOrder] = field(default_factory=dict)
    backlog: dict[str, BacklogOrder] = field(default_factory=dict)
    completed_orders: dict[str, float] = field(default_factory=dict)
    time: int = 0


@dataclass(frozen=True)
class ReorderAction:
    """Decision to create or increase supply pipeline for an item at a plant."""

    order_id: str
    material_id: str
    plant_id: str
    supplier_id: str
    quantity: float
    expected_receipt_date: str


@dataclass(frozen=True)
class ExpediteAction:
    """Decision to pull an open purchase order's expected receipt date earlier."""

    order_id: str
    new_expected_receipt_date: str
    expedite_cost: float = 0.0


@dataclass(frozen=True)
class AllocateStockAction:
    """Decision to allocate available inventory to a known backlog order."""

    order_id: str
    quantity: float
    material_id: str | None = None
    plant_id: str | None = None


@dataclass(frozen=True)
class Decision:
    """Policy output for one decision epoch.

    A policy may reorder inventory, expedite existing purchase orders, and
    allocate stock to already-known backlog in the same decision.
    """

    reorders: tuple[ReorderAction, ...] = ()
    expedites: tuple[ExpediteAction, ...] = ()
    allocations: tuple[AllocateStockAction, ...] = ()


@dataclass(frozen=True)
class ExogenousInfo:
    """Information revealed after the policy decision.

    This contains the uncertainty for the next day: demand arrivals and realized
    supply timing/receipts.
    """

    date: str
    demand_arrivals: tuple[DemandArrival, ...] = ()
    supply_arrivals: tuple[SupplyArrival, ...] = ()


@dataclass(frozen=True)
class TransitionInfo:
    """Diagnostics from applying one transition step."""

    reordered_quantity: float = 0.0
    expedited_orders: int = 0
    received_quantity: float = 0.0
    demand_arrival_quantity: float = 0.0
    allocated_quantity: float = 0.0
    allocated_value: float = 0.0
    stockout_quantity: float = 0.0
    backlog_quantity: float = 0.0
    backlog_value: float = 0.0
    expedite_cost: float = 0.0
