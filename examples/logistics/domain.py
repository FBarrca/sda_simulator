from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Order:
    id: str
    origin: str
    destination: str
    sku: str
    quantity: int
    priority: int = 1
    deadline: int = 999


@dataclass
class Vehicle:
    id: str
    location: str
    capacity: int
    load: int = 0
    status: str = "available"  # available | en_route
    route: list[Order] = field(default_factory=list)
    time_remaining: int = 0

    @property
    def remaining_capacity(self) -> int:
        return self.capacity - self.load

    def complete_current_stop(self) -> None:
        if self.route:
            delivered = self.route.pop(0)
            self.location = delivered.origin
            self.load = max(0, self.load - delivered.quantity)
        self.status = "available" if not self.route else "en_route"
        self.time_remaining = 0 if not self.route else 1


@dataclass
class State:
    inventory: dict[str, dict[str, int]] = field(default_factory=dict)  # warehouse -> sku -> qty
    vehicles: dict[str, Vehicle] = field(default_factory=dict)
    pending_orders: list[Order] = field(default_factory=list)
    completed_orders: list[Order] = field(default_factory=list)
    time: int = 0
    day_of_week: int = 0


@dataclass(frozen=True)
class Assignment:
    order_id: str
    warehouse: str
    vehicle_id: str


@dataclass(frozen=True)
class Decision:
    order_assignments: list[Assignment]


@dataclass(frozen=True)
class ExogenousInfo:
    new_orders: list[Order]
    travel_times: dict[str, int]
    vehicle_availability: dict[str, bool]
