from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace

from domain import Decision, ExogenousInfo, State
from network import route_days


def logistics_transition(state: State, decision: Decision, exogenous: ExogenousInfo) -> State:
    new_state = deepcopy(state)
    orders_by_id = {order.id: order for order in new_state.pending_orders}

    # 1. Apply decisions
    assigned_order_ids: set[str] = set()
    for assignment in decision.order_assignments:
        order = orders_by_id.get(assignment.order_id)
        vehicle = new_state.vehicles.get(assignment.vehicle_id)
        warehouse_inventory = new_state.inventory.get(assignment.warehouse, {})

        if order is None or vehicle is None:
            continue
        if not exogenous.vehicle_availability.get(vehicle.id, True):
            continue
        if vehicle.location != assignment.warehouse:
            continue
        if vehicle.status != "available":
            continue
        if vehicle.remaining_capacity < order.quantity:
            continue
        if warehouse_inventory.get(order.sku, 0) < order.quantity:
            continue

        warehouse_inventory[order.sku] -= order.quantity
        routed_order = replace(order, origin=assignment.warehouse)
        vehicle.route.append(routed_order)
        vehicle.load += order.quantity
        vehicle.status = "en_route"
        traffic_delay = exogenous.travel_times.get(vehicle.id, 0)
        vehicle.time_remaining = route_days(assignment.warehouse, order.destination, traffic_delay)
        assigned_order_ids.add(order.id)

    new_state.pending_orders = [o for o in new_state.pending_orders if o.id not in assigned_order_ids]

    # 2. Process vehicle movements
    for vehicle in new_state.vehicles.values():
        if vehicle.status == "en_route":
            vehicle.time_remaining -= 1
            if vehicle.time_remaining <= 0:
                before = list(vehicle.route)
                vehicle.complete_current_stop()
                if before:
                    new_state.completed_orders.append(before[0])

    # 3. Add new demand realization
    new_state.pending_orders.extend(exogenous.new_orders)

    # 4. Advance time
    new_state.time += 1
    new_state.day_of_week = (new_state.day_of_week + 1) % 7
    return new_state


@dataclass(frozen=True)
class RewardComponents:
    service_value: float
    late_penalty: float

    @property
    def net_reward(self) -> float:
        return self.service_value - self.late_penalty


def reward_components(previous: State, current: State) -> RewardComponents:
    newly_completed = current.completed_orders[len(previous.completed_orders) :]
    service_value = sum(
        order.priority * 12 + min(order.quantity, 32) * 0.5 for order in newly_completed
    )
    late_penalty = sum(
        order.priority * 5 * (current.time - order.deadline)
        for order in current.pending_orders
        if current.time > order.deadline
    )
    return RewardComponents(float(service_value), float(late_penalty))


def reward_completed_minus_late(previous: State, current: State) -> float:
    return reward_components(previous, current).net_reward
