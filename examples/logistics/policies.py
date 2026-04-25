from __future__ import annotations

from random import Random

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from domain import Assignment, Decision, State
from network import lane_distance_km, route_days


def assignment_score(state: State, assignment: Assignment) -> float:
    order = next(order for order in state.pending_orders if order.id == assignment.order_id)
    distance = lane_distance_km(assignment.warehouse, order.destination)
    duration = route_days(assignment.warehouse, order.destination)
    days_until_deadline = order.deadline - state.time
    urgency_pressure = order.priority * max(0, 5 - days_until_deadline) * 140
    rescue_pressure = order.priority * max(0, -days_until_deadline) * 260
    value = order.priority * 200 + min(order.quantity, 32) * 6
    return value + urgency_pressure + rescue_pressure - duration * 35 - distance * 0.45


def feasible_assignments(state: State) -> list[Assignment]:
    assignments: list[Assignment] = []
    for order in state.pending_orders:
        for warehouse, warehouse_inventory in state.inventory.items():
            if warehouse_inventory.get(order.sku, 0) < order.quantity:
                continue
            for vehicle in state.vehicles.values():
                if vehicle.location != warehouse:
                    continue
                if vehicle.status == "available" and vehicle.remaining_capacity >= order.quantity:
                    assignments.append(Assignment(order.id, warehouse, vehicle.id))
    return assignments


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = Random(seed)

    def decide(self, state: State) -> Decision:
        assignments = feasible_assignments(state)
        self.rng.shuffle(assignments)
        used_orders: set[str] = set()
        used_vehicles: set[str] = set()
        chosen: list[Assignment] = []
        for assignment in assignments:
            if assignment.order_id in used_orders or assignment.vehicle_id in used_vehicles:
                continue
            chosen.append(assignment)
            used_orders.add(assignment.order_id)
            used_vehicles.add(assignment.vehicle_id)
        return Decision(chosen)


class GreedyPolicy:
    name = "greedy_nearest_available"

    def decide(self, state: State) -> Decision:
        chosen: list[Assignment] = []
        used_orders: set[str] = set()
        used_vehicles: set[str] = set()
        orders_by_id = {order.id: order for order in state.pending_orders}
        assignments = sorted(
            feasible_assignments(state),
            key=lambda assignment: lane_distance_km(
                assignment.warehouse,
                orders_by_id[assignment.order_id].destination,
            ),
        )
        for assignment in assignments:
            if assignment.order_id in used_orders or assignment.vehicle_id in used_vehicles:
                continue
            chosen.append(assignment)
            used_orders.add(assignment.order_id)
            used_vehicles.add(assignment.vehicle_id)
        return Decision(chosen)


class PriorityPolicy:
    name = "priority_deadline"

    def decide(self, state: State) -> Decision:
        chosen: list[Assignment] = []
        used_orders: set[str] = set()
        used_vehicles: set[str] = set()
        orders_by_id = {order.id: order for order in state.pending_orders}
        reserved_inventory = {
            warehouse: dict(skus) for warehouse, skus in state.inventory.items()
        }
        candidates: list[tuple[float, Assignment]] = []

        for assignment in feasible_assignments(state):
            candidates.append((-assignment_score(state, assignment), assignment))

        for _, assignment in sorted(candidates, key=lambda candidate: candidate[0]):
            if assignment.order_id in used_orders or assignment.vehicle_id in used_vehicles:
                continue
            order = orders_by_id[assignment.order_id]
            if reserved_inventory[assignment.warehouse].get(order.sku, 0) < order.quantity:
                continue
            chosen.append(assignment)
            used_orders.add(assignment.order_id)
            used_vehicles.add(assignment.vehicle_id)
            reserved_inventory[assignment.warehouse][order.sku] -= order.quantity

        return Decision(chosen)


class MilpPolicy:
    name = "milp_distance_priority"

    def __init__(self, time_limit_seconds: float = 0.25) -> None:
        self.time_limit_seconds = time_limit_seconds

    def decide(self, state: State) -> Decision:
        orders_by_id = {order.id: order for order in state.pending_orders}
        candidates = [
            (assignment_score(state, assignment), assignment)
            for assignment in feasible_assignments(state)
        ]
        if not candidates:
            return Decision([])

        objective = np.array([-score for score, _assignment in candidates], dtype=float)
        constraint_rows: list[list[float]] = []
        upper_bounds: list[float] = []

        for order_id in orders_by_id:
            constraint_rows.append(
                [1.0 if assignment.order_id == order_id else 0.0 for _score, assignment in candidates]
            )
            upper_bounds.append(1.0)

        available_vehicle_ids = {
            assignment.vehicle_id for _score, assignment in candidates
        }
        for vehicle_id in available_vehicle_ids:
            constraint_rows.append(
                [
                    1.0 if assignment.vehicle_id == vehicle_id else 0.0
                    for _score, assignment in candidates
                ]
            )
            upper_bounds.append(1.0)

        for warehouse, skus in state.inventory.items():
            for sku, quantity_available in skus.items():
                constraint_rows.append(
                    [
                        float(orders_by_id[assignment.order_id].quantity)
                        if assignment.warehouse == warehouse
                        and orders_by_id[assignment.order_id].sku == sku
                        else 0.0
                        for _score, assignment in candidates
                    ]
                )
                upper_bounds.append(float(quantity_available))

        constraints = LinearConstraint(
            np.array(constraint_rows, dtype=float),
            lb=np.full(len(constraint_rows), -np.inf),
            ub=np.array(upper_bounds, dtype=float),
        )
        result = milp(
            c=objective,
            integrality=np.ones(len(candidates)),
            bounds=Bounds(0, 1),
            constraints=constraints,
            options={"time_limit": self.time_limit_seconds, "mip_rel_gap": 0.01},
        )
        if result.x is None or not result.success:
            return PriorityPolicy().decide(state)

        chosen = [
            assignment
            for selected, (_score, assignment) in zip(result.x, candidates, strict=True)
            if selected >= 0.5
        ]
        return Decision(chosen)
