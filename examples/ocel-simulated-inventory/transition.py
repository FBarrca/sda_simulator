from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import date, timedelta

from domain import (
    BacklogOrder,
    Decision,
    ExogenousInfo,
    OpenPurchaseOrder,
    State,
    TransitionInfo,
)


def inventory_transition(
    state: State, decision: Decision, exogenous: ExogenousInfo
) -> State:
    """Apply reorder/expedite/allocate decisions plus realized demand/supply.

    The simulator calls policies before sampling exogenous information. Under
    that convention, allocations in `decision` can only fulfill demand already
    in `state.backlog`; same-day demand arrivals become backlog for the next
    decision epoch.
    """

    next_state, _ = inventory_transition_with_info(state, decision, exogenous)
    return next_state


def inventory_transition_with_info(
    state: State, decision: Decision, exogenous: ExogenousInfo
) -> tuple[State, TransitionInfo]:
    next_state = deepcopy(state)
    # The simulation clock advances monotonically and is anchored at the initial
    # state's date. It is intentionally decoupled from the sampled historical
    # record's date (`exogenous.date`), which can jump ~1 year backward on
    # wraparound and would otherwise strand date-gated SIM-PO receipts.
    next_state.date = _next_day(state.date)

    reordered_quantity = _apply_reorders(next_state, decision)
    expedited_orders, expedite_cost = _apply_expedites(next_state, decision)
    due_pipeline_quantity = _apply_due_simulated_pipeline_receipts(next_state, next_state.date)
    received_quantity = due_pipeline_quantity + _apply_supply_arrivals(next_state, exogenous)
    allocated_quantity, allocated_value = _apply_allocations(next_state, decision)
    demand_arrival_quantity = _apply_demand_arrivals(next_state, exogenous)

    backlog_quantity = sum(order.quantity_open for order in next_state.backlog.values())
    backlog_value = sum(
        order.priority * order.quantity_open for order in next_state.backlog.values()
    )
    stockout_quantity = _stockout_quantity(next_state)
    next_state.time += 1

    return next_state, TransitionInfo(
        reordered_quantity=reordered_quantity,
        expedited_orders=expedited_orders,
        received_quantity=received_quantity,
        demand_arrival_quantity=demand_arrival_quantity,
        allocated_quantity=allocated_quantity,
        allocated_value=allocated_value,
        stockout_quantity=stockout_quantity,
        backlog_quantity=backlog_quantity,
        backlog_value=backlog_value,
        expedite_cost=expedite_cost,
    )


def reward_stockout_overstock_service(
    state: State, decision: Decision, exogenous: ExogenousInfo
) -> float:
    """Contribution C(S_t, x_t, W_{t+1}): service minus backlog and overstock.

    Derived from the decision and realized exogenous information via the same
    transition the simulator uses; the allocated and backlog values come
    straight from its TransitionInfo instead of diffing two states. Service and
    backlog are priority-weighted, so serving (and failing to serve) high-priority
    demand counts more. Expedite cost is charged so the expedite lever trades off.
    """
    next_state, info = inventory_transition_with_info(state, decision, exogenous)
    overstock_quantity = sum(max(0.0, quantity) for quantity in next_state.inventory.values())
    return (
        info.allocated_value
        - 3.0 * info.backlog_value
        - 0.02 * overstock_quantity
        - info.expedite_cost
    )


def _apply_reorders(state: State, decision: Decision) -> float:
    total = 0.0
    for reorder in decision.reorders:
        if reorder.quantity <= 0:
            continue
        existing = state.pipeline.get(reorder.order_id)
        if existing is None:
            state.pipeline[reorder.order_id] = OpenPurchaseOrder(
                order_id=reorder.order_id,
                material_id=reorder.material_id,
                plant_id=reorder.plant_id,
                supplier_id=reorder.supplier_id,
                quantity_open=reorder.quantity,
                order_date=state.date,
                expected_receipt_date=reorder.expected_receipt_date,
            )
        else:
            existing.quantity_open += reorder.quantity
            existing.expected_receipt_date = reorder.expected_receipt_date
        total += reorder.quantity
    return total


def _apply_expedites(state: State, decision: Decision) -> tuple[int, float]:
    expedited = 0
    cost = 0.0
    for expedite in decision.expedites:
        order = state.pipeline.get(expedite.order_id)
        if order is None:
            continue
        order.expected_receipt_date = min(
            order.expected_receipt_date,
            expedite.new_expected_receipt_date,
        )
        order.expedited = True
        expedited += 1
        cost += expedite.expedite_cost
    return expedited, cost


def _apply_supply_arrivals(state: State, exogenous: ExogenousInfo) -> float:
    total = 0.0
    for arrival in exogenous.supply_arrivals:
        if arrival.quantity <= 0:
            continue
        item_key = (arrival.material_id, arrival.plant_id)
        state.inventory[item_key] = state.inventory.get(item_key, 0.0) + arrival.quantity
        total += arrival.quantity

        if arrival.purchase_order_id is None:
            continue
        order = state.pipeline.get(arrival.purchase_order_id)
        if order is None:
            continue
        order.quantity_open = max(0.0, order.quantity_open - arrival.quantity)
        if order.quantity_open == 0:
            del state.pipeline[arrival.purchase_order_id]
    return total


def _apply_due_simulated_pipeline_receipts(state: State, current_date: str) -> float:
    total = 0.0
    received_order_ids = []
    for order in state.pipeline.values():
        if not order.order_id.startswith("SIM-PO-"):
            continue
        if order.expected_receipt_date > current_date:
            continue
        item_key = (order.material_id, order.plant_id)
        state.inventory[item_key] = state.inventory.get(item_key, 0.0) + order.quantity_open
        total += order.quantity_open
        received_order_ids.append(order.order_id)

    for order_id in received_order_ids:
        del state.pipeline[order_id]
    return total


def _apply_allocations(state: State, decision: Decision) -> tuple[float, float]:
    total = 0.0
    weighted_total = 0.0
    for allocation in decision.allocations:
        order = state.backlog.get(allocation.order_id)
        if order is None or allocation.quantity <= 0:
            continue

        material_id = allocation.material_id or order.material_id
        plant_id = allocation.plant_id or order.plant_id
        item_key = (material_id, plant_id)
        available = state.inventory.get(item_key, 0.0)
        quantity = min(allocation.quantity, order.quantity_open, available)
        if quantity <= 0:
            continue

        state.inventory[item_key] = available - quantity
        order.quantity_open -= quantity
        state.completed_orders[order.order_id] = (
            state.completed_orders.get(order.order_id, 0.0) + quantity
        )
        total += quantity
        weighted_total += order.priority * quantity
        if order.quantity_open == 0:
            del state.backlog[order.order_id]
    return total, weighted_total


def _apply_demand_arrivals(state: State, exogenous: ExogenousInfo) -> float:
    total = 0.0
    for arrival in exogenous.demand_arrivals:
        if arrival.quantity <= 0:
            continue
        existing = state.backlog.get(arrival.order_id)
        if existing is None:
            state.backlog[arrival.order_id] = BacklogOrder(
                order_id=arrival.order_id,
                material_id=arrival.material_id,
                plant_id=arrival.plant_id,
                quantity_open=arrival.quantity,
                customer_id=arrival.customer_id,
                arrival_date=exogenous.date,
                priority=arrival.priority,
                due_date=arrival.due_date,
            )
        else:
            state.backlog[arrival.order_id] = replace(
                existing,
                quantity_open=existing.quantity_open + arrival.quantity,
            )
        total += arrival.quantity
    return total


def _next_day(date_text: str) -> str:
    """Return the ISO date one day after `date_text` (the monotonic sim clock)."""

    return (date.fromisoformat(date_text) + timedelta(days=1)).isoformat()


def _stockout_quantity(state: State) -> float:
    by_item: dict[tuple[str, str], float] = {}
    for order in state.backlog.values():
        item_key = (order.material_id, order.plant_id)
        by_item[item_key] = by_item.get(item_key, 0.0) + order.quantity_open
    return sum(
        max(0.0, quantity - state.inventory.get(item_key, 0.0))
        for item_key, quantity in by_item.items()
    )
