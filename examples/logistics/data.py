from __future__ import annotations

from dataclasses import dataclass
from random import Random

from domain import ExogenousInfo, Order, State, Vehicle
from network import nearest_warehouse


SKUS = ("AMBIENT_FOOD", "COLD_CHAIN", "PHARMA", "ELECTRONICS", "SPARE_PARTS")

WAREHOUSES = {
    "W_MADRID": {
        "AMBIENT_FOOD": 620,
        "COLD_CHAIN": 260,
        "PHARMA": 150,
        "ELECTRONICS": 170,
        "SPARE_PARTS": 220,
    },
    "W_BARCELONA": {
        "AMBIENT_FOOD": 540,
        "COLD_CHAIN": 300,
        "PHARMA": 130,
        "ELECTRONICS": 230,
        "SPARE_PARTS": 170,
    },
    "W_VALENCIA": {
        "AMBIENT_FOOD": 360,
        "COLD_CHAIN": 210,
        "PHARMA": 90,
        "ELECTRONICS": 90,
        "SPARE_PARTS": 140,
    },
}

SKU_PROFILES = {
    "AMBIENT_FOOD": {"mean": 18, "spread": 11, "priority": (1, 1, 2), "lead_time": (2, 5)},
    "COLD_CHAIN": {"mean": 12, "spread": 7, "priority": (2, 2, 3), "lead_time": (1, 3)},
    "PHARMA": {"mean": 7, "spread": 4, "priority": (2, 3, 3), "lead_time": (1, 2)},
    "ELECTRONICS": {"mean": 9, "spread": 6, "priority": (1, 2, 2), "lead_time": (3, 7)},
    "SPARE_PARTS": {"mean": 6, "spread": 5, "priority": (1, 2, 3), "lead_time": (1, 5)},
}

DEMAND_BY_DAY = (0.90, 1.15, 1.20, 1.10, 0.95, 0.55, 0.40)
MONTHLY_SEASONALITY = (0.92, 0.95, 1.02, 1.08, 1.10, 1.00, 0.88, 0.82, 1.12, 1.18, 1.24, 1.35)
REGIONAL_DEMAND_WEIGHTS = {
    "C_MADRID_CENTRO": 1.35,
    "C_ALCALA": 0.80,
    "C_BARCELONA_PORT": 1.25,
    "C_TARRAGONA": 0.75,
    "C_VALENCIA": 1.10,
    "C_CASTELLON": 0.60,
    "C_ZARAGOZA": 0.85,
    "C_BILBAO": 0.95,
    "C_SEVILLA": 1.00,
    "C_MALAGA": 0.90,
    "C_MURCIA": 0.70,
    "C_GIRONA": 0.65,
}

@dataclass(frozen=True)
class Disruption:
    demand_lift: float = 1.0
    traffic_lift: int = 0
    outage_probability: float = 0.0


def initial_state() -> State:
    return State(
        inventory={warehouse: dict(skus) for warehouse, skus in WAREHOUSES.items()},
        vehicles={
            "V_MADRID_1": Vehicle(id="V_MADRID_1", location="W_MADRID", capacity=32),
            "V_MADRID_2": Vehicle(id="V_MADRID_2", location="W_MADRID", capacity=26),
            "V_MADRID_3": Vehicle(id="V_MADRID_3", location="W_MADRID", capacity=18),
            "V_BARCELONA_1": Vehicle(id="V_BARCELONA_1", location="W_BARCELONA", capacity=30),
            "V_BARCELONA_2": Vehicle(id="V_BARCELONA_2", location="W_BARCELONA", capacity=22),
            "V_VALENCIA_1": Vehicle(id="V_VALENCIA_1", location="W_VALENCIA", capacity=24),
        },
        pending_orders=[
            Order("O_BOOT_1", "W_MADRID", "C_VALENCIA", "COLD_CHAIN", 12, priority=3, deadline=2),
            Order("O_BOOT_2", "W_BARCELONA", "C_GIRONA", "AMBIENT_FOOD", 18, priority=1, deadline=4),
            Order("O_BOOT_3", "W_VALENCIA", "C_MURCIA", "PHARMA", 6, priority=3, deadline=1),
            Order("O_BOOT_4", "W_MADRID", "C_BILBAO", "SPARE_PARTS", 8, priority=2, deadline=3),
        ],
    )


def synthetic_history(days: int = 180, seed: int = 11) -> list[ExogenousInfo]:
    rng = Random(seed)
    history: list[ExogenousInfo] = []
    for day in range(days):
        day_of_week = day % 7
        month = (day // 30) % 12
        disruption = _daily_disruption(rng, day_of_week, month)
        demand_level = (
            5.5
            * DEMAND_BY_DAY[day_of_week]
            * MONTHLY_SEASONALITY[month]
            * disruption.demand_lift
        )
        order_count = _sample_order_count(rng, demand_level)

        new_orders = []
        for k in range(order_count):
            customer = _weighted_choice(rng, REGIONAL_DEMAND_WEIGHTS)
            sku = _sample_sku(rng, customer, month)
            profile = SKU_PROFILES[sku]
            qty = _sample_quantity(rng, int(profile["mean"]), int(profile["spread"]))
            min_lead, max_lead = profile["lead_time"]
            new_orders.append(
                Order(
                    id=f"H{day}_{k}",
                    origin=nearest_warehouse(customer),
                    destination=customer,
                    sku=sku,
                    quantity=qty,
                    priority=rng.choice(profile["priority"]),
                    deadline=day + rng.randint(min_lead, max_lead),
                )
            )

        travel_times = _travel_times(rng, disruption)
        vehicle_availability = _vehicle_availability(rng, disruption)
        history.append(
            ExogenousInfo(
                new_orders=new_orders,
                travel_times=travel_times,
                vehicle_availability=vehicle_availability,
            )
        )
    return history


def _daily_disruption(rng: Random, day_of_week: int, month: int) -> Disruption:
    holiday_peak = month == 11 and rng.random() < 0.18
    promotion = day_of_week in {1, 2, 3} and rng.random() < 0.08
    severe_weather = month in {0, 1, 9, 10} and rng.random() < 0.07
    port_congestion = rng.random() < 0.05

    demand_lift = 1.0
    traffic_lift = 0
    outage_probability = 0.0

    if holiday_peak:
        demand_lift += rng.uniform(0.45, 0.95)
        traffic_lift += 1
    if promotion:
        demand_lift += rng.uniform(0.25, 0.60)
    if severe_weather:
        demand_lift -= rng.uniform(0.05, 0.20)
        traffic_lift += rng.randint(1, 2)
        outage_probability += 0.06
    if port_congestion:
        traffic_lift += 1
        outage_probability += 0.03

    return Disruption(max(0.35, demand_lift), traffic_lift, outage_probability)


def _sample_order_count(rng: Random, expected: float) -> int:
    baseline = int(expected)
    fractional_order = 1 if rng.random() < expected - baseline else 0
    noise = rng.choice([-2, -1, 0, 0, 1, 1, 2, 3])
    return max(0, baseline + fractional_order + noise)


def _sample_sku(rng: Random, customer: str, month: int) -> str:
    weights = {
        "AMBIENT_FOOD": 4.0,
        "COLD_CHAIN": 2.1,
        "PHARMA": 1.2,
        "ELECTRONICS": 1.8,
        "SPARE_PARTS": 1.6,
    }
    if customer in {"C_BARCELONA_PORT", "C_VALENCIA", "C_SEVILLA", "C_MALAGA"}:
        weights["COLD_CHAIN"] += 0.8
    if customer in {"C_BILBAO", "C_ZARAGOZA", "C_MURCIA"}:
        weights["SPARE_PARTS"] += 0.7
    if month in {10, 11}:
        weights["ELECTRONICS"] += 1.4
    if month in {5, 6, 7}:
        weights["COLD_CHAIN"] += 0.9
    return _weighted_choice(rng, weights)


def _sample_quantity(rng: Random, mean: int, spread: int) -> int:
    triangular = int(rng.triangular(max(1, mean - spread), mean + spread, mean))
    if rng.random() < 0.06:
        triangular += rng.randint(8, 14)
    return max(1, min(32, triangular))


def _travel_times(rng: Random, disruption: Disruption) -> dict[str, int]:
    vehicles = (
        "V_MADRID_1",
        "V_MADRID_2",
        "V_MADRID_3",
        "V_BARCELONA_1",
        "V_BARCELONA_2",
        "V_VALENCIA_1",
    )
    return {vehicle_id: disruption.traffic_lift + rng.choice([0, 0, 0, 1]) for vehicle_id in vehicles}


def _vehicle_availability(rng: Random, disruption: Disruption) -> dict[str, bool]:
    maintenance_probability = {
        "V_MADRID_1": 0.04,
        "V_MADRID_2": 0.06,
        "V_MADRID_3": 0.08,
        "V_BARCELONA_1": 0.05,
        "V_BARCELONA_2": 0.07,
        "V_VALENCIA_1": 0.06,
    }
    return {
        vehicle_id: rng.random() > probability + disruption.outage_probability
        for vehicle_id, probability in maintenance_probability.items()
    }


def _weighted_choice(rng: Random, weights: dict[str, float]) -> str:
    total = sum(weights.values())
    draw = rng.uniform(0, total)
    cumulative = 0.0
    for item, weight in weights.items():
        cumulative += weight
        if draw <= cumulative:
            return item
    return next(reversed(weights))
