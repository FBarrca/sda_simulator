from __future__ import annotations

WAREHOUSES = ("W_MADRID", "W_BARCELONA", "W_VALENCIA")

CUSTOMERS = (
    "C_MADRID_CENTRO",
    "C_ALCALA",
    "C_BARCELONA_PORT",
    "C_TARRAGONA",
    "C_VALENCIA",
    "C_CASTELLON",
    "C_ZARAGOZA",
    "C_BILBAO",
    "C_SEVILLA",
    "C_MALAGA",
    "C_MURCIA",
    "C_GIRONA",
)

LANE_DISTANCE_KM = {
    "W_MADRID": {
        "C_MADRID_CENTRO": 12,
        "C_ALCALA": 36,
        "C_BARCELONA_PORT": 625,
        "C_TARRAGONA": 545,
        "C_VALENCIA": 360,
        "C_CASTELLON": 420,
        "C_ZARAGOZA": 315,
        "C_BILBAO": 405,
        "C_SEVILLA": 535,
        "C_MALAGA": 530,
        "C_MURCIA": 400,
        "C_GIRONA": 700,
    },
    "W_BARCELONA": {
        "C_MADRID_CENTRO": 625,
        "C_ALCALA": 595,
        "C_BARCELONA_PORT": 10,
        "C_TARRAGONA": 100,
        "C_VALENCIA": 350,
        "C_CASTELLON": 280,
        "C_ZARAGOZA": 315,
        "C_BILBAO": 610,
        "C_SEVILLA": 995,
        "C_MALAGA": 965,
        "C_MURCIA": 590,
        "C_GIRONA": 105,
    },
    "W_VALENCIA": {
        "C_MADRID_CENTRO": 360,
        "C_ALCALA": 330,
        "C_BARCELONA_PORT": 350,
        "C_TARRAGONA": 260,
        "C_VALENCIA": 8,
        "C_CASTELLON": 75,
        "C_ZARAGOZA": 310,
        "C_BILBAO": 610,
        "C_SEVILLA": 655,
        "C_MALAGA": 620,
        "C_MURCIA": 225,
        "C_GIRONA": 455,
    },
}


def lane_distance_km(warehouse: str, customer: str) -> int:
    return LANE_DISTANCE_KM[warehouse][customer]


def route_days(warehouse: str, customer: str, traffic_delay: int = 0) -> int:
    distance = lane_distance_km(warehouse, customer)
    if distance <= 75:
        base_days = 1
    elif distance <= 375:
        base_days = 2
    elif distance <= 650:
        base_days = 3
    else:
        base_days = 4
    return base_days + traffic_delay


def nearest_warehouse(customer: str) -> str:
    return min(WAREHOUSES, key=lambda warehouse: lane_distance_km(warehouse, customer))
