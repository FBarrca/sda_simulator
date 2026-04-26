from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import sqlite3
from typing import Any


DB_PATH = Path(__file__).resolve().parent / "data" / "inventory_management.db"


@dataclass(frozen=True)
class InventoryPosition:
    material_id: str
    plant_id: str
    storage_location: str
    available_quantity: float
    quality_inspection_quantity: float
    transfer_quantity: float
    blocked_quantity: float
    returns_quantity: float


@dataclass(frozen=True)
class Material:
    material_id: str
    material_type: str
    industry_sector: str
    material_group: str
    valuation_class: str
    gross_weight: float
    net_weight: float
    volume: float
    transport_group: str


@dataclass(frozen=True)
class DemandOrder:
    order_id: str
    customer_id: str
    material_id: str
    plant_id: str
    quantity: float
    order_date: str
    order_type: str
    order_reason: str
    net_price: float


@dataclass(frozen=True)
class PurchaseOrder:
    order_id: str
    supplier_id: str
    material_id: str
    plant_id: str
    quantity: float
    order_date: str
    expected_receipt_date: str | None
    planned_delivery_days: int | None
    blocking_indicator: bool
    net_price: float


@dataclass(frozen=True)
class StockMovement:
    movement_id: str
    movement_type: str
    material_id: str
    plant_id: str
    quantity: float
    signed_quantity: float
    posting_date: str
    purchase_order_id: str | None
    customer_reference_id: str | None


@dataclass(frozen=True)
class ReorderSuggestion:
    suggestion_id: str
    material_id: str
    plant_id: str
    quantity: float
    suggestion_date: str
    order_date: str
    delivery_date: str
    lead_time_days: int


@dataclass(frozen=True)
class CustomerProfile:
    customer_id: str
    total_orders: int
    total_quantity: float
    average_order_quantity: float


@dataclass(frozen=True)
class SupplierProfile:
    supplier_id: str
    total_purchase_orders: int
    total_quantity: float
    average_purchase_quantity: float
    average_lead_time_days: float | None
    materials: tuple[str, ...]


@dataclass(frozen=True)
class DailyInventoryExogenous:
    date: str
    demand_arrivals: tuple[DemandOrder, ...]
    supply_arrivals: tuple[StockMovement, ...]
    purchase_orders_created: tuple[PurchaseOrder, ...]
    reorder_suggestions: tuple[ReorderSuggestion, ...]
    stock_movements: tuple[StockMovement, ...]


@dataclass(frozen=True)
class InventorySimulationData:
    inventory: dict[str, InventoryPosition]
    materials: dict[str, Material]
    customers: dict[str, CustomerProfile]
    suppliers: dict[str, SupplierProfile]
    demand_orders: tuple[DemandOrder, ...]
    purchase_orders: tuple[PurchaseOrder, ...]
    stock_movements: tuple[StockMovement, ...]
    reorder_suggestions: tuple[ReorderSuggestion, ...]
    history: tuple[DailyInventoryExogenous, ...]
    daily_levels: tuple[dict[str, Any], ...]
    policy_item_rows: tuple[dict[str, Any], ...]


def load_inventory_simulation_data(db_path: Path | str = DB_PATH) -> InventorySimulationData:
    """Load simulation-ready state objects and historical exogenous records.

    The result separates the policy state inputs from uncertainty that will be
    sampled later:

    - state: inventory, material, customer, supplier, demand, and purchase objects
    - exogenous history: daily demand arrivals, supply arrivals, PO creation, suggestions
    """

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        materials = _load_materials(conn)
        inventory = _load_inventory(conn)
        demand_orders = tuple(_load_demand_orders(conn))
        purchase_orders = tuple(_load_purchase_orders(conn))
        stock_movements = tuple(_load_stock_movements(conn))
        reorder_suggestions = tuple(_load_reorder_suggestions(conn))
        customers = _build_customer_profiles(demand_orders)
        suppliers = _build_supplier_profiles(purchase_orders)
        history = tuple(
            _build_daily_history(
                demand_orders=demand_orders,
                purchase_orders=purchase_orders,
                stock_movements=stock_movements,
                reorder_suggestions=reorder_suggestions,
            )
        )
        daily_levels = tuple(
            _build_daily_inventory_levels(
                inventory=inventory,
                demand_orders=demand_orders,
                purchase_orders=purchase_orders,
                stock_movements=stock_movements,
                reorder_suggestions=reorder_suggestions,
            )
        )
        policy_item_rows = tuple(
            _build_policy_item_rows(
                inventory=inventory,
                materials=materials,
                demand_orders=demand_orders,
                purchase_orders=purchase_orders,
                stock_movements=stock_movements,
                reorder_suggestions=reorder_suggestions,
            )
        )
        return InventorySimulationData(
            inventory=inventory,
            materials=materials,
            customers=customers,
            suppliers=suppliers,
            demand_orders=demand_orders,
            purchase_orders=purchase_orders,
            stock_movements=stock_movements,
            reorder_suggestions=reorder_suggestions,
            history=history,
            daily_levels=daily_levels,
            policy_item_rows=policy_item_rows,
        )
    finally:
        conn.close()


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {k: to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [to_jsonable(v) for v in value]
    return value


def write_daily_levels_csv(data: InventorySimulationData, path: Path | str) -> None:
    rows = list(data.daily_levels)
    _write_rows_csv(rows, path)


def write_policy_item_csv(data: InventorySimulationData, path: Path | str) -> None:
    rows = list(data.policy_item_rows)
    _write_rows_csv(rows, path)


def _write_rows_csv(rows: list[dict[str, Any]], path: Path | str) -> None:
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_sampling_objects_json(data: InventorySimulationData, path: Path | str) -> None:
    payload = {
        "inventory": data.inventory,
        "materials": data.materials,
        "customers": data.customers,
        "suppliers": data.suppliers,
        "demand_orders": data.demand_orders,
        "purchase_orders": data.purchase_orders,
        "stock_movements": data.stock_movements,
        "reorder_suggestions": data.reorder_suggestions,
        "history": data.history,
    }
    Path(path).write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def _load_materials(conn: sqlite3.Connection) -> dict[str, Material]:
    rows = conn.execute(
        """
        SELECT
            material_number,
            material_type,
            industry_sector,
            material_group,
            valuation_class,
            gross_weight,
            net_weight,
            volume,
            transport_group
        FROM Materials
        ORDER BY material_number
        """
    )
    return {
        _material_id(row["material_number"]): Material(
            material_id=_material_id(row["material_number"]),
            material_type=row["material_type"],
            industry_sector=row["industry_sector"],
            material_group=row["material_group"],
            valuation_class=row["valuation_class"],
            gross_weight=float(row["gross_weight"] or 0),
            net_weight=float(row["net_weight"] or 0),
            volume=float(row["volume"] or 0),
            transport_group=row["transport_group"],
        )
        for row in rows
    }


def _load_inventory(conn: sqlite3.Connection) -> dict[str, InventoryPosition]:
    rows = conn.execute(
        """
        SELECT
            material_number,
            plant,
            storage_location,
            stock_in_quality_inspection,
            stock_in_transfer,
            stock_in_posting,
            blocked_stock,
            returns_stock
        FROM MaterialStocks
        ORDER BY material_number, plant, storage_location
        """
    )
    inventory = {}
    for row in rows:
        material_id = _material_id(row["material_number"])
        plant_id = str(row["plant"])
        transfer_quantity = float(row["stock_in_transfer"] or 0)
        returns_quantity = float(row["returns_stock"] or 0)
        posting_quantity = float(row["stock_in_posting"] or 0)
        blocked_quantity = float(row["blocked_stock"] or 0)
        available_quantity = max(0.0, posting_quantity + transfer_quantity + returns_quantity - blocked_quantity)
        key = _material_plant_key(material_id, plant_id)
        inventory[key] = InventoryPosition(
            material_id=material_id,
            plant_id=plant_id,
            storage_location=str(row["storage_location"]),
            available_quantity=available_quantity,
            quality_inspection_quantity=float(row["stock_in_quality_inspection"] or 0),
            transfer_quantity=transfer_quantity,
            blocked_quantity=blocked_quantity,
            returns_quantity=returns_quantity,
        )
    return inventory


def _load_demand_orders(conn: sqlite3.Connection) -> list[DemandOrder]:
    rows = conn.execute(
        """
        SELECT
            soi.sales_document_number,
            soi.item_number,
            sod.customer_number,
            soi.material_number,
            soi.plant,
            soi.order_quantity,
            sod.document_creation_date,
            sod.order_type,
            sod.order_reason,
            soi.net_price
        FROM SalesOrderItems soi
        JOIN SalesOrderDocuments sod
            ON sod.sales_document_number = soi.sales_document_number
        WHERE soi.material_number IS NOT NULL
        ORDER BY sod.document_creation_date, soi.sales_document_number, soi.item_number
        """
    )
    return [
        DemandOrder(
            order_id=f"SO-{row['sales_document_number']}-{row['item_number']}",
            customer_id=_customer_id(row["customer_number"]),
            material_id=_material_id(row["material_number"]),
            plant_id=str(row["plant"]),
            quantity=float(row["order_quantity"] or 0),
            order_date=str(row["document_creation_date"]),
            order_type=row["order_type"],
            order_reason=row["order_reason"],
            net_price=float(row["net_price"] or 0),
        )
        for row in rows
    ]


def _load_purchase_orders(conn: sqlite3.Connection) -> list[PurchaseOrder]:
    rows = conn.execute(
        """
        WITH ReceiptDates AS (
            SELECT
                purchase_document_number,
                line_item_in_purchase_document,
                MIN(date_of_the_posting_in_the_document) AS first_receipt_date
            FROM GoodsReceiptsAndIssues
            WHERE movement_type = 'Goods Receipt'
            GROUP BY purchase_document_number, line_item_in_purchase_document
        ),
        RequisitionLeadTimes AS (
            SELECT
                purchase_document_number,
                item_number_of_purchasing_document,
                MIN(planned_delivery_time) AS planned_delivery_days,
                MIN(latest_possible_goods_receipt) AS expected_receipt_date
            FROM PurchaseRequisitions
            GROUP BY purchase_document_number, item_number_of_purchasing_document
        )
        SELECT
            poi.purchase_order_number,
            poi.purchase_order_item_number,
            pod.account_number_of_vendor,
            poi.material_number,
            poi.plant,
            poi.quantity,
            pod.purchase_order_date,
            COALESCE(rd.first_receipt_date, rlt.expected_receipt_date) AS expected_receipt_date,
            rlt.planned_delivery_days,
            pod.blocking_indicator,
            poi.net_price
        FROM PurchaseOrderItems poi
        JOIN PurchaseOrderDocuments pod
            ON pod.purchase_document_number = poi.purchase_order_number
        LEFT JOIN ReceiptDates rd
            ON rd.purchase_document_number = poi.purchase_order_number
            AND rd.line_item_in_purchase_document = poi.purchase_order_item_number
        LEFT JOIN RequisitionLeadTimes rlt
            ON rlt.purchase_document_number = poi.purchase_order_number
            AND rlt.item_number_of_purchasing_document = poi.purchase_order_item_number
        WHERE poi.material_number IS NOT NULL
        ORDER BY pod.purchase_order_date, poi.purchase_order_number, poi.purchase_order_item_number
        """
    )
    return [
        PurchaseOrder(
            order_id=f"PO-{row['purchase_order_number']}-{row['purchase_order_item_number']}",
            supplier_id=_supplier_id(row["account_number_of_vendor"]),
            material_id=_material_id(row["material_number"]),
            plant_id=str(row["plant"]),
            quantity=float(row["quantity"] or 0),
            order_date=str(row["purchase_order_date"]),
            expected_receipt_date=(
                str(row["expected_receipt_date"]) if row["expected_receipt_date"] is not None else None
            ),
            planned_delivery_days=(
                int(row["planned_delivery_days"]) if row["planned_delivery_days"] is not None else None
            ),
            blocking_indicator=bool(row["blocking_indicator"]),
            net_price=float(row["net_price"] or 0),
        )
        for row in rows
    ]


def _load_stock_movements(conn: sqlite3.Connection) -> list[StockMovement]:
    rows = conn.execute(
        """
        SELECT
            document_number,
            accounting_document_line,
            movement_type,
            material_number,
            plant,
            quantity,
            date_of_the_posting_in_the_document,
            purchase_document_number,
            line_item_in_purchase_document,
            reference_document_number
        FROM GoodsReceiptsAndIssues
        WHERE material_number IS NOT NULL
        ORDER BY date_of_the_posting_in_the_document, document_number, accounting_document_line
        """
    )
    movements = []
    for row in rows:
        quantity = float(row["quantity"] or 0)
        signed_quantity = quantity if row["movement_type"] == "Goods Receipt" else -quantity
        purchase_order_id = None
        if row["purchase_document_number"] is not None and row["line_item_in_purchase_document"] is not None:
            purchase_order_id = f"PO-{row['purchase_document_number']}-{row['line_item_in_purchase_document']}"
        movements.append(
            StockMovement(
                movement_id=f"GM-{row['document_number']}-{row['accounting_document_line']}",
                movement_type=row["movement_type"],
                material_id=_material_id(row["material_number"]),
                plant_id=str(row["plant"]),
                quantity=quantity,
                signed_quantity=signed_quantity,
                posting_date=str(row["date_of_the_posting_in_the_document"]),
                purchase_order_id=purchase_order_id,
                customer_reference_id=(
                    _customer_id(row["reference_document_number"])
                    if row["movement_type"] == "Goods Issue"
                    else None
                ),
            )
        )
    return movements


def _load_reorder_suggestions(conn: sqlite3.Connection) -> list[ReorderSuggestion]:
    rows = conn.execute(
        """
        SELECT
            order_number,
            order_position,
            article_number,
            order_quantity,
            date,
            order_date,
            delivery_date,
            plant
        FROM OrderSuggestions
        WHERE article_number IS NOT NULL
        ORDER BY date, order_number, order_position
        """
    )
    suggestions = []
    for row in rows:
        order_date = _parse_date(row["order_date"])
        delivery_date = _parse_date(row["delivery_date"])
        suggestions.append(
            ReorderSuggestion(
                suggestion_id=f"RS-{row['order_number']}-{row['order_position']}",
                material_id=_material_id(row["article_number"]),
                plant_id=str(row["plant"]),
                quantity=float(row["order_quantity"] or 0),
                suggestion_date=str(row["date"]),
                order_date=str(row["order_date"]),
                delivery_date=str(row["delivery_date"]),
                lead_time_days=(delivery_date - order_date).days,
            )
        )
    return suggestions


def _build_customer_profiles(demand_orders: tuple[DemandOrder, ...]) -> dict[str, CustomerProfile]:
    by_customer: dict[str, list[DemandOrder]] = defaultdict(list)
    for order in demand_orders:
        by_customer[order.customer_id].append(order)
    return {
        customer_id: CustomerProfile(
            customer_id=customer_id,
            total_orders=len(orders),
            total_quantity=sum(order.quantity for order in orders),
            average_order_quantity=sum(order.quantity for order in orders) / len(orders),
        )
        for customer_id, orders in sorted(by_customer.items())
    }


def _build_supplier_profiles(purchase_orders: tuple[PurchaseOrder, ...]) -> dict[str, SupplierProfile]:
    by_supplier: dict[str, list[PurchaseOrder]] = defaultdict(list)
    for order in purchase_orders:
        by_supplier[order.supplier_id].append(order)

    profiles = {}
    for supplier_id, orders in sorted(by_supplier.items()):
        lead_times = [
            (_parse_date(order.expected_receipt_date) - _parse_date(order.order_date)).days
            for order in orders
            if order.expected_receipt_date is not None
            and _parse_date(order.expected_receipt_date) >= _parse_date(order.order_date)
        ]
        total_quantity = sum(order.quantity for order in orders)
        profiles[supplier_id] = SupplierProfile(
            supplier_id=supplier_id,
            total_purchase_orders=len(orders),
            total_quantity=total_quantity,
            average_purchase_quantity=total_quantity / len(orders),
            average_lead_time_days=(
                sum(lead_times) / len(lead_times) if lead_times else None
            ),
            materials=tuple(sorted({order.material_id for order in orders})),
        )
    return profiles


def _build_daily_history(
    *,
    demand_orders: tuple[DemandOrder, ...],
    purchase_orders: tuple[PurchaseOrder, ...],
    stock_movements: tuple[StockMovement, ...],
    reorder_suggestions: tuple[ReorderSuggestion, ...],
) -> list[DailyInventoryExogenous]:
    demand_by_date = _group_by_date(demand_orders, "order_date")
    purchase_by_date = _group_by_date(purchase_orders, "order_date")
    movement_by_date = _group_by_date(stock_movements, "posting_date")
    suggestion_by_date = _group_by_date(reorder_suggestions, "suggestion_date")

    active_dates = sorted(
        set(demand_by_date)
        | set(purchase_by_date)
        | set(movement_by_date)
        | set(suggestion_by_date)
    )
    dates = _date_range(active_dates[0], active_dates[-1]) if active_dates else []
    return [
        DailyInventoryExogenous(
            date=day,
            demand_arrivals=tuple(demand_by_date.get(day, [])),
            supply_arrivals=tuple(
                movement
                for movement in movement_by_date.get(day, [])
                if movement.movement_type == "Goods Receipt"
            ),
            purchase_orders_created=tuple(purchase_by_date.get(day, [])),
            reorder_suggestions=tuple(suggestion_by_date.get(day, [])),
            stock_movements=tuple(movement_by_date.get(day, [])),
        )
        for day in dates
    ]


def _build_daily_inventory_levels(
    *,
    inventory: dict[str, InventoryPosition],
    demand_orders: tuple[DemandOrder, ...],
    purchase_orders: tuple[PurchaseOrder, ...],
    stock_movements: tuple[StockMovement, ...],
    reorder_suggestions: tuple[ReorderSuggestion, ...],
) -> list[dict[str, Any]]:
    demand_by_day_item = _group_by_day_item(demand_orders, "order_date")
    movement_by_day_item = _group_by_day_item(stock_movements, "posting_date")
    expected_supply_by_day_item = _group_expected_supply_by_day_item(purchase_orders)
    demand_by_date = _group_by_date(demand_orders, "order_date")
    purchase_by_date = _group_by_date(purchase_orders, "order_date")
    movement_by_date = _group_by_date(stock_movements, "posting_date")
    suggestion_by_date = _group_by_date(reorder_suggestions, "suggestion_date")
    expected_supply_by_date = _group_expected_supply_by_date(purchase_orders)

    dates = _all_observed_dates(
        demand_orders=demand_orders,
        purchase_orders=purchase_orders,
        stock_movements=stock_movements,
        reorder_suggestions=reorder_suggestions,
    )
    plants = sorted(
        {position.plant_id for position in inventory.values()}
        | {order.plant_id for order in demand_orders}
        | {order.plant_id for order in purchase_orders}
        | {movement.plant_id for movement in stock_movements}
        | {suggestion.plant_id for suggestion in reorder_suggestions}
    )
    item_keys = sorted(
        set(inventory)
        | {key for _, key in demand_by_day_item}
        | {key for _, key in movement_by_day_item}
        | {
            _material_plant_key(order.material_id, order.plant_id)
            for order in purchase_orders
        }
        | {
            _material_plant_key(suggestion.material_id, suggestion.plant_id)
            for suggestion in reorder_suggestions
        }
    )
    current_stock = {
        key: inventory[key].available_quantity if key in inventory else 0.0 for key in item_keys
    }
    mean_daily_demand = (
        sum(order.quantity for order in demand_orders) / len(dates) if dates else 0.0
    )

    levels = []
    for day in dates:
        stock_start_by_item = dict(current_stock)
        stock_start_by_plant = _stock_by_plant(current_stock)
        demand = demand_by_date.get(day, [])
        purchases = purchase_by_date.get(day, [])
        movements = movement_by_date.get(day, [])
        suggestions = suggestion_by_date.get(day, [])
        actual_supply_arrivals = [
            movement for movement in movements if movement.movement_type == "Goods Receipt"
        ]
        expected_supply_arrivals = expected_supply_by_date.get(day, [])

        goods_receipt_quantity = sum(
            movement.quantity for movement in actual_supply_arrivals
        )
        goods_issue_quantity = sum(
            movement.quantity for movement in movements if movement.movement_type == "Goods Issue"
        )

        stockout_quantity = 0.0
        for item_key in item_keys:
            item_demand_quantity = sum(
                order.quantity for order in demand_by_day_item.get((day, item_key), [])
            )
            item_receipt_quantity = sum(
                movement.quantity
                for movement in movement_by_day_item.get((day, item_key), [])
                if movement.movement_type == "Goods Receipt"
            )
            if item_demand_quantity > 0:
                stockout_quantity += max(
                    0.0, item_demand_quantity - current_stock[item_key] - item_receipt_quantity
                )

        for movement in movements:
            item_key = _material_plant_key(movement.material_id, movement.plant_id)
            if item_key not in current_stock:
                current_stock[item_key] = 0.0
            current_stock[item_key] += movement.signed_quantity

        stock_end_by_plant = _stock_by_plant(current_stock)
        demand_quantity = sum(order.quantity for order in demand)
        lead_times = [
            (_parse_date(order.expected_receipt_date) - _parse_date(order.order_date)).days
            for order in purchases
            if order.expected_receipt_date is not None
            and _parse_date(order.expected_receipt_date) >= _parse_date(order.order_date)
        ]
        stock_end_total = sum(stock_end_by_plant.values())
        row: dict[str, Any] = {
            "date": day,
            "inventory_available_start_total": round(sum(stock_start_by_plant.values()), 4),
            "inventory_available_end_total": round(stock_end_total, 4),
            "demand_arrival_quantity": round(demand_quantity, 4),
            "demand_arrival_count": len(demand),
            "demand_arrivals": json.dumps(
                [
                    {
                        "order_id": order.order_id,
                        "customer_id": order.customer_id,
                        "material_id": order.material_id,
                        "plant_id": order.plant_id,
                        "quantity": order.quantity,
                    }
                    for order in demand
                ],
                separators=(",", ":"),
            ),
            "actual_supply_arrival_quantity": round(goods_receipt_quantity, 4),
            "actual_supply_arrival_count": len(actual_supply_arrivals),
            "actual_supply_arrivals": json.dumps(
                [
                    {
                        "movement_id": movement.movement_id,
                        "purchase_order_id": movement.purchase_order_id,
                        "material_id": movement.material_id,
                        "plant_id": movement.plant_id,
                        "quantity": movement.quantity,
                    }
                    for movement in actual_supply_arrivals
                ],
                separators=(",", ":"),
            ),
            "expected_supply_arrival_quantity": round(
                sum(order.quantity for order in expected_supply_arrivals), 4
            ),
            "expected_supply_arrival_count": len(expected_supply_arrivals),
            "expected_supply_arrivals": json.dumps(
                [
                    {
                        "order_id": order.order_id,
                        "supplier_id": order.supplier_id,
                        "material_id": order.material_id,
                        "plant_id": order.plant_id,
                        "quantity": order.quantity,
                        "order_date": order.order_date,
                        "expected_receipt_date": order.expected_receipt_date,
                        "planned_delivery_days": order.planned_delivery_days,
                    }
                    for order in expected_supply_arrivals
                ],
                separators=(",", ":"),
            ),
            "goods_issue_quantity": round(goods_issue_quantity, 4),
            "goods_receipt_quantity": round(goods_receipt_quantity, 4),
            "purchase_order_quantity": round(sum(order.quantity for order in purchases), 4),
            "purchase_order_count": len(purchases),
            "reorder_suggestion_quantity": round(
                sum(suggestion.quantity for suggestion in suggestions), 4
            ),
            "reorder_suggestion_count": len(suggestions),
            "stockout_quantity": round(stockout_quantity, 4),
            "service_level": round(
                1.0 if demand_quantity == 0 else max(0.0, 1.0 - stockout_quantity / demand_quantity),
                4,
            ),
            "overstock_quantity": round(max(0.0, stock_end_total - 2 * mean_daily_demand), 4),
            "active_material_count": len({order.material_id for order in demand}),
            "customer_count": len({order.customer_id for order in demand}),
            "supplier_count": len({order.supplier_id for order in purchases}),
            "average_supplier_lead_time_days": (
                round(sum(lead_times) / len(lead_times), 4) if lead_times else None
            ),
        }
        for plant in plants:
            plant_demand = sum(order.quantity for order in demand if order.plant_id == plant)
            plant_actual_supply = sum(
                movement.quantity for movement in actual_supply_arrivals if movement.plant_id == plant
            )
            plant_expected_supply = sum(
                order.quantity for order in expected_supply_arrivals if order.plant_id == plant
            )
            plant_issue = sum(
                movement.quantity
                for movement in movements
                if movement.plant_id == plant and movement.movement_type == "Goods Issue"
            )
            plant_receipt = sum(
                movement.quantity
                for movement in movements
                if movement.plant_id == plant and movement.movement_type == "Goods Receipt"
            )
            row[f"inventory_available_start_{plant}"] = round(
                stock_start_by_plant.get(plant, 0.0), 4
            )
            row[f"inventory_available_end_{plant}"] = round(stock_end_by_plant.get(plant, 0.0), 4)
            row[f"demand_arrival_quantity_{plant}"] = round(plant_demand, 4)
            row[f"demand_arrival_count_{plant}"] = sum(1 for order in demand if order.plant_id == plant)
            row[f"actual_supply_arrival_quantity_{plant}"] = round(plant_actual_supply, 4)
            row[f"actual_supply_arrival_count_{plant}"] = sum(
                1 for movement in actual_supply_arrivals if movement.plant_id == plant
            )
            row[f"expected_supply_arrival_quantity_{plant}"] = round(plant_expected_supply, 4)
            row[f"expected_supply_arrival_count_{plant}"] = sum(
                1 for order in expected_supply_arrivals if order.plant_id == plant
            )
            row[f"goods_issue_quantity_{plant}"] = round(plant_issue, 4)
            row[f"goods_receipt_quantity_{plant}"] = round(plant_receipt, 4)
        for item_key in item_keys:
            material_id, plant_id = item_key.split("@", 1)
            column_key = _wide_column_key(plant_id, material_id)
            item_demand = demand_by_day_item.get((day, item_key), [])
            item_movements = movement_by_day_item.get((day, item_key), [])
            item_expected_supply = expected_supply_by_day_item.get((day, item_key), [])
            item_actual_supply_quantity = sum(
                movement.quantity
                for movement in item_movements
                if movement.movement_type == "Goods Receipt"
            )
            item_goods_issue_quantity = sum(
                movement.quantity
                for movement in item_movements
                if movement.movement_type == "Goods Issue"
            )
            row[f"inventory_available_start_{column_key}"] = round(
                stock_start_by_item.get(item_key, 0.0), 4
            )
            row[f"inventory_available_end_{column_key}"] = round(
                current_stock.get(item_key, 0.0), 4
            )
            row[f"demand_arrival_quantity_{column_key}"] = round(
                sum(order.quantity for order in item_demand), 4
            )
            row[f"demand_arrival_count_{column_key}"] = len(item_demand)
            row[f"actual_supply_arrival_quantity_{column_key}"] = round(
                item_actual_supply_quantity, 4
            )
            row[f"actual_supply_arrival_count_{column_key}"] = sum(
                1 for movement in item_movements if movement.movement_type == "Goods Receipt"
            )
            row[f"expected_supply_arrival_quantity_{column_key}"] = round(
                sum(order.quantity for order in item_expected_supply), 4
            )
            row[f"expected_supply_arrival_count_{column_key}"] = len(item_expected_supply)
            row[f"goods_issue_quantity_{column_key}"] = round(item_goods_issue_quantity, 4)
        levels.append(row)
    return levels


def _build_policy_item_rows(
    *,
    inventory: dict[str, InventoryPosition],
    materials: dict[str, Material],
    demand_orders: tuple[DemandOrder, ...],
    purchase_orders: tuple[PurchaseOrder, ...],
    stock_movements: tuple[StockMovement, ...],
    reorder_suggestions: tuple[ReorderSuggestion, ...],
) -> list[dict[str, Any]]:
    demand_by_day_item = _group_by_day_item(demand_orders, "order_date")
    movement_by_day_item = _group_by_day_item(stock_movements, "posting_date")
    suggestion_by_day_item = _group_by_day_item(reorder_suggestions, "suggestion_date")
    expected_supply_by_day_item = _group_expected_supply_by_day_item(purchase_orders)
    movement_by_date = _group_by_date(stock_movements, "posting_date")

    dates = _all_observed_dates(
        demand_orders=demand_orders,
        purchase_orders=purchase_orders,
        stock_movements=stock_movements,
        reorder_suggestions=reorder_suggestions,
    )
    item_keys = sorted(
        set(inventory)
        | {key for _, key in demand_by_day_item}
        | {key for _, key in movement_by_day_item}
        | {key for _, key in suggestion_by_day_item}
        | {key for _, key in expected_supply_by_day_item}
        | {
            _material_plant_key(order.material_id, order.plant_id)
            for order in purchase_orders
        }
    )
    current_stock = {
        key: inventory[key].available_quantity if key in inventory else 0.0 for key in item_keys
    }

    rows = []
    for day in dates:
        stock_start = dict(current_stock)
        for item_key in item_keys:
            material_id, plant_id = item_key.split("@", 1)
            material = materials.get(material_id)
            demand = demand_by_day_item.get((day, item_key), [])
            movements = movement_by_day_item.get((day, item_key), [])
            suggestions = suggestion_by_day_item.get((day, item_key), [])
            expected_supply = expected_supply_by_day_item.get((day, item_key), [])
            actual_supply = [
                movement for movement in movements if movement.movement_type == "Goods Receipt"
            ]
            goods_issues = [
                movement for movement in movements if movement.movement_type == "Goods Issue"
            ]
            demand_quantity = sum(order.quantity for order in demand)
            actual_supply_quantity = sum(movement.quantity for movement in actual_supply)
            goods_issue_quantity = sum(movement.quantity for movement in goods_issues)
            expected_supply_quantity = sum(order.quantity for order in expected_supply)
            inventory_start = stock_start.get(item_key, 0.0)

            rows.append(
                {
                    "date": day,
                    "material_id": material_id,
                    "plant_id": plant_id,
                    "material_type": material.material_type if material else "",
                    "material_group": material.material_group if material else "",
                    "transport_group": material.transport_group if material else "",
                    "state_inventory_available_start": round(inventory_start, 4),
                    "state_pipeline_quantity_due_today": round(
                        _pipeline_quantity(purchase_orders, day, item_key, max_days=0), 4
                    ),
                    "state_pipeline_quantity_due_7_days": round(
                        _pipeline_quantity(purchase_orders, day, item_key, max_days=7), 4
                    ),
                    "state_pipeline_quantity_due_14_days": round(
                        _pipeline_quantity(purchase_orders, day, item_key, max_days=14), 4
                    ),
                    "state_open_purchase_order_count": _open_purchase_order_count(
                        purchase_orders, day, item_key
                    ),
                    "state_reorder_suggestion_quantity": round(
                        sum(suggestion.quantity for suggestion in suggestions), 4
                    ),
                    "state_reorder_suggestion_count": len(suggestions),
                    "exogenous_demand_arrival_quantity": round(demand_quantity, 4),
                    "exogenous_demand_arrival_count": len(demand),
                    "exogenous_demand_customer_count": len({order.customer_id for order in demand}),
                    "exogenous_actual_supply_arrival_quantity": round(actual_supply_quantity, 4),
                    "exogenous_actual_supply_arrival_count": len(actual_supply),
                    "exogenous_expected_supply_arrival_quantity": round(
                        expected_supply_quantity, 4
                    ),
                    "exogenous_expected_supply_arrival_count": len(expected_supply),
                    "exogenous_goods_issue_quantity": round(goods_issue_quantity, 4),
                    "next_inventory_available_end": round(
                        inventory_start + actual_supply_quantity - goods_issue_quantity, 4
                    ),
                    "outcome_stockout_quantity": round(
                        max(0.0, demand_quantity - inventory_start - actual_supply_quantity),
                        4,
                    ),
                    "outcome_service_level": round(
                        1.0
                        if demand_quantity == 0
                        else max(
                            0.0,
                            1.0
                            - max(0.0, demand_quantity - inventory_start - actual_supply_quantity)
                            / demand_quantity,
                        ),
                        4,
                    ),
                    "demand_arrivals": json.dumps(
                        [
                            {
                                "order_id": order.order_id,
                                "customer_id": order.customer_id,
                                "quantity": order.quantity,
                            }
                            for order in demand
                        ],
                        separators=(",", ":"),
                    ),
                    "actual_supply_arrivals": json.dumps(
                        [
                            {
                                "movement_id": movement.movement_id,
                                "purchase_order_id": movement.purchase_order_id,
                                "quantity": movement.quantity,
                            }
                            for movement in actual_supply
                        ],
                        separators=(",", ":"),
                    ),
                    "expected_supply_arrivals": json.dumps(
                        [
                            {
                                "order_id": order.order_id,
                                "supplier_id": order.supplier_id,
                                "quantity": order.quantity,
                                "order_date": order.order_date,
                                "expected_receipt_date": order.expected_receipt_date,
                                "planned_delivery_days": order.planned_delivery_days,
                            }
                            for order in expected_supply
                        ],
                        separators=(",", ":"),
                    ),
                }
            )

        for movement in movement_by_date.get(day, []):
            item_key = _material_plant_key(movement.material_id, movement.plant_id)
            current_stock[item_key] = current_stock.get(item_key, 0.0) + movement.signed_quantity
    return rows


def _all_observed_dates(
    *,
    demand_orders: tuple[DemandOrder, ...],
    purchase_orders: tuple[PurchaseOrder, ...],
    stock_movements: tuple[StockMovement, ...],
    reorder_suggestions: tuple[ReorderSuggestion, ...],
) -> list[str]:
    active_dates = sorted(
        {order.order_date for order in demand_orders}
        | {order.order_date for order in purchase_orders}
        | {movement.posting_date for movement in stock_movements}
        | {suggestion.suggestion_date for suggestion in reorder_suggestions}
    )
    return _date_range(active_dates[0], active_dates[-1]) if active_dates else []


def _group_by_day_item(records: tuple[Any, ...], date_field: str) -> dict[tuple[str, str], list[Any]]:
    grouped: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for record in records:
        key = _material_plant_key(record.material_id, record.plant_id)
        grouped[(getattr(record, date_field), key)].append(record)
    return grouped


def _group_expected_supply_by_date(purchase_orders: tuple[PurchaseOrder, ...]) -> dict[str, list[PurchaseOrder]]:
    grouped: dict[str, list[PurchaseOrder]] = defaultdict(list)
    for order in purchase_orders:
        if order.expected_receipt_date is not None:
            grouped[order.expected_receipt_date].append(order)
    return grouped


def _group_expected_supply_by_day_item(
    purchase_orders: tuple[PurchaseOrder, ...],
) -> dict[tuple[str, str], list[PurchaseOrder]]:
    grouped: dict[tuple[str, str], list[PurchaseOrder]] = defaultdict(list)
    for order in purchase_orders:
        if order.expected_receipt_date is not None:
            key = _material_plant_key(order.material_id, order.plant_id)
            grouped[(order.expected_receipt_date, key)].append(order)
    return grouped


def _pipeline_quantity(
    purchase_orders: tuple[PurchaseOrder, ...], day: str, item_key: str, max_days: int
) -> float:
    current_date = _parse_date(day)
    total = 0.0
    for order in purchase_orders:
        if _material_plant_key(order.material_id, order.plant_id) != item_key:
            continue
        if order.expected_receipt_date is None:
            continue
        order_date = _parse_date(order.order_date)
        receipt_date = _parse_date(order.expected_receipt_date)
        if order_date < current_date and current_date <= receipt_date <= current_date + timedelta(days=max_days):
            total += order.quantity
    return total


def _open_purchase_order_count(
    purchase_orders: tuple[PurchaseOrder, ...], day: str, item_key: str
) -> int:
    current_date = _parse_date(day)
    count = 0
    for order in purchase_orders:
        if _material_plant_key(order.material_id, order.plant_id) != item_key:
            continue
        if order.expected_receipt_date is None:
            continue
        if _parse_date(order.order_date) < current_date <= _parse_date(order.expected_receipt_date):
            count += 1
    return count


def _stock_by_plant(current_stock: dict[str, float]) -> dict[str, float]:
    stock: dict[str, float] = defaultdict(float)
    for item_key, quantity in current_stock.items():
        _, plant_id = item_key.split("@", 1)
        stock[plant_id] += quantity
    return dict(stock)


def _date_range(start: str, end: str) -> list[str]:
    start_date = _parse_date(start)
    end_date = _parse_date(end)
    days = (end_date - start_date).days
    return [(start_date + timedelta(days=offset)).isoformat() for offset in range(days + 1)]


def _group_by_date(records: tuple[Any, ...], date_field: str) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        grouped[getattr(record, date_field)].append(record)
    return grouped


def _material_id(value: Any) -> str:
    return f"MAT-{int(value)}"


def _customer_id(value: Any) -> str:
    return f"CUSTOMER-{int(value)}"


def _supplier_id(value: Any) -> str:
    return f"SUPPLIER-{int(value)}"


def _material_plant_key(material_id: str, plant_id: str) -> str:
    return f"{material_id}@{plant_id}"


def _wide_column_key(plant_id: str, material_id: str) -> str:
    return f"{plant_id}_{material_id}".replace("-", "_")


def _parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def main() -> None:
    data = load_inventory_simulation_data()
    print(
        "Loaded inventory simulation data: "
        f"{len(data.inventory)} inventory positions, "
        f"{len(data.demand_orders)} demand orders, "
        f"{len(data.purchase_orders)} purchase orders, "
        f"{len(data.stock_movements)} stock movements, "
        f"{len(data.history)} daily exogenous records, "
        f"{len(data.daily_levels)} daily inventory rows, "
        f"{len(data.policy_item_rows)} policy item rows."
    )

    sample_path = Path(__file__).resolve().parent / "data" / "inventory_sampling_data.json"
    write_sampling_objects_json(data, sample_path)
    print(f"Wrote {sample_path}")

    csv_path = Path(__file__).resolve().parent / "data" / "daily_inventory_levels.csv"
    write_daily_levels_csv(data, csv_path)
    print(f"Wrote {csv_path}")

    policy_item_path = Path(__file__).resolve().parent / "data" / "policy_item_inputs.csv"
    write_policy_item_csv(data, policy_item_path)
    print(f"Wrote {policy_item_path}")


if __name__ == "__main__":
    main()
