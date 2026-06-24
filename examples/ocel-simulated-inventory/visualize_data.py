"""Render the historical inventory data as a two-panel figure, using only
Pillow (no matplotlib dependency), mirroring the logistics example's
``visualize_demand.py``.

Top panel:    cumulative demand vs cumulative supply receipts over the ~1-year
              history -> shows the structural shortfall the policies must close
              with replenishment, and the lumpy timing of receipts.
Bottom panel: demand quantity by order type (= priority tier)
              -> shows the triage structure that the priority-weighted reward
              and the MILP exploit.

Run from the example directory::

    python3 examples/ocel-simulated-inventory/visualize_data.py

It produces ``inventory_demand_supply.png`` next to this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from extract_policy_inputs import load_inventory_simulation_data

WIDTH = 1000
HEIGHT = 640

FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

DEMAND_COLOR = (31, 119, 180)
SUPPLY_COLOR = (44, 160, 44)
GAP_COLOR = (250, 224, 224)

# Order type -> priority tier the sampler assigns (Urgent->3, Normal->2, else 1).
ORDER_TYPE_PRIORITY = {"Urgent": 3, "Normal": 2, "Backorder": 1}
# Drawn left -> right, highest priority first.
ORDER_TYPE_ORDER = ("Urgent", "Normal", "Backorder")
ORDER_TYPE_COLORS = {
    "Urgent": (214, 39, 40),
    "Normal": (255, 127, 14),
    "Backorder": (148, 103, 189),
}


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def _nice_axis(value: float, step: float) -> float:
    """Round `value` up to the next multiple of `step` for a clean axis top."""

    return (int(value // step) + 1) * step


def daily_demand_and_supply(history) -> tuple[list[float], list[float]]:
    """Total demand-arrival and supply-receipt quantity per historical day."""

    demand = [sum(order.quantity for order in record.demand_arrivals) for record in history]
    supply = [sum(movement.quantity for movement in record.supply_arrivals) for record in history]
    return demand, supply


def demand_quantity_by_order_type(history) -> dict[str, float]:
    """Total ordered quantity grouped by order type (the priority tiers)."""

    totals = {order_type: 0.0 for order_type in ORDER_TYPE_ORDER}
    for record in history:
        for order in record.demand_arrivals:
            totals[order.order_type] = totals.get(order.order_type, 0.0) + order.quantity
    return totals


def _cumulative(values: list[float]) -> list[float]:
    out: list[float] = []
    running = 0.0
    for value in values:
        running += value
        out.append(running)
    return out


def render(output: Path) -> None:
    data = load_inventory_simulation_data()
    history = data.history
    days = len(history)
    start_date, end_date = history[0].date, history[-1].date

    demand, supply = daily_demand_and_supply(history)
    cum_demand = _cumulative(demand)
    cum_supply = _cumulative(supply)
    y_axis_top = _nice_axis(max(cum_demand[-1], cum_supply[-1], 1.0), 5000)

    by_type = demand_quantity_by_order_type(history)
    y_axis_week = _nice_axis(max(by_type.values(), default=1.0), 2000)

    image = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = _font(FONT_BOLD, 22)
    panel_font = _font(FONT_BOLD, 15)
    label_font = _font(FONT_REGULAR, 13)
    legend_font = _font(FONT_REGULAR, 14)
    small_font = _font(FONT_REGULAR, 12)

    draw.text(
        (60, 16),
        f"Historical demand vs supply  ({start_date} → {end_date}, {days} days)",
        fill=(20, 20, 20),
        font=title_font,
    )

    # ---- Top panel: cumulative demand vs cumulative supply ----
    tx0, ty0, tx1, ty1 = 95, 70, 760, 360
    tw, th = tx1 - tx0, ty1 - ty0

    def top_py(value: float) -> float:
        return ty1 - (value / y_axis_top) * th

    def top_px(day_index: int) -> float:
        return tx0 + (day_index / max(days - 1, 1)) * tw

    draw.text(
        (tx0, ty0 - 26),
        "Cumulative units ordered vs received",
        fill=(40, 40, 40),
        font=panel_font,
    )

    for i in range(5 + 1):
        value = y_axis_top * i / 5
        yy = top_py(value)
        draw.line([(tx0, yy), (tx1, yy)], fill=(230, 230, 230), width=1)
        draw.text((tx0 - 10, yy), f"{int(value):,}", fill=(90, 90, 90), font=small_font, anchor="rm")

    demand_points = [(top_px(i), top_py(cum_demand[i])) for i in range(days)]
    supply_points = [(top_px(i), top_py(cum_supply[i])) for i in range(days)]

    # Shade the structural shortfall between the two curves.
    gap_polygon = demand_points + list(reversed(supply_points))
    draw.polygon(gap_polygon, fill=GAP_COLOR)
    draw.line(demand_points, fill=DEMAND_COLOR, width=3)
    draw.line(supply_points, fill=SUPPLY_COLOR, width=3)

    draw.line([(tx0, ty0), (tx0, ty1)], fill=(60, 60, 60), width=2)
    draw.line([(tx0, ty1), (tx1, ty1)], fill=(60, 60, 60), width=2)
    for month in range(0, days, 30):
        xx = top_px(month)
        draw.line([(xx, ty1), (xx, ty1 + 5)], fill=(60, 60, 60), width=1)
        draw.text((xx, ty1 + 8), f"d{month}", fill=(90, 90, 90), font=small_font, anchor="ma")
    draw.text((tx0 - 70, ty0 + th / 2), "cumulative\nunits", fill=(90, 90, 90), font=label_font, anchor="mm")

    # Legend (right of the top panel).
    legend_x, legend_y = tx1 + 26, ty0 + 4
    draw.text((legend_x, legend_y - 26), "Flow", fill=(20, 20, 20), font=panel_font)
    for color, label, total in (
        (DEMAND_COLOR, "demand (sales)", cum_demand[-1]),
        (SUPPLY_COLOR, "supply (receipts)", cum_supply[-1]),
    ):
        draw.rectangle([(legend_x, legend_y), (legend_x + 18, legend_y + 18)], fill=color)
        draw.text(
            (legend_x + 26, legend_y + 9),
            f"{label}\n{int(total):,} units",
            fill=(40, 40, 40),
            font=legend_font,
            anchor="lm",
        )
        legend_y += 44
    shortfall = cum_demand[-1] - cum_supply[-1]
    draw.rectangle([(legend_x, legend_y), (legend_x + 18, legend_y + 18)], fill=GAP_COLOR)
    draw.text(
        (legend_x + 26, legend_y + 9),
        f"shortfall\n{int(shortfall):,} units",
        fill=(40, 40, 40),
        font=legend_font,
        anchor="lm",
    )
    draw.text(
        (legend_x, legend_y + 44),
        "Historical receipts alone\ncannot meet demand —\nreplenishment decisions\nmust close the gap.",
        fill=(90, 90, 90),
        font=label_font,
    )

    # ---- Bottom panel: demand quantity by order type (priority tier) ----
    bx0, by0, bx1, by1 = 95, 460, 495, 600
    bw = bx1 - bx0

    def week_py(value: float) -> float:
        return by1 - (value / y_axis_week) * (by1 - by0)

    draw.text(
        (bx0, by0 - 26),
        "Demand by order type (priority tier)",
        fill=(40, 40, 40),
        font=panel_font,
    )
    for i in range(4 + 1):
        value = y_axis_week * i / 4
        yy = week_py(value)
        draw.line([(bx0, yy), (bx1, yy)], fill=(230, 230, 230), width=1)
        draw.text((bx0 - 10, yy), f"{int(value):,}", fill=(90, 90, 90), font=small_font, anchor="rm")

    slot = bw / len(ORDER_TYPE_ORDER)
    bar_w = slot * 0.6
    for index, order_type in enumerate(ORDER_TYPE_ORDER):
        x_left = bx0 + index * slot + (slot - bar_w) / 2
        x_right = x_left + bar_w
        value = by_type[order_type]
        draw.rectangle(
            [(x_left, week_py(value)), (x_right, by1)],
            fill=ORDER_TYPE_COLORS[order_type],
        )
        draw.text(
            ((x_left + x_right) / 2, week_py(value) - 4),
            f"{int(value):,}",
            fill=(70, 70, 70),
            font=small_font,
            anchor="mb",
        )
        draw.text(
            ((x_left + x_right) / 2, by1 + 6),
            f"{order_type}\np{ORDER_TYPE_PRIORITY[order_type]}",
            fill=(90, 90, 90),
            font=small_font,
            anchor="ma",
        )

    draw.line([(bx0, by0), (bx0, by1)], fill=(60, 60, 60), width=2)
    draw.line([(bx0, by1), (bx1, by1)], fill=(60, 60, 60), width=2)
    draw.text((bx0 - 70, (by0 + by1) / 2), "units\nordered", fill=(90, 90, 90), font=label_font, anchor="mm")
    draw.text(
        (bx1 + 30, by0 + 8),
        "Order type sets each arrival's\npriority (Urgent→3, Normal→2,\nelse 1), which weights both\nservice and backlog in the reward.",
        fill=(90, 90, 90),
        font=label_font,
    )

    image.save(output)
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "inventory_demand_supply.png",
    )
    args = parser.parse_args()
    render(args.output)


if __name__ == "__main__":
    main()
