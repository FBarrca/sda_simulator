"""Render the synthetic demand history from ``data.py`` as a two-panel figure,
using only Pillow (no matplotlib dependency), mirroring the approach in
``visualize_network.py``.

Top panel:    7-day rolling-mean stacked area of daily ordered quantity by SKU
              -> shows annual seasonality and SKU mix without day-to-day noise.
Bottom panel: average units/day by day-of-week
              -> shows the weekly rhythm, which a 7-day average would otherwise
              smooth away.

Run from the example directory::

    python3 examples/logistics/visualize_demand.py [--seed 7] [--days 365]

It produces ``logistics_synthetic_demand.png`` next to this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from data import SKUS, synthetic_history

WIDTH = 1000
HEIGHT = 640

FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

# Stacking order, bottom -> top, with a color per SKU.
SKU_COLORS = {
    "AMBIENT_FOOD": (31, 119, 180),
    "COLD_CHAIN": (44, 160, 44),
    "PHARMA": (214, 39, 40),
    "ELECTRONICS": (255, 127, 14),
    "SPARE_PARTS": (148, 103, 189),
}
DOW_LABELS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
ROLLING_WINDOW = 7


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def daily_quantity_by_sku(days: int, seed: int) -> list[dict[str, int]]:
    """Total ordered quantity per SKU, per day, for the synthetic history."""
    history = synthetic_history(days=days, seed=seed)
    series: list[dict[str, int]] = []
    for info in history:
        totals = {sku: 0 for sku in SKUS}
        for order in info.new_orders:
            totals[order.sku] += order.quantity
        series.append(totals)
    return series


def _rolling_mean(values: list[float], window: int) -> list[float]:
    """Centered rolling mean, shrinking the window at the edges."""
    half = window // 2
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        window_values = values[lo:hi]
        out.append(sum(window_values) / len(window_values))
    return out


def render(seed: int, days: int, output: Path) -> None:
    series = daily_quantity_by_sku(days, seed)

    # Smoothed per-SKU series and cumulative stack tops.
    smoothed = {sku: _rolling_mean([d[sku] for d in series], ROLLING_WINDOW) for sku in SKUS}
    cumulative: list[list[float]] = []  # cumulative[i] = running tops over SKUs at day i
    for i in range(days):
        running = 0.0
        tops = []
        for sku in SKUS:
            running += smoothed[sku][i]
            tops.append(running)
        cumulative.append(tops)
    y_max_top = max((tops[-1] for tops in cumulative), default=1.0)
    y_axis_top = ((int(y_max_top) // 20) + 1) * 20

    # Weekly profile: mean total units/day grouped by day-of-week.
    dow_totals = [0.0] * 7
    dow_counts = [0] * 7
    for i, day in enumerate(series):
        dow = i % 7
        dow_totals[dow] += sum(day.values())
        dow_counts[dow] += 1
    dow_means = [t / c if c else 0.0 for t, c in zip(dow_totals, dow_counts)]
    y_max_week = max(dow_means) if dow_means else 1.0
    y_axis_week = ((int(y_max_week) // 20) + 1) * 20

    image = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = _font(FONT_BOLD, 22)
    panel_font = _font(FONT_BOLD, 15)
    label_font = _font(FONT_REGULAR, 13)
    legend_font = _font(FONT_REGULAR, 14)
    small_font = _font(FONT_REGULAR, 12)

    draw.text(
        (60, 16),
        f"Synthetic demand  (seed={seed}, {days} days)",
        fill=(20, 20, 20),
        font=title_font,
    )

    # ---- Top panel: smoothed stacked area by SKU ----
    tx0, ty0, tx1, ty1 = 95, 70, 760, 360
    tw, th = tx1 - tx0, ty1 - ty0

    def top_py(value: float) -> float:
        return ty1 - (value / y_axis_top) * th

    def top_px(day_index: int) -> float:
        return tx0 + (day_index / max(days - 1, 1)) * tw

    draw.text((tx0, ty0 - 26), "7-day rolling mean, stacked by SKU", fill=(40, 40, 40), font=panel_font)

    for i in range(5 + 1):
        value = y_axis_top * i / 5
        yy = top_py(value)
        draw.line([(tx0, yy), (tx1, yy)], fill=(230, 230, 230), width=1)
        draw.text((tx0 - 10, yy), f"{int(value)}", fill=(90, 90, 90), font=small_font, anchor="rm")

    # Draw stacked areas from top SKU down so lower layers stay visible.
    for layer, sku in enumerate(SKUS):
        top_points = [(top_px(i), top_py(cumulative[i][layer])) for i in range(days)]
        if layer == 0:
            bottom_points = [(top_px(i), top_py(0.0)) for i in range(days - 1, -1, -1)]
        else:
            bottom_points = [(top_px(i), top_py(cumulative[i][layer - 1])) for i in range(days - 1, -1, -1)]
        draw.polygon(top_points + bottom_points, fill=SKU_COLORS[sku])

    draw.line([(tx0, ty0), (tx0, ty1)], fill=(60, 60, 60), width=2)
    draw.line([(tx0, ty1), (tx1, ty1)], fill=(60, 60, 60), width=2)
    for month in range(0, days, 30):
        xx = top_px(month)
        draw.line([(xx, ty1), (xx, ty1 + 5)], fill=(60, 60, 60), width=1)
        draw.text((xx, ty1 + 8), f"d{month}", fill=(90, 90, 90), font=small_font, anchor="ma")
    draw.text((tx0 - 65, ty0 + th / 2), "units / day", fill=(90, 90, 90), font=label_font, anchor="mm")

    # Legend (right of the top panel).
    legend_x, legend_y = tx1 + 26, ty0 + 4
    draw.text((legend_x, legend_y - 26), "SKU", fill=(20, 20, 20), font=panel_font)
    totals_by_sku = {sku: sum(d[sku] for d in series) for sku in SKUS}
    for sku in SKUS:
        draw.rectangle([(legend_x, legend_y), (legend_x + 18, legend_y + 18)], fill=SKU_COLORS[sku])
        draw.text(
            (legend_x + 26, legend_y + 9),
            f"{sku}  ({totals_by_sku[sku]:,})",
            fill=(40, 40, 40),
            font=legend_font,
            anchor="lm",
        )
        legend_y += 30
    draw.text(
        (legend_x, legend_y + 12),
        f"total: {sum(totals_by_sku.values()):,} units",
        fill=(20, 20, 20),
        font=_font(FONT_BOLD, 14),
    )

    # ---- Bottom panel: average units/day by day-of-week ----
    bx0, by0, bx1, by1 = 95, 460, 495, 600
    bw = bx1 - bx0

    def week_py(value: float) -> float:
        return by1 - (value / y_axis_week) * (by1 - by0)

    draw.text((bx0, by0 - 26), "Average demand by day of week", fill=(40, 40, 40), font=panel_font)
    for i in range(4 + 1):
        value = y_axis_week * i / 4
        yy = week_py(value)
        draw.line([(bx0, yy), (bx1, yy)], fill=(230, 230, 230), width=1)
        draw.text((bx0 - 10, yy), f"{int(value)}", fill=(90, 90, 90), font=small_font, anchor="rm")

    slot = bw / 7
    bar_w = slot * 0.66
    for dow in range(7):
        x_left = bx0 + dow * slot + (slot - bar_w) / 2
        x_right = x_left + bar_w
        weekend = dow >= 5
        color = (160, 160, 160) if weekend else (31, 119, 180)
        draw.rectangle([(x_left, week_py(dow_means[dow])), (x_right, by1)], fill=color)
        draw.text(
            ((x_left + x_right) / 2, week_py(dow_means[dow]) - 4),
            f"{dow_means[dow]:.0f}",
            fill=(70, 70, 70),
            font=small_font,
            anchor="mb",
        )
        draw.text(((x_left + x_right) / 2, by1 + 6), DOW_LABELS[dow], fill=(90, 90, 90), font=small_font, anchor="ma")

    draw.line([(bx0, by0), (bx0, by1)], fill=(60, 60, 60), width=2)
    draw.line([(bx0, by1), (bx1, by1)], fill=(60, 60, 60), width=2)
    draw.text((bx0 - 65, (by0 + by1) / 2), "units / day", fill=(90, 90, 90), font=label_font, anchor="mm")
    draw.text(
        (bx1 + 30, by0 + 8),
        "Weekend (Sat/Sun) demand\nfalls to ~40-55% of weekday\nlevel via DEMAND_BY_DAY.",
        fill=(90, 90, 90),
        font=label_font,
    )

    image.save(output)
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "logistics_synthetic_demand.png",
    )
    args = parser.parse_args()
    render(args.seed, args.days, args.output)


if __name__ == "__main__":
    main()
