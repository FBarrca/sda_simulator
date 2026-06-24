"""Render how each policy shapes the supply/demand state over the horizon,
using only Pillow (no matplotlib), mirroring ``visualize_data.py``.

Demand is exogenous and identical for every policy; what a policy *controls* is
supply (via reorders) and how it spends on-hand stock. So the clearest "policy
effect on demand vs supply" view is the pair of state trajectories it produces:

Top panel:    on-hand inventory over time, one line per policy
              -> the supply side the policy builds up (or lets balloon).
Bottom panel: open backlog over time, one line per policy
              -> the unmet demand the policy fails to cover.

Each line is the mean across replications (same config as ``run.py``).

Run from the example directory::

    python3 examples/ocel-simulated-inventory/visualize_policies.py

It produces ``inventory_policy_trajectories.png`` next to this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from extract_policy_inputs import load_inventory_simulation_data
from policy import (
    AggressiveReorderPolicy,
    AllocationOnlyPolicy,
    MilpReorderPolicy,
    NoOpPolicy,
    ReorderAllocatePolicy,
    ReorderExpediteAllocatePolicy,
)
from run import initial_state
from sampler import InventoryHistoricalSampler
from transition import inventory_transition, reward_stockout_overstock_service

from sda_mc import Simulator, SimulatorConfig

WIDTH = 1160
HEIGHT = 720

FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

# Drawn (and listed in the legend) in ladder order; the MILP is emphasized.
POLICY_COLORS = {
    "no_action": (140, 140, 140),
    "allocation_only": (31, 119, 180),
    "reorder_allocate": (44, 160, 44),
    "reorder_expedite_allocate": (23, 190, 207),
    "aggressive_reorder_expedite_allocate": (214, 39, 40),
    "milp_reorder_budget": (148, 103, 189),
}
HORIZON = 60
REPLICATIONS = 20
SEED = 42


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def _nice_axis(value: float, step: float) -> float:
    return (int(value // step) + 1) * step


def policy_trajectories(data) -> dict[str, dict[str, list[float]]]:
    """Mean on-hand inventory and backlog per step, for each policy."""

    policies = [
        NoOpPolicy(),
        AllocationOnlyPolicy(),
        ReorderAllocatePolicy(),
        ReorderExpediteAllocatePolicy(),
        AggressiveReorderPolicy(),
        MilpReorderPolicy(),
    ]
    config = SimulatorConfig(horizon=HORIZON, replications=REPLICATIONS, parallel=False)
    start = initial_state(data)
    inv0 = sum(start.inventory.values())
    backlog0 = sum(order.quantity_open for order in start.backlog.values())

    series: dict[str, dict[str, list[float]]] = {}
    for policy in policies:
        sampler = InventoryHistoricalSampler(data.history, seed=SEED)
        simulator = Simulator(
            transition=inventory_transition,
            sampler=sampler,
            reward_fn=reward_stockout_overstock_service,
        )
        trajectories = simulator.run(
            initial_state=lambda _replication: initial_state(data),
            policy=policy,
            config=config,
        )
        inventory_sum = [inv0 * len(trajectories)]
        backlog_sum = [backlog0 * len(trajectories)]
        for t in range(HORIZON):
            inventory_sum.append(0.0)
            backlog_sum.append(0.0)
        for trajectory in trajectories:
            for step in trajectory.steps:
                inventory_sum[step.t + 1] += sum(step.next_state.inventory.values())
                backlog_sum[step.t + 1] += sum(
                    order.quantity_open for order in step.next_state.backlog.values()
                )
        count = len(trajectories)
        series[policy.name] = {
            "inventory": [value / count for value in inventory_sum],
            "backlog": [value / count for value in backlog_sum],
        }
    return series


def _draw_panel(draw, box, series, key, y_axis, title, y_label, fonts) -> None:
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    panel_font, small_font, label_font = fonts

    def px(t: int) -> float:
        return x0 + (t / max(HORIZON, 1)) * width

    def py(value: float) -> float:
        return y1 - (value / y_axis) * height

    draw.text((x0, y0 - 26), title, fill=(40, 40, 40), font=panel_font)
    for i in range(5 + 1):
        value = y_axis * i / 5
        yy = py(value)
        draw.line([(x0, yy), (x1, yy)], fill=(230, 230, 230), width=1)
        draw.text((x0 - 10, yy), f"{int(value):,}", fill=(90, 90, 90), font=small_font, anchor="rm")

    for name, traces in series.items():
        points = [(px(t), py(value)) for t, value in enumerate(traces[key])]
        width_line = 3 if name == "milp_reorder_budget" else 2
        draw.line(points, fill=POLICY_COLORS[name], width=width_line)

    draw.line([(x0, y0), (x0, y1)], fill=(60, 60, 60), width=2)
    draw.line([(x0, y1), (x1, y1)], fill=(60, 60, 60), width=2)
    for t in range(0, HORIZON + 1, 10):
        xx = px(t)
        draw.line([(xx, y1), (xx, y1 + 5)], fill=(60, 60, 60), width=1)
        draw.text((xx, y1 + 8), f"d{t}", fill=(90, 90, 90), font=small_font, anchor="ma")
    draw.text((x0 - 72, (y0 + y1) / 2), y_label, fill=(90, 90, 90), font=label_font, anchor="mm")


def render(output: Path) -> None:
    data = load_inventory_simulation_data()
    series = policy_trajectories(data)

    inv_max = max(value for traces in series.values() for value in traces["inventory"])
    backlog_max = max(value for traces in series.values() for value in traces["backlog"])
    inv_axis = _nice_axis(inv_max, 10000)
    backlog_axis = _nice_axis(backlog_max, 1000)

    image = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = _font(FONT_BOLD, 22)
    panel_font = _font(FONT_BOLD, 15)
    label_font = _font(FONT_REGULAR, 13)
    legend_font = _font(FONT_REGULAR, 13)
    small_font = _font(FONT_REGULAR, 12)
    fonts = (panel_font, small_font, label_font)

    draw.text(
        (60, 16),
        f"How each policy shapes supply & demand  ({HORIZON}-day horizon, mean of {REPLICATIONS} runs)",
        fill=(20, 20, 20),
        font=title_font,
    )

    _draw_panel(
        draw,
        (95, 80, 760, 350),
        series,
        "inventory",
        inv_axis,
        "On-hand inventory (supply the policy builds)",
        "units\non hand",
        fonts,
    )
    _draw_panel(
        draw,
        (95, 460, 760, 660),
        series,
        "backlog",
        backlog_axis,
        "Open backlog (demand left unmet)",
        "units\nbacklog",
        fonts,
    )

    # Legend (right of the panels), with each policy's final inventory/backlog.
    legend_x, legend_y = 786, 84
    draw.text((legend_x, legend_y - 26), "Policy", fill=(20, 20, 20), font=panel_font)
    for name in POLICY_COLORS:
        traces = series[name]
        draw.rectangle([(legend_x, legend_y), (legend_x + 18, legend_y + 14)], fill=POLICY_COLORS[name])
        draw.text(
            (legend_x + 26, legend_y + 7),
            name,
            fill=(40, 40, 40),
            font=legend_font,
            anchor="lm",
        )
        draw.text(
            (legend_x + 26, legend_y + 22),
            f"end: {int(traces['inventory'][-1]):,} inv / {int(traces['backlog'][-1]):,} bklg",
            fill=(110, 110, 110),
            font=_font(FONT_REGULAR, 11),
            anchor="lm",
        )
        legend_y += 42

    image.save(output)
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "inventory_policy_trajectories.png",
    )
    args = parser.parse_args()
    render(args.output)


if __name__ == "__main__":
    main()
