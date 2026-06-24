"""Render what each policy *does* and what it *gets*, using only Pillow.

Demand is exogenous and identical for every policy; what a policy controls is
the three levers it pulls each day — **reorder**, **expedite**, **allocate** —
and the state and cost those levers produce. This dashboard lays that whole
causal chain out as six small-multiple panels (same config as ``run.py``):

    what the policy DOES            what STATE results        the OUTCOME
    ─────────────────────           ──────────────────         ───────────
    cumulative units reordered      on-hand inventory          cumulative cost
    cumulative expedites            open backlog               (= −reward)
    cumulative units allocated

Each line is the mean across replications. Panels whose values span orders of
magnitude use a log y-axis (marked "log") so the lean, right-sized policies
(``demand_scaled``, ``milp``) are distinguishable from the overstocked ones
instead of being crushed against the floor of a linear scale.

Run from the example directory::

    python3 examples/ocel-simulated-inventory/visualize_policies.py

It produces ``inventory_policy_trajectories.png`` next to this script.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from extract_policy_inputs import load_inventory_simulation_data
from policy import (
    AggressiveReorderPolicy,
    AllocationOnlyPolicy,
    DemandScaledReorderExpeditePolicy,
    MilpReorderPolicy,
    NoOpPolicy,
    ReorderAllocatePolicy,
    ReorderExpediteAllocatePolicy,
    compute_demand_scaled_targets,
)
from run import initial_state
from sampler import InventoryHistoricalSampler
from transition import inventory_transition, reward_stockout_overstock_service

from sda_mc import Simulator, SimulatorConfig

WIDTH = 1200
HEIGHT = 1020

FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

# Drawn in ladder order. The two right-sized policies (the interesting pair) are
# emphasized with a thicker line; the rest are context.
POLICY_COLORS = {
    "no_action": (140, 140, 140),
    "allocation_only": (31, 119, 180),
    "reorder_allocate": (44, 160, 44),
    "reorder_expedite_allocate": (23, 190, 207),
    "aggressive_reorder_expedite_allocate": (214, 39, 40),
    "demand_scaled_reorder_expedite_allocate": (255, 127, 14),
    "milp_reorder_budget": (148, 103, 189),
}
EMPHASIZED = {"demand_scaled_reorder_expedite_allocate", "milp_reorder_budget"}

HORIZON = 60
REPLICATIONS = 20
SEED = 42

# Panel grid geometry.
PANEL_W = 430
PANEL_H = 165
COL_X = (95, 700)
ROW_Y = (110, 350, 590)


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def _abbrev(value: float) -> str:
    """Compact 3-significant-figure label: 1009340 -> '1.01M', 12000 -> '12k'."""

    magnitude = abs(value)
    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:.3g}M"
    if magnitude >= 1_000:
        return f"{value / 1_000:.3g}k"
    return f"{value:.3g}"


def policy_trajectories(data) -> dict[str, dict[str, list[float]]]:
    """Mean per-step lever, state, and cost trajectories for each policy.

    Returns, per policy name, six series of length ``HORIZON + 1`` (an initial
    point plus one per step):

    - ``reordered`` / ``expedites`` / ``allocated``: cumulative levers pulled
      (units ordered, expedite actions, units allocated to backlog).
    - ``inventory`` / ``backlog``: on-hand stock and open backlog after the step.
    - ``cum_reward``: cumulative reward (negative; plotted as cost = −reward).
    """

    reorder_points, order_up_to_targets = compute_demand_scaled_targets(
        data.history, lead_time_days=7
    )
    policies = [
        NoOpPolicy(),
        AllocationOnlyPolicy(),
        ReorderAllocatePolicy(),
        ReorderExpediteAllocatePolicy(),
        AggressiveReorderPolicy(),
        DemandScaledReorderExpeditePolicy(reorder_points, order_up_to_targets),
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

        # Accumulators indexed 0..HORIZON; index 0 is the pre-decision starting
        # point. Levers and reward are cumulative; inventory/backlog are levels.
        zeros = lambda first: [first] + [0.0] * HORIZON  # noqa: E731
        sums = {
            "reordered": zeros(0.0),
            "expedites": zeros(0.0),
            "allocated": zeros(0.0),
            "inventory": zeros(inv0 * len(trajectories)),
            "backlog": zeros(backlog0 * len(trajectories)),
            "cum_reward": zeros(0.0),
        }
        for trajectory in trajectories:
            running_reorder = running_expedite = running_alloc = running_reward = 0.0
            for step in trajectory.steps:
                index = step.t + 1
                # Levers come straight from the decision: the policy's allocation
                # quantities are already capped at on-hand stock, and receipts in
                # the transition only raise availability, so requested == realized.
                running_reorder += sum(action.quantity for action in step.decision.reorders)
                running_expedite += len(step.decision.expedites)
                running_alloc += sum(action.quantity for action in step.decision.allocations)
                running_reward += step.reward
                sums["reordered"][index] += running_reorder
                sums["expedites"][index] += running_expedite
                sums["allocated"][index] += running_alloc
                sums["cum_reward"][index] += running_reward
                sums["inventory"][index] += sum(step.next_state.inventory.values())
                sums["backlog"][index] += sum(
                    order.quantity_open for order in step.next_state.backlog.values()
                )
        count = len(trajectories)
        series[policy.name] = {key: [value / count for value in values] for key, values in sums.items()}
    return series


def _linear_ticks(max_value: float) -> tuple[float, list[float]]:
    """Return (axis_max, tick_values) giving ~5 round ticks for a linear axis."""

    if max_value <= 0:
        return 1.0, [0.0, 1.0]
    raw = max_value / 5
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 5, 10) if m * magnitude >= raw)
    axis_max = math.ceil(max_value / step) * step
    ticks = [step * i for i in range(int(axis_max / step) + 1)]
    return axis_max, ticks


def _log_bounds(values: list[float]) -> tuple[float, float, list[float]]:
    """Return (lo, hi, decade_ticks) bracketing the positive values by decades."""

    positive = [v for v in values if v > 0]
    if not positive:
        return 1.0, 10.0, [1.0, 10.0]
    lo_exp = math.floor(math.log10(min(positive)))
    hi_exp = math.ceil(math.log10(max(positive)))
    if hi_exp == lo_exp:
        hi_exp += 1
    ticks = [10.0**exp for exp in range(lo_exp, hi_exp + 1)]
    return 10.0**lo_exp, 10.0**hi_exp, ticks


def _draw_panel(draw, box, series, panel, fonts) -> None:
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    panel_font, small_font, label_font = fonts
    key = panel["key"]
    log_scale = panel["scale"] == "log"
    sign = -1.0 if panel.get("negate") else 1.0

    traces = {name: [sign * value for value in series[name][key]] for name in POLICY_COLORS}
    all_values = [value for values in traces.values() for value in values]

    if log_scale:
        lo, hi, ticks = _log_bounds(all_values)
        log_lo, log_hi = math.log10(lo), math.log10(hi)

        def py(value: float) -> float:
            clamped = value if value > lo else lo
            frac = (math.log10(clamped) - log_lo) / (log_hi - log_lo)
            return y1 - frac * height
    else:
        axis_max, ticks = _linear_ticks(max(all_values) if all_values else 1.0)

        def py(value: float) -> float:
            return y1 - (value / axis_max) * height

    def px(t: int) -> float:
        return x0 + (t / max(HORIZON, 1)) * width

    # Title (with a scale hint) and horizontal gridlines + tick labels.
    title = panel["title"] + ("   (log)" if log_scale else "")
    draw.text((x0, y0 - 24), title, fill=(40, 40, 40), font=panel_font)
    for tick in ticks:
        yy = py(tick)
        if yy < y0 - 1 or yy > y1 + 1:
            continue
        draw.line([(x0, yy), (x1, yy)], fill=(232, 232, 232), width=1)
        draw.text((x0 - 8, yy), _abbrev(tick), fill=(90, 90, 90), font=small_font, anchor="rm")

    for name, values in traces.items():
        points = [(px(t), py(value)) for t, value in enumerate(values)]
        line_width = 3 if name in EMPHASIZED else 2
        draw.line(points, fill=POLICY_COLORS[name], width=line_width)

    draw.line([(x0, y0), (x0, y1)], fill=(60, 60, 60), width=2)
    draw.line([(x0, y1), (x1, y1)], fill=(60, 60, 60), width=2)
    for t in range(0, HORIZON + 1, 15):
        xx = px(t)
        draw.line([(xx, y1), (xx, y1 + 5)], fill=(60, 60, 60), width=1)
        draw.text((xx, y1 + 8), f"d{t}", fill=(90, 90, 90), font=small_font, anchor="ma")
    draw.text((x0 - 60, y0 - 24), panel["y_label"], fill=(120, 120, 120), font=label_font, anchor="lm")


# Six panels: levers the policy pulls (left col), the state they produce and the
# resulting cost (right col). Order reads "what it does" -> "what it gets".
PANELS = [
    {"title": "Units reordered (cumulative)", "key": "reordered", "scale": "log", "y_label": "ordered"},
    {"title": "On-hand inventory", "key": "inventory", "scale": "log", "y_label": "on hand"},
    {"title": "Expedites (cumulative)", "key": "expedites", "scale": "linear", "y_label": "actions"},
    {"title": "Open backlog", "key": "backlog", "scale": "log", "y_label": "backlog"},
    {"title": "Units allocated to demand (cumulative)", "key": "allocated", "scale": "linear", "y_label": "served"},
    {"title": "Cost = −reward (cumulative, lower=better)", "key": "cum_reward", "scale": "log", "y_label": "cost", "negate": True},
]


def render(output: Path) -> None:
    data = load_inventory_simulation_data()
    series = policy_trajectories(data)

    image = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = _font(FONT_BOLD, 22)
    subtitle_font = _font(FONT_REGULAR, 14)
    panel_font = _font(FONT_BOLD, 14)
    label_font = _font(FONT_REGULAR, 11)
    legend_sub_font = _font(FONT_REGULAR, 11)
    small_font = _font(FONT_REGULAR, 11)
    fonts = (panel_font, small_font, label_font)

    draw.text(
        (60, 18),
        "What each policy does, and what it gets",
        fill=(20, 20, 20),
        font=title_font,
    )
    draw.text(
        (60, 48),
        f"Levers pulled -> state produced -> cost incurred   ·   {HORIZON}-day horizon, mean of "
        f"{REPLICATIONS} runs   ·   thick lines = the two right-sized policies",
        fill=(90, 90, 90),
        font=subtitle_font,
    )

    for index, panel in enumerate(PANELS):
        row, col = index // 2, index % 2
        x0 = COL_X[col]
        y0 = ROW_Y[row]
        _draw_panel(draw, (x0, y0, x0 + PANEL_W, y0 + PANEL_H), series, panel, fonts)

    # Legend across the bottom: two columns, each entry showing the policy's
    # end-of-horizon headline numbers so the lines are identifiable at a glance.
    legend_top = ROW_Y[2] + PANEL_H + 52
    draw.text((95, legend_top - 26), "Policy  (end of horizon)", fill=(20, 20, 20), font=panel_font)
    names = list(POLICY_COLORS)
    columns = ((95, names[:4]), (700, names[4:]))
    for legend_x, column_names in columns:
        legend_y = legend_top
        for name in column_names:
            traces = series[name]
            draw.rectangle(
                [(legend_x, legend_y), (legend_x + 20, legend_y + 14)], fill=POLICY_COLORS[name]
            )
            weight = FONT_BOLD if name in EMPHASIZED else FONT_REGULAR
            draw.text(
                (legend_x + 28, legend_y + 7),
                name,
                fill=(30, 30, 30),
                font=_font(weight, 13),
                anchor="lm",
            )
            draw.text(
                (legend_x + 28, legend_y + 24),
                f"cost {_abbrev(-traces['cum_reward'][-1])}  ·  "
                f"inv {_abbrev(traces['inventory'][-1])}  ·  "
                f"bklg {int(traces['backlog'][-1]):,}",
                fill=(115, 115, 115),
                font=legend_sub_font,
                anchor="lm",
            )
            legend_y += 44

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
