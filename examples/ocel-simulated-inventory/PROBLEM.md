# Simulated Inventory Management — Problem Description

## What we are solving

We run the materials inventory of a multi-plant manufacturer. Across **5
plants** we stock **100 materials** — raw, semi-finished, and finished goods —
that **50 customers** order and **29 suppliers** replenish. Every day customer
**sales orders** arrive for a given material at a given plant, and replenishment
**purchase orders** placed earlier arrive from suppliers after a lead time.

The tension is the classic inventory one. Stock is finite and refilling it is
not instant: a purchase order takes days to receive, so what we can fulfil today
is bounded by what we ordered days ago. Demand we cannot cover from on-hand
stock becomes **backlog** — unhappy customers waiting; stock we hold but do not
need ties up cash as **overstock**. We cannot eliminate both at once, so the
job is to keep the right amount of the right material at the right plant as
demand and supply timing shift underneath us.

The decision we make each day is a combination of three levers: **reorder**
(open or grow a purchase order to refill a low inventory position),
**expedite** (pull an open purchase order's receipt date earlier), and
**allocate** (assign available on-hand stock to known backlog). The goal is to
find a *replenishment-and-allocation policy* that maximizes fulfilled demand
while minimizing backlog and overstock — under uncertainty about future demand
and supply timing.

This is framed as a **Sequential Decision Analytics (SDA)** problem and
evaluated by Monte Carlo simulation: we run candidate policies over many
sampled futures and compare their outcome distributions, not just point
estimates.

## The data

The world is reconstructed from a synthetic but **SAP-style ERP dataset** in
`data/`. We have the familiar order tables — `SalesOrderDocuments` /
`SalesOrderItems` (demand), `PurchaseOrderDocuments` / `PurchaseOrderItems` and
`PurchaseRequisitions` (the supply pipeline), `Materials`, `MaterialStocks`
(on-hand positions), `GoodsReceiptsAndIssues` (stock movements), and
`OrderSuggestions` (reorder suggestions). These are also flattened into an
**Object-Centric Event Log (OCEL)**, where each event (*Create Purchase Order
Item*, *Goods Receipt*, *Goods Issue*, …) is linked to the objects it touches
(material, plant, purchase-order item, sales-order item, customer, supplier).
The dataset spans roughly **one year** (2023-09-26 → 2024-09-25).

### Inspecting the data

Before running any policy, it helps to look at what the year of history actually
contains. The helper script renders it as a two-panel figure:

```bash
python3 visualize_data.py
```

![Historical demand vs supply, and demand by priority tier](inventory_demand_supply.png)

It reads the same `load_inventory_simulation_data()` the simulator uses, so the
picture is exactly the world the policies face. Two facts jump out, and both
drive the whole problem:

- **Supply tracks demand, but does not quite cover it (top panel).** Over the
  year, customers order **~30,800 units** while historical goods receipts deliver
  **~21,600** — receipts meet about **70%** of demand, leaving a **~9,200-unit
  shortfall** (the shaded gap). Together with the opening on-hand stock the
  system is *roughly* balanced, so this is not a desperate catch-up: the gap, plus
  the **lumpy timing** of receipts and the mismatch between *which* `(material,
  plant)` has stock and which has orders, is what the reorder, expedite, and
  allocate levers exist to manage. A do-nothing policy still falls behind, but a
  good one has real room to close the gap without drowning in overstock.
- **Demand arrives in three priority tiers (bottom panel).** Each sales order's
  type sets its priority — **Urgent → 3** (~11,300 units), **Normal → 2**
  (~9,700), and everything else (here **Backorder**) **→ 1** (~9,800). Because the
  reward weights both service and backlog by priority, *which* demand a scarce
  unit of stock protects matters as much as how much is served — the lever the
  allocation and MILP policies pull on.

### Extraction (`extract_policy_inputs.py`)

`load_inventory_simulation_data()` reads the database and separates two things
the simulator needs:

- **State inputs** — the opening `inventory` positions (per
  `(material, plant)`, with available stock derived from posting/transfer/
  returns minus blocked), the open `purchase_orders` pipeline (with expected
  receipt dates and planned lead times), plus material/customer/supplier
  profiles.
- **Exogenous history** — a per-day timeline (`DailyInventoryExogenous`) of
  **demand arrivals** (sales-order items posted that day) and **supply
  arrivals** (goods receipts posted that day, linked back to their purchase
  order). This 1-year history is what the sampler later resamples.

The same module also emits flattened analysis tables
(`daily_inventory_levels.csv`, `policy_item_inputs.csv`) with per-day,
per-item state/exogenous/outcome features for offline policy learning.

## The environment

### State (`domain.py`)

The `State` captures everything the policy can observe at a decision epoch:

- `inventory`: available stock keyed by `(material_id, plant_id)`.
- `pipeline`: open `OpenPurchaseOrder`s — outstanding replenishment in transit,
  each with a `quantity_open`, `expected_receipt_date`, and `expedited` flag.
- `backlog`: known-but-unfulfilled `BacklogOrder`s awaiting allocation, each
  with quantity, customer, priority, arrival date, and optional due date.
- `completed_orders`: cumulative fulfilled quantity per order.
- `date` and `time` (the integer step index).

### Decision (`domain.py`, `policy.py`)

A **`Decision`** bundles three optional action sets, all applied in the same
epoch:

- **`ReorderAction`** — create or grow a purchase order for a `(material,
  plant)`, with an `expected_receipt_date` set `lead_time_days` out. Simulated
  reorders carry a `SIM-PO-…` id so the transition can receive them on time.
- **`ExpediteAction`** — move an open purchase order's receipt date earlier
  (capped at the current date), at an `expedite_cost`.
- **`AllocateStockAction`** — commit available inventory to a specific backlog
  order. Allocation is bounded by `min(requested, order_open, on_hand)`.

### Exogenous information (`domain.py`, `sampler.py`)

What the policy does **not** control, revealed *after* its decision each day as
`ExogenousInfo`:

- **Demand arrivals** — new sales-order quantities for `(material, plant)`,
  which become next-epoch backlog. Each arrival carries a **priority** derived
  from its order type (`Urgent → 3`, `Normal → 2`, else `1`) and a **due date**
  set by a priority-dependent SLA (tighter for higher priority), so allocation
  can favor the orders that matter most.
- **Supply arrivals** — realized goods receipts that raise inventory and burn
  down the open pipeline.
- **Supply lead-time shocks** — the timing risk on the orders the policy
  *itself* places. A reorder is planned to arrive after a nominal lead time, but
  the realized receipt slips by a sampled `lead_time_delay_days` (mean ≈ 1.4 days
  on a 7-day lead, occasionally a full week late); an expedite pulls a receipt in
  but cannot beat the requested date by more than the supplier can recover
  (`expedite_delay_days`), so expediting usually — but not always — lands the
  goods on time. The policy observes only the *planned* receipt date; the
  realized one is revealed when the goods actually arrive.

`InventoryHistoricalSampler` drives this. On `reset(replication)` it picks a
reproducible random start day in the 1-year history and then returns
**consecutive days with wraparound** — preserving the observed day-to-day
demand/supply timing within each sample path rather than shuffling it — and
draws the two lead-time shocks above with a fixed number of RNG calls per step,
so every policy faces the *identical* demand and supply-timing stream for a given
seed. The simulation clock advances **monotonically** (one day per step, anchored
at the initial date) independent of the wrapped historical dates, so date-gated
replenishment receipts stay consistent across a year boundary.

### Transition (`transition.py`)

`inventory_transition(state, decision, exogenous)` advances one day, in order:

1. **Apply reorders** — add/grow `SIM-PO-…` orders in the pipeline, stamping
   each with a *realized* receipt date = planned date + the epoch's
   `lead_time_delay_days`.
2. **Apply expedites** — pull receipt dates earlier (for the controllable
   `SIM-PO-…` orders, whose timing the policy can actually move) at a cost,
   capped by the epoch's `expedite_delay_days` so the pull-in is not guaranteed.
3. **Receive supply** — credit any `SIM-PO-…` whose *realized* receipt date has
   arrived (not merely its planned date), then apply realized exogenous goods
   receipts (decrementing the matching open PO).
4. **Apply allocations** — move on-hand stock to backlog orders, recording
   completed quantity and clearing fully-served orders.
5. **Inject new demand** — append the day's demand arrivals to backlog.
6. **Advance the clock**.

Because the simulator calls the policy *before* sampling exogenous info,
allocations can only serve backlog already known at the epoch; same-day demand
must wait for the next decision.

### Reward (`transition.py`)

`reward_stockout_overstock_service` returns, per step, a single contribution
computed from the very same transition the simulator runs:

> **`reward = allocated_value − 3.0 · backlog_value − 0.02 · overstock_quantity − expedite_cost`**

- **Service value** (`+`) — units allocated to demand this step, **weighted by
  priority** (a priority-3 unit is worth 3× a priority-1 unit).
- **Backlog penalty** (`−3.0×`) — the heavy cost of unmet, known demand, also
  **priority-weighted**, so failing high-priority orders hurts most.
- **Overstock penalty** (`−0.02×`) — a small holding cost on every on-hand
  unit, discouraging carrying stock for its own sake.
- **Expedite cost** (`−`) — the fee charged for each pulled-in purchase order,
  so expediting is a genuine tradeoff rather than free speed.

The 3.0-vs-0.02 asymmetry makes a missed sale far more expensive than a unit of
excess inventory, so good policies lean toward availability without hoarding —
while the priority weighting and expedite cost force them to spend scarce stock
and expediting budget where they buy the most service.

## Policies compared (`policy.py`)

| Policy | Idea |
|---|---|
| `NoOpPolicy` | Do nothing — a baseline to measure how much the action levers are worth. |
| `AllocationOnlyPolicy` | Only allocate existing stock to backlog (by priority, then due date and age); never reorder or expedite. Isolates allocation value. |
| `ReorderAllocatePolicy` | Allocate backlog **and** reorder any item whose inventory position (on-hand + pipeline) falls below a reorder point, up to an order-up-to target. |
| `ReorderExpediteAllocatePolicy` | The full baseline: reorder low items, **expedite** open POs for items currently in backlog, and allocate. |
| `AggressiveReorderPolicy` | Same full action set with higher targets (reorder point 130, order-up-to 220, shorter lead time) — carries more inventory to avoid stockouts. |
| `DemandScaledReorderExpeditePolicy` | The **fair** (s, S) baseline: same levers as `ReorderExpediteAllocate`, but each `(material, plant)` gets its *own* reorder point and order-up-to, sized from demand history (lead-time demand + safety stock). Safety stock uses the service level the reward *implies* — the critical ratio `3.0/(3.0+0.02) ≈ 0.993`, z ≈ 2.48 — so it is calibrated to the same objective the MILP optimizes. Items with no demand history are never stocked. |
| `MilpReorderPolicy` | Same allocation + expedite, but reorders are chosen by a budget-constrained MILP instead of fixed thresholds. A 0/1 knapsack (`scipy.optimize.milp`) spends a per-epoch **reorder budget** across competing items to protect the most priority- and urgency-weighted *uncovered* backlog; falls back to `ReorderExpediteAllocate` if scipy is unavailable. |

The first three reordering policies share a **global** (s, S) rule — one
`(reorder_point, order_up_to)` pair applied to every item — which is deliberately
naive: it over-stocks slow movers and is the foil for everything below.
`DemandScaledReorderExpedite` keeps the (s, S) structure but sizes the thresholds
*per item* from demand, the way a competent planner would. The MILP instead
reorders *reactively* — only to cover known backlog that on-hand stock and the
open pipeline cannot, allocating a scarce reorder budget where it buys the most
weighted service. The interesting comparison is the last two: a well-tuned (s, S)
versus a budget optimizer, both of which right-size supply rather than blanket it.

## How it is evaluated (`run.py`)

Each policy is run for a **60-day horizon** over **20 replications** using the
historical sampler (seed 42). The metrics collected are:

- `reward` (↑) — total per-step contribution, reported with a **95% CI** (how
  precisely the mean is known) and its **CVaR (95%)** — the mean reward in the
  *worst 5%* of runs, i.e. how bad a bad month looks.
- `final_backlog` (↓) — open backlog quantity remaining at the horizon, also with
  its upper-tail **CVaR (95%)** (the worst-case leftover backlog).
- `final_inventory` (—) — total on-hand stock left, reported for context (a
  policy can score well by carrying just enough, not the most).

Because outcomes are stochastic, policies are compared on **distributions**, not
a single run: the **CI95** says whether two means are really different, while the
**CVaR95** says how *stable* a policy is — a policy whose worst-case is close to
its mean is dependable, one whose tail blows out is a gamble. Each
`evaluate_metrics` report can also be logged to Weights & Biases.

## Interpreting Results

Running `SDA_MC_WANDB=0 python3 run.py` (60-day horizon, 20 replications, seed
42) produces a table like this. These are **actual numbers** from the example,
so your run should reproduce them closely:

```text
policy                                 reward_mean                reward_ci95  reward_cvar95  backlog_mean  backlog_cvar95  inventory_mean
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
no_action                              -1009335.45  (-1098417.27, -920253.64)    -1340250.73       5081.85         6243.50        17101.41
allocation_only                         -412720.92   (-451140.76, -374301.08)     -576503.68       1952.05         2458.39        13971.61
reorder_allocate                         -84085.04     (-87525.34, -80644.74)     -100783.18        112.05          398.00        50898.80
reorder_expedite_allocate                -79026.23     (-81062.69, -76989.77)      -87756.92        112.05          398.00        50899.00
aggressive_reorder_expedite_allocate    -110139.51  (-112207.30, -108071.72)     -118514.49        112.05          398.00        77652.65
demand_scaled_reorder_expedite_allocate  -77670.51     (-81397.65, -73943.37)      -91291.80        192.11          528.65        16826.28
milp_reorder_budget                      -69992.29     (-73900.06, -66084.52)      -84699.36        179.36          555.32        14157.77
```

(Every reward is negative because each replication starts from a large standing
backlog inherited from the opening state — the ranking, i.e. how much of that
cost each policy claws back, is what matters, not the sign.)

**How to read the columns:**

- **reward_mean** (↑) — priority-weighted service minus priority-weighted
  backlog, overstock holding cost, and expedite fees, summed over the horizon.
  The headline objective.
- **reward_ci95** — 95% confidence interval on reward across the 20 replications.
  Intervals that **don't overlap** mark a real, repeatable difference; ones that
  **overlap** are statistically close.
- **reward_cvar95** (↑, stability) — the mean reward in the **worst 5%** of runs.
  The closer it sits to `reward_mean`, the more *dependable* the policy; a CVaR
  far below the mean means the policy occasionally has a very bad month.
- **backlog_mean** (↓) — open (unfulfilled) demand quantity left at the horizon.
- **backlog_cvar95** (↓, stability) — the worst-5% leftover backlog, i.e. how bad
  the service shortfall gets in a bad scenario.
- **inventory_mean** (—) — on-hand stock left at the horizon. *Not* "more is
  better": stock you carry but don't need is pure overstock cost. Read it
  alongside backlog as the two sides of the tradeoff.

**What this tells you:**

- **Doing nothing falls behind on both fronts.** ``no_action`` ends with 5,082
  units of backlog *and* lets ~17k of arriving supply pile up unused (−1,009k).
  Every other policy is judged by how much of that it recovers.
- **Allocation alone is now a real lever.** Because opening stock sits where
  demand is, ``allocation_only`` clears most backlog just by spending what is on
  hand — backlog 5,082 → 1,952, reward −1,009k → −413k (well over half the
  recovery) without ordering a thing.
- **A blanket reorder target clears backlog but over-stocks — and that is a
  *sizing* artifact, not a flaw of (s, S).** The global-threshold policies drive
  backlog down to ~112, yet because one `order_up_to=140` is applied to every
  item regardless of its demand, inventory piles to **~51k**. Reward lands at
  −84k/−79k: the backlog is gone, but overstock holding cost is now the dominant
  drag. Expedite still earns its cost (``reorder_expedite_allocate`` −79k edges
  ``reorder_allocate`` −84k), and ``aggressive_reorder`` shows hoarding is
  punished — same 112 backlog, but **77k** of stock drags it to −110k.
- **Sizing the (s, S) to demand removes almost all of that overstock.**
  ``demand_scaled_reorder_expedite_allocate`` keeps the exact same levers but
  gives each item its own demand-based threshold (safety stock set to the reward's
  implied ~99.3% service level). Inventory collapses from ~51k to **~17k** — a
  third of the blanket policy, right next to the MILP — proving the overstock was
  never inherent to (s, S), just to applying one target to 500 different items.
  Its reward, **−78k**, is *better* than the blanket policies despite holding far
  less, because under stochastic lead times the lean book still covers demand
  (backlog 192) without paying to warehouse stock nobody orders.
- **The budget MILP still wins — but now only narrowly, over a fair opponent.**
  ``milp_reorder_budget`` posts **−70k** holding just **14k** of inventory. Its CI
  (−73.9k, −66.1k) sits just above the tuned (s, S)'s (−81.4k, −73.9k) — the
  intervals barely touch, so the win is real but *slim*, a far cry from the
  blowout it looked like against the blanket baselines. Both policies right-size
  supply; the MILP's remaining edge comes from reordering *reactively* to the
  priority-weighted backlog it can actually protect, rather than to a
  distribution-based target, shaving the last bit of inventory and backlog.
- **Leanness costs tail stability — for both lean policies.** Read down the
  `reward_cvar95` column: running lean under random lead times means a slipped
  receipt lands as backlog with no buffer to absorb it, so both right-sized
  policies have a wider mean-to-tail gap than their lean means suggest — MILP
  −70k mean / **−85k** worst-5%, tuned (s, S) −78k / **−91k**. The MILP still owns
  the best absolute tail of any policy, and the overstocked baselines are far
  worse in absolute terms (``no_action`` −1,009k → **−1,340k**), but the example
  now shows the real tradeoff honestly: low inventory buys the best worst-case,
  yet not a free one.

**The bottom line for the business:**

- **Use what you have first.** With stock positioned where demand is, disciplined
  allocation recovers more than half the value before a single new order — the
  cheapest lever, and the one most easily overlooked.
- **Right-size before you optimize.** Most of the apparent "optimization win" is
  just sizing: a blanket order-up-to rule carried ~51–77k of stock, but giving
  each item a demand-based threshold cut that to ~17k at *better* reward. The
  cheapest, biggest gain is calibrating reorder levels to demand — not buying a
  solver.
- **The solver's edge is real but marginal once the baseline is fair.** Against a
  reward-calibrated (s, S), the budget MILP wins by only ~8k reward (−70k vs −78k)
  and ~3k inventory, with confidence intervals that barely separate. Reactive
  budget allocation squeezes out the last bit, but a competent (s, S) is most of
  the way there — worth knowing before paying for the heavier machinery.
- **Compare distributions, not single runs.** Confidence intervals show the MILP's
  lead over the tuned (s, S) is slim (barely non-overlapping) while its lead over
  the blanket baselines is decisive, and the CVaR95 tail tells you which policy
  survives a bad month — a single run could not reliably tell an overstock-heavy
  −84k from a −110k, let alone how stable either
  is.

## The objective in one line

> Find a daily reorder/expedite/allocate policy that fulfills as much demand as
> possible across uncertain demand and supply timing, while keeping both
> backlog and excess inventory low.

## What each policy does, and what it gets

Demand is exogenous — every policy faces the identical order stream. What a
policy *controls* is the three levers it pulls each day, and the state and cost
those levers produce. The dashboard lays the whole causal chain out as six
panels — **what it does** (left column) → **what state results** (right top/mid)
→ **the cost outcome** (right bottom):

```bash
python3 visualize_policies.py
```

![What each policy does (levers) and what it gets (state and cost)](inventory_policy_trajectories.png)

Each line is the mean of 20 runs; the two right-sized policies are drawn thick.
Panels whose values span orders of magnitude use a **log** y-axis (marked) so the
lean policies are not crushed against the floor of a linear scale.

- **Units reordered / expedites / allocated (left) — the levers.** The supply a
  policy *injects* is the sharpest tell: ``aggressive`` (red) and the
  global-threshold policies (green/cyan) dump **~50–78k** units into the pipeline
  in the first week and flatline, while the right-sized
  ``demand_scaled`` (orange) and ``milp`` (purple) trickle in only **~2–5k** total,
  reactively. ``no_action``/``allocation_only`` order nothing. The expedite panel
  shows the tuned (s, S) leans on expediting hardest (it runs lean, so it pulls
  receipts in often); allocation is near-identical for every reordering policy —
  they all serve essentially the full ~5k of demand.
- **On-hand inventory / open backlog (right, top & middle) — the state.** On the
  log inventory axis the lean band is finally legible: ``demand_scaled`` (~17k)
  and ``milp`` (~14k) sit clearly *below* the ~51k/78k overstock band. Backlog
  separates the do-little policies (``no_action`` → ~5k, ``allocation_only`` →
  ~2k) from every reordering policy, which all hold backlog near ~100–250.
- **Cost = −reward (right, bottom) — the outcome.** The cumulative cost log axis
  ranks everything at a glance: ``no_action`` ~1M, ``allocation_only`` ~413k, then
  the tight cluster where ``milp`` (70k) edges ``demand_scaled`` (78k) below the
  blanket-target policies (79–110k).

Read together, the dashboard is the whole argument in one picture: the
global-threshold policies buy low backlog by *injecting and holding* far more
supply than they need, while the tuned (s, S) and the MILP reach the same service
with a third of the orders and inventory — and the MILP's lines sit just a hair
below the tuned (s, S) everywhere, exactly the slim, honest margin the reward
table reports.
