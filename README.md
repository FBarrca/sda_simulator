# SDA Monte Carlo Simulator

A small Python repository for simulating **sequential decision problems under uncertainty**.

The design follows the structure from the provided pages:

```text
state_t + decision_t + exogenous_information_{t+1} -> state_{t+1}
```

The simulator itself does **not** decide. It only runs the loop:

1. expose the current state to a policy
2. ask the policy for a decision
3. sample exogenous information
4. apply the transition function
5. record the trajectory
6. repeat many times for Monte Carlo evaluation

This keeps the implementation policy-agnostic: random, greedy, priority, optimization-based, ML/RL, and MIP policies can all use the same simulator.

## Core idea

```python
for t in horizon:
    decision = policy.decide(state)
    exogenous = sampler.sample(state, t)
    next_state = transition(state, decision, exogenous)
```

The simulator is intentionally separated from the policy:

```text
Simulator = business physics
Policy    = decision logic
Sampler   = uncertainty generator
```

## Repository layout

```text
src/sda_mc/
  core.py              # protocols and trajectory records
  simulator.py         # Monte Carlo simulation engine
  samplers.py          # bootstrap and random samplers
  evaluation.py        # objective aggregation and confidence intervals
  tracking.py          # optional W&B-compatible tracking hook
examples/logistics/
  domain.py            # State, Order, Vehicle, Warehouse model
  policies.py          # Random, greedy, and priority policies
  transition.py        # logistics transition function
  data.py              # synthetic historical data generator
  run.py               # runnable example
```

## Install

```bash
uv venv
uv pip install -e .
```

or:

```bash
pip install -e .
```

## Getting started

Read the [getting started guide](docs/getting-started.md) to define a transition
function, policy, sampler, and simulator run.

## Run the logistics example

```bash
python examples/logistics/run.py
```

## What the example models

The included logistics example mirrors the pages in your photos:

- state contains inventory, vehicles, pending orders, completed orders, time, and day of week
- policies assign pending orders to feasible vehicles
- the transition function applies decisions, moves vehicles, adds new orders, and advances time
- the exogenous sampler bootstraps historical day records instead of fitting a distribution

## Bootstrap sampling

The default sampler supports replaying historical records:

```python
sampler = HistoricalBootstrapSampler(history, block_size=1, seed=7)
```

Use block bootstrap when correlations matter:

```python
sampler = HistoricalBootstrapSampler(history, block_size=7, seed=7)
```

Sampling whole days or blocks preserves realistic patterns such as high-demand days, poor vehicle availability, and high travel times occurring together.

## Limitation

Historical bootstrap only generates scenarios similar to what happened before. It does not create novel extreme scenarios unless those events exist in the historical data. For stress testing, combine it with custom scenario samplers or forecast/error samplers.
