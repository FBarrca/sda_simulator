Background
==========

Sequential Decision Problems
-----------------------------

A sequential decision problem involves making a sequence of decisions over time where:

- Each decision affects the future state of the system
- Future decisions depend on outcomes of current decisions plus exogenous (external) factors
- We want to find decisions that optimize some objective (minimize cost, maximize profit, etc.)

Mathematical Framework
~~~~~~~~~~~~~~~~~~~~~~

In each period :math:`t = 1, 2, \ldots, T`:

1. Observe the current state :math:`s_t` (system condition: inventory level, asset prices, etc.)
2. Make a decision :math:`d_t = \pi(s_t)` using policy :math:`\pi` (what action to take)
3. Exogenous factors :math:`\omega_t` occur (external shocks: demand spikes, equipment failures, etc.)
4. System transitions to new state :math:`s_{t+1} = f(s_t, d_t, \omega_t)` (the outcome)
5. Incur cost/reward :math:`c_t = c(s_t, d_t, \omega_t)` (immediate consequence of the decision)

The goal is to find a policy that minimizes total expected cost (or maximizes total expected reward).

Concrete Example: Inventory Management
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Imagine managing a warehouse:

- **State** :math:`s_t` = current inventory level (units on hand)
- **Decision** :math:`d_t` = how many units to order this period
- **Exogenous** :math:`\omega_t` = customer demand (random)
- **Transition** :math:`s_{t+1}` = :math:`s_t + d_t - \omega_t` (inventory changes by orders minus demand)
- **Cost** = ordering cost (proportional to :math:`d_t`) + holding cost (inventory :math:`s_t`) + stockout penalty (if demand exceeds supply)

A policy might be: "If inventory drops below 50 units, order 100. Otherwise, order nothing." Simulating this policy many times with random demand lets you estimate total expected cost and compare it against alternatives.

Monte Carlo Simulation
---------------------

Since sequential decision problems often have complex dynamics and many exogenous factors, we typically cannot solve them analytically. Instead, we use Monte Carlo simulation:

1. Sample random exogenous factors :math:`\omega_t` from their distributions
2. Apply the policy :math:`\pi` at each period to decide :math:`d_t`
3. Track metrics (costs, inventory levels, service levels, etc.)
4. Repeat many times to estimate expected outcomes
5. Compare different policies based on aggregate metrics

This avoids the need to analytically compute probabilities over all possible futures — we just simulate many possible futures and average the results.

This Library
~~~~~~~~~~~~

This simulator provides a flexible, pluggable framework for this workflow:

- **Policies** (the decision function :math:`\pi`) can range from simple rule-based heuristics ("if inventory < 50, order 100") to learned models (neural networks trained on historical data). Policies are where you encode your strategy.
- **Transition Models** (the function :math:`f`) represent your system dynamics. Different domains (supply chain, resource allocation, finance) have different rules for how state evolves. The framework lets you plug in custom models.
- **Exogenous Samplers** generate random events :math:`\omega_t`. They support various distributions (normal, Poisson, custom) and can model dependencies between periods (e.g., autocorrelated demand).
- **Metrics Extraction** computes statistics from simulation runs: average costs, confidence intervals, service levels, and other KPIs to compare policies fairly.

Types of Policies
~~~~~~~~~~~~~~~~~~

Warren Powell's *unified framework* for sequential decision problems shows that every
policy — from simple rules of thumb to deep reinforcement learning — falls into one of
**four fundamental classes**. These classes group into two broad strategies: *policy
search* (tune a policy to work well on average) and *lookahead approximations*
(estimate the downstream value of a decision).

**1. Policy Function Approximations (PFAs)**

An analytic function that maps the state directly to a decision, :math:`d_t = f^{\pi}(s_t \mid \theta)`,
with tunable parameters :math:`\theta`. No optimization is solved at decision time.

- *Forms:* lookup tables, rules, parametric/linear functions, neural networks.
- *Inventory example:* the **order-up-to (s, S) policy** — "if inventory drops below
  :math:`s`, order enough to reach :math:`S`." Here :math:`\theta = (s, S)` is tuned by
  simulation. Fast and interpretable, but you must already know the policy's structure.

**2. Cost Function Approximations (CFAs)**

Optimize a *parameterized, deterministic approximation* of the immediate cost,
:math:`d_t = \arg\min_{d} \bar{C}^{\pi}(s_t, d \mid \theta)`. The approximation adds
tunable bonuses or penalties that implicitly account for uncertainty.

- *Inventory example:* order to meet forecast demand **plus** a safety-stock buffer
  :math:`\theta`, where :math:`\theta` is tuned so that occasional stockouts are penalized.
  Widely used in practice because it reuses existing deterministic optimizers.

**3. Value Function Approximations (VFAs)**

Choose the decision that minimizes immediate cost **plus** an approximation of the
downstream value, :math:`d_t = \arg\min_{d}\left( C(s_t, d) + \mathbb{E}\,\bar{V}(s_{t+1}) \right)`.
This is the world of Bellman's equation, approximate dynamic programming, and Q-learning.

- *Inventory example:* learn the expected future cost :math:`\bar{V}` of carrying each
  inventory level, then order to balance ordering cost against that learned future cost.
  Powerful, but estimating :math:`\bar{V}` accurately is the hard part.

**4. Direct Lookahead Approximations (DLAs)**

Make a decision by explicitly *solving an approximate model of the future* over a horizon,
rather than approximating its value. This is what the ``LookaheadRolloutPolicy`` in the
logistics example does.

- *Deterministic lookahead:* replace random demand with a forecast and optimize the next
  :math:`H` periods (model predictive control).
- *Stochastic lookahead:* sample future scenarios and optimize across them — rollout,
  Monte Carlo tree search, stochastic programming.
- *Inventory example:* simulate the next several weeks of demand under each candidate order
  quantity and pick the one with the lowest projected total cost. Most accurate, most
  expensive.

.. note::

   No single class dominates — the best choice depends on problem structure, available
   data, and compute budget. A central use of this simulator is to implement policies from
   different classes and compare them on the **same** sampled exogenous scenarios.

Applications
~~~~~~~~~~~~

Sequential decision problems appear in many domains:

- **Inventory Management** — When to order, how much to stock
- **Supply Chain** — Routing, procurement, network design
- **Revenue Management** — Pricing, capacity allocation
- **Portfolio Optimization** — Asset allocation over time
- **Control Systems** — Temperature control, resource allocation

References
----------

For more information on sequential decision problems and stochastic optimization, see:

- Powell, W. B. (2022). *Reinforcement Learning and Stochastic Optimization: A Unified Framework for Sequential Decisions* — source of the four-class policy taxonomy above
- Powell, W. B. (2007). *Approximate Dynamic Programming: Solving the Curses of Dimensionality*
- Bertsekas, D. P. (2005). *Dynamic Programming and Optimal Control*
- Puterman, M. L. (1994). *Markov Decision Processes: Discrete Stochastic Dynamic Programming*
