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

1. Observe the current state :math:`s_t`
2. Make a decision :math:`d_t = \pi(s_t)` using policy :math:`\pi`
3. Exogenous factors :math:`\omega_t` occur (demand shocks, breakdowns, etc.)
4. System transitions to new state :math:`s_{t+1} = f(s_t, d_t, \omega_t)`
5. Incur cost/reward :math:`c_t = c(s_t, d_t, \omega_t)`

The goal is to find a policy that minimizes total expected cost (or maximizes total expected reward).

Monte Carlo Simulation
---------------------

Since sequential decision problems often have complex dynamics and many exogenous factors, we typically use Monte Carlo simulation:

1. Sample random exogenous factors from their distributions
2. Apply the policy at each period
3. Track metrics (costs, inventory levels, service levels, etc.)
4. Repeat many times to estimate expected outcomes
5. Compare different policies based on aggregate metrics

This Library
~~~~~~~~~~~~

This simulator provides a flexible, pluggable framework for this process:

- **Policies** can range from simple heuristics to learned models
- **Transition Models** can represent different types of systems
- **Exogenous Samplers** support various distributions and dependencies
- **Metrics Extraction** provides statistical summaries and confidence intervals

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

- Bertsekas, D. P. (2005). Dynamic Programming and Optimal Control
- Powell, W. B. (2007). Approximate Dynamic Programming
- Puterman, M. L. (1994). Markov Decision Processes
