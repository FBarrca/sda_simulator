Tutorial
========

This tutorial walks through the basic workflow of using the sda-mc-simulator to set up and run a simulation.

Basic Concepts
--------------

The simulator has four main components:

1. **Policy** — Decision-making rule that determines actions based on state
2. **Transition Model** — Defines how the system evolves given a decision and exogenous factors
3. **Exogenous Sampler** — Generates external random factors (noise, demand, etc.)
4. **Metrics** — Tracks and extracts statistics from simulation runs

Running a Simulation
--------------------

Here's a minimal example:

.. code-block:: python

    from sda_mc.simulator import MonteCarloSimulator
    from sda_mc.policy import BasePolicy

    # Define a simple policy
    class MyPolicy(BasePolicy):
        def decide(self, state, exogenous):
            # Your decision logic here
            return action

    # Create a simulator instance
    simulator = MonteCarloSimulator(
        policy=MyPolicy(),
        transition_model=my_model,
        exogenous_sampler=my_sampler,
        num_periods=100,
        num_replications=1000
    )

    # Run the simulation
    results = simulator.run()

    # Extract metrics
    metrics = simulator.extract_metrics()

Examples
--------

See the ``examples/`` directory in the repository for complete working examples:

- Inventory management with stochastic demand
- Supply chain network optimization
- Multi-period decision problems

For more details, refer to the :doc:`api` documentation and the :doc:`background` section on sequential decision problems.
