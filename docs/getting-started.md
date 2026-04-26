# Getting Started

To model a sequential decision problem, define three pieces:

```text
transition  = how the system changes after a decision and new information
policy      = which decision to make from the current state
sampler     = how new exogenous information is generated
```

The simulator runs them in this order:

```python
decision_t = policy.decide(state_t)
exogenous_{t+1} = sampler.sample(state_t, t)
next_state_{t+1} = transition(state_t, decision_t, exogenous_{t+1})
```

## 1. Define the transition function

The transition function is the system dynamics, or "business physics", of your
model. It receives the current state, the decision that has already been made,
and the exogenous information that was revealed after that decision. It returns
the next state:

```python
def transition(state: State, decision: Decision, exogenous: ExogenousInfo) -> State:
    ...
    return next_state
```

Keep the transition separate from the policy. The transition should not decide
what to do. It should model what happens because a decision was made.

For example, in a logistics model, the transition might:

- apply accepted order assignments from the decision
- update inventory and vehicle loads
- use sampled travel times and vehicle availability
- add newly sampled customer orders
- advance the clock to the next decision epoch

This answers: given the decision and the sampled external information, how does
the state change?

## 2. Define one or more policies

A policy is the decision rule. It looks at the current state and returns a
decision:

```python
class MyPolicy:
    name = "my_policy"

    def decide(self, state: State) -> Decision:
        ...
        return decision
```

The policy creates the state transition by choosing an action, but it does not
apply the transition itself. This makes it easy to compare many policies against
the same transition model and the same sampled uncertainty.

Policies can be simple heuristics, greedy rules, priority rules, optimization
models, ML models, or RL policies. They only need to implement `decide(state)`.

## 3. Define the exogenous information sampler

The sampler generates information that is outside the policy's control. This is
the uncertainty revealed after the policy makes a decision, such as demand,
travel times, prices, outages, arrivals, cancellations, or weather.

The most common starting point is a data-driven sampler based on historical
traces:

```python
from sda_mc import HistoricalBootstrapSampler

sampler = HistoricalBootstrapSampler(history, block_size=1, seed=7)
```

Use `block_size > 1` when consecutive records should stay together because they
contain temporal correlation:

```python
sampler = HistoricalBootstrapSampler(history, block_size=7, seed=7)
```

You can also provide a custom sampler with `ScenarioSampler`:

```python
from sda_mc import ScenarioSampler

sampler = ScenarioSampler(
    lambda *, state, t, replication: ExogenousInfo(...)
)
```

The sampler should describe how uncertainty is generated. It should not decide
which action to take, and it should not apply the state transition.

## 4. Run the simulator

Once you have a transition function, a policy, and a sampler, wire them into the
simulator:

```python
from sda_mc import Simulator, SimulatorConfig

simulator = Simulator(
    transition=transition,
    sampler=sampler,
    reward_fn=reward_fn,
)

trajectories = simulator.run(
    initial_state=initial_state,
    policy=MyPolicy(),
    config=SimulatorConfig(horizon=30, replications=100),
)
```

Each trajectory records the state, decision, sampled exogenous information, next
state, and reward for every step. Use those trajectories to compare policies
under the same model assumptions.
