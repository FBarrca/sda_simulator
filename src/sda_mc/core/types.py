from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class Policy[StateT, DecisionT](Protocol):
    """A decision rule. The simulator calls this; it does not own decision logic."""

    name: str

    def decide(self, state: StateT) -> DecisionT: ...


class RewardFn[StateT, DecisionT, ExogenousT](Protocol):
    """Immediate contribution function, C(S_t, x_t, W_{t+1}).

    Returns the immediate reward or cost of taking decision x_t in state S_t,
    optionally using the exogenous information W_{t+1} revealed afterward, with
    the model's sign convention. The contribution depends on the decision, not
    on the next state: anything a model needs is derivable from these inputs,
    since S_{t+1} is itself a function of (S_t, x_t, W_{t+1}).
    """

    def __call__(self, state: StateT, decision: DecisionT, exogenous: ExogenousT) -> float: ...


class ExogenousSampler[StateT, ExogenousT](Protocol):
    """Samples information revealed after a decision."""

    def reset(self, replication: int) -> None: ...

    def sample(self, state: StateT, t: int) -> ExogenousT: ...


class Transition[StateT, DecisionT, ExogenousT](Protocol):
    """Business physics: S_{t+1} = S_M(S_t, x_t, W_{t+1})."""

    def __call__(self, state: StateT, decision: DecisionT, exogenous: ExogenousT) -> StateT: ...


class PostDecisionFn[StateT, DecisionT](Protocol):
    """Maps a state and decision to the post-decision state, S^x_t.

    The post-decision state captures the deterministic effect of the decision
    *before* the exogenous information W_{t+1} is revealed:
    S^x_t = S^{M,x}(S_t, x_t). It is the natural argument for value-function
    approximations V(S^x_t) in Powell's VFA policy class, because conditioning
    on it removes the expectation from inside the max.
    """

    def __call__(self, state: StateT, decision: DecisionT) -> StateT: ...


class Objective[StateT](Protocol):
    """Returns a scalar contribution or final score from a state."""

    def __call__(self, state: StateT) -> float: ...


@dataclass(frozen=True)
class StepRecord[StateT, DecisionT, ExogenousT]:
    replication: int
    t: int
    state: StateT
    decision: DecisionT
    exogenous: ExogenousT
    next_state: StateT
    reward: float = 0.0
    post_decision_state: StateT | None = None
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Trajectory[StateT, DecisionT, ExogenousT]:
    replication: int
    policy_name: str
    steps: list[StepRecord[StateT, DecisionT, ExogenousT]]
    total_reward: float
    final_state: StateT
