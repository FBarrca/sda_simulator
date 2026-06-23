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


@dataclass(frozen=True)
class LookaheadModel[StateT, DecisionT, ExogenousT]:
    """The base model handed to direct-lookahead (DLA) policies for planning.

    A DLA policy — rolling-horizon, MPC, stochastic-tree lookahead — chooses
    x_t by rolling the system forward: drawing future information W from the
    sampler, evolving the state with the transition, and scoring paths with the
    contribution function. Powell calls this the lookahead model; it is usually
    a (possibly simplified) copy of the base model.

    The simulator constructs one of these and injects it into any policy that
    exposes `set_lookahead_model`, so a DLA policy never has to re-import or
    reconstruct the system's physics. The `sampler` is an *independent* copy of
    the simulator's own sampler: a policy can draw planning scenarios from it
    without perturbing the realized sample path. The policy owns it and should
    reset/seed it as needed before drawing scenarios.
    """

    transition: Transition[StateT, DecisionT, ExogenousT]
    sampler: ExogenousSampler[StateT, ExogenousT]
    reward_fn: RewardFn[StateT, DecisionT, ExogenousT]
    post_decision: PostDecisionFn[StateT, DecisionT] | None = None


class LookaheadPolicy[StateT, DecisionT, ExogenousT](Protocol):
    """A policy that plans against a lookahead model handed to it.

    The simulator detects `set_lookahead_model` and calls it once per
    replication, before stepping, so the policy can roll the model forward
    inside `decide`. Policies that only need the current state implement the
    plain `Policy` protocol and are left untouched.
    """

    name: str

    def decide(self, state: StateT) -> DecisionT: ...

    def set_lookahead_model(
        self, model: LookaheadModel[StateT, DecisionT, ExogenousT]
    ) -> None: ...


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
