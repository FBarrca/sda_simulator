from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class Policy[StateT, DecisionT](Protocol):
    """A decision rule. The simulator calls this; it does not own decision logic."""

    name: str

    def decide(self, state: StateT) -> DecisionT: ...


class RewardFn[StateT](Protocol):
    """Immediate reward or cost function for a simulator step.

    The reward function is also called the contribution function, C(s_t, x_t).
    It returns the immediate reward or cost, using the model's sign convention,
    at time t.
    """

    def __call__(self, state: StateT, next_state: StateT) -> float: ...


class ExogenousSampler[StateT, ExogenousT](Protocol):
    """Samples information revealed after a decision."""

    def reset(self, replication: int) -> None: ...

    def sample(self, state: StateT, t: int) -> ExogenousT: ...


class Transition[StateT, DecisionT, ExogenousT](Protocol):
    """Business physics: S_{t+1} = S_M(S_t, x_t, W_{t+1})."""

    def __call__(self, state: StateT, decision: DecisionT, exogenous: ExogenousT) -> StateT: ...


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
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Trajectory[StateT, DecisionT, ExogenousT]:
    replication: int
    policy_name: str
    steps: list[StepRecord[StateT, DecisionT, ExogenousT]]
    total_reward: float
    final_state: StateT
