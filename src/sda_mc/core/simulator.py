from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

from .types import (
    ExogenousSampler,
    RewardFn,
    Policy,
    StepRecord,
    Trajectory,
    Transition,
)


type StateFactory[StateT] = Callable[[int], StateT]


def _default_max_workers() -> int:
    return max(1, os.cpu_count() or 1)


@dataclass
class SimulatorConfig:
    """Configuration for one call to `Simulator.run`.

    Args:
        horizon: Number of decision epochs in each replication.
        replications: Number of independent sample paths to simulate.
        copy_initial_state: Deep-copy the initial state before each replication.
            Keep this enabled for mutable states. Disable it only when states
            are immutable or when `initial_state` already creates a fresh state.
        parallel: Run independent replications in separate worker threads.
        max_workers: Maximum number of worker threads when `parallel` is enabled.
            Defaults to all detected CPU cores.
        early_stop_no_improvement_rounds: Reserved for higher-level evaluation
            loops that stop after repeated non-improving rounds.
    """

    horizon: int
    replications: int
    copy_initial_state: bool = True
    parallel: bool = True
    max_workers: int | None = None
    early_stop_no_improvement_rounds: int | None = None


class Simulator[StateT, DecisionT, ExogenousT]:
    """A simulator for sequential decision problems under uncertainty.

    Args:
        transition: Callable transition model with signature
            `(state, decision, exogenous) -> next_state`.
        sampler: Exogenous information sampler. It materializes the information
            W_{t+1} revealed after a decision is made, which is then used to
            evolve the state with the transition function.
        reward_fn: Immediate reward or cost function for one simulator step.
            Also called the contribution function, C(s_t, x_t).
    """

    def __init__(
        self,
        *,
        transition: Transition[StateT, DecisionT, ExogenousT],
        sampler: ExogenousSampler[StateT, ExogenousT],
        reward_fn: RewardFn[StateT] | None = None,
    ) -> None:
        self.transition = transition
        self.sampler = sampler
        self.reward_fn = reward_fn or (lambda _previous, _next: 0.0)

    def run(
        self,
        *,
        initial_state: StateT | StateFactory[StateT],
        policy: Policy[StateT, DecisionT],
        config: SimulatorConfig,
    ) -> list[Trajectory[StateT, DecisionT, ExogenousT]]:
        """
        Run the simulator for one policy.

        Args:
            initial_state: The initial state of the system.
            policy: The policy to use to make decisions.
            config: The configuration for the simulation.

        Returns:
            A list of sample paths, each containing a trajectory of the system.
        """
        if config.max_workers is not None and config.max_workers <= 0:
            raise ValueError("max_workers must be greater than 0")

        if config.parallel and config.replications > 1:
            max_workers = config.max_workers or _default_max_workers()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                return list(
                    executor.map(
                        lambda replication: self._run_replication(
                            replication,
                            initial_state=initial_state,
                            policy=policy,
                            config=config,
                            isolate_objects=True,
                        ),
                        range(config.replications),
                    )
                )

        return [
            self._run_replication(
                r,
                initial_state=initial_state,
                policy=policy,
                config=config,
            )
            for r in range(config.replications)
        ]

    def _run_replication(
        self,
        replication: int,
        *,
        initial_state: StateT | StateFactory[StateT],
        policy: Policy[StateT, DecisionT],
        config: SimulatorConfig,
        isolate_objects: bool = False,
    ) -> Trajectory[StateT, DecisionT, ExogenousT]:
        sampler = deepcopy(self.sampler) if isolate_objects else self.sampler
        policy = deepcopy(policy) if isolate_objects else policy
        sampler.reset(replication)
        state = initial_state(replication) if callable(initial_state) else initial_state
        if config.copy_initial_state:
            state = deepcopy(state)

        steps: list[StepRecord[StateT, DecisionT, ExogenousT]] = []
        total_reward = 0.0

        for t in range(config.horizon):
            decision = policy.decide(state)
            exogenous = sampler.sample(state, t)
            next_state = self.transition(state, decision, exogenous)
            reward = self.reward_fn(state, next_state)
            total_reward += reward

            steps.append(
                StepRecord(
                    replication=replication,
                    t=t,
                    state=state,
                    decision=decision,
                    exogenous=exogenous,
                    next_state=next_state,
                    reward=reward,
                )
            )
            state = next_state

        return Trajectory(
            replication=replication,
            policy_name=getattr(policy, "name", policy.__class__.__name__),
            steps=steps,
            total_reward=total_reward,
            final_state=state,
        )
