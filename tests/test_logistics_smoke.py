from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples" / "logistics"))

from sda_mc import HistoricalBootstrapSampler, Simulator, SimulatorConfig  # noqa: E402
from data import initial_state, synthetic_history  # noqa: E402
from policies import GreedyPolicy, LookaheadRolloutPolicy  # noqa: E402
from transition import logistics_transition, reward_completed_minus_late  # noqa: E402


def _initial_state_for_replication(_replication: int):
    return initial_state()


def test_simulator_runs():
    simulator = Simulator(
        transition=logistics_transition,
        sampler=HistoricalBootstrapSampler(synthetic_history(10), block_size=2, seed=1),
        reward_fn=reward_completed_minus_late,
    )
    trajectories = simulator.run(
        initial_state=lambda _r: initial_state(),
        policy=GreedyPolicy(),
        config=SimulatorConfig(horizon=5, replications=3),
    )
    assert len(trajectories) == 3
    assert all(len(t.steps) == 5 for t in trajectories)


def test_parallel_simulator_matches_sequential_for_deterministic_policy():
    history = synthetic_history(10)
    sequential_simulator = Simulator(
        transition=logistics_transition,
        sampler=HistoricalBootstrapSampler(history, block_size=2, seed=1),
        reward_fn=reward_completed_minus_late,
    )
    parallel_simulator = Simulator(
        transition=logistics_transition,
        sampler=HistoricalBootstrapSampler(history, block_size=2, seed=1),
        reward_fn=reward_completed_minus_late,
    )

    sequential = sequential_simulator.run(
        initial_state=_initial_state_for_replication,
        policy=GreedyPolicy(),
        config=SimulatorConfig(horizon=5, replications=3, parallel=False),
    )
    parallel = parallel_simulator.run(
        initial_state=_initial_state_for_replication,
        policy=GreedyPolicy(),
        config=SimulatorConfig(horizon=5, replications=3, parallel=True, max_workers=2),
    )

    assert [trajectory.replication for trajectory in parallel] == [0, 1, 2]
    assert [trajectory.replication for trajectory in parallel] == [
        trajectory.replication for trajectory in sequential
    ]
    assert [len(trajectory.steps) for trajectory in parallel] == [
        len(trajectory.steps) for trajectory in sequential
    ]
    assert [trajectory.total_reward for trajectory in parallel] == [
        trajectory.total_reward for trajectory in sequential
    ]
    assert [trajectory.final_state for trajectory in parallel] == [
        trajectory.final_state for trajectory in sequential
    ]


def test_lookahead_policy_receives_model_and_plans():
    sampler = HistoricalBootstrapSampler(synthetic_history(10), block_size=2, seed=1)
    stepping_sampler_id = id(sampler)
    policy = LookaheadRolloutPolicy(scenarios=2, horizon=3)
    simulator = Simulator(
        transition=logistics_transition,
        sampler=sampler,
        reward_fn=reward_completed_minus_late,
    )

    trajectories = simulator.run(
        initial_state=_initial_state_for_replication,
        policy=policy,
        config=SimulatorConfig(horizon=5, replications=2, parallel=False),
    )

    assert len(trajectories) == 2
    assert all(len(t.steps) == 5 for t in trajectories)
    # The simulator injected a lookahead model whose sampler is an independent
    # copy, so planning rollouts cannot consume the stepping sampler's stream.
    assert policy.model is not None
    assert policy.model.transition is logistics_transition
    assert id(policy.model.sampler) != stepping_sampler_id


def test_simulator_rejects_non_positive_max_workers():
    simulator = Simulator(
        transition=logistics_transition,
        sampler=HistoricalBootstrapSampler(synthetic_history(10), block_size=2, seed=1),
        reward_fn=reward_completed_minus_late,
    )

    with pytest.raises(ValueError, match="max_workers must be greater than 0"):
        simulator.run(
            initial_state=_initial_state_for_replication,
            policy=GreedyPolicy(),
            config=SimulatorConfig(horizon=5, replications=3, parallel=True, max_workers=0),
        )
