from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples" / "logistics"))

from sda_mc import HistoricalBootstrapSampler, Simulator, SimulatorConfig
from data import initial_state, synthetic_history
from policies import GreedyPolicy
from transition import logistics_transition, reward_completed_minus_late


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
