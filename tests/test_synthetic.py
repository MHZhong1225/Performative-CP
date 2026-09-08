from __future__ import annotations

import torch

from synthetic import DEFAULT_SEEDS, SyntheticConfig, build_synthetic_problem


def test_default_synthetic_contract_matches_the_paper_setting() -> None:
    config = SyntheticConfig()

    assert config.gamma == -4.0
    assert config.horizon == 12
    assert config.radius_low == 1.0
    assert config.radius_high == 1.8
    assert config.gamma_difficulty_coefficient == 1.0
    assert config.gamma_tail_coefficient == 1.5
    assert config.tail_scale == 6.0
    assert DEFAULT_SEEDS == tuple(range(370_000, 370_200, 10))


def test_synthetic_problem_has_two_outcomes_and_radius_responsive_policy() -> None:
    problem = build_synthetic_problem(seed=370_000, samples=128)

    states = problem.trajectories.states[:, 0]
    low = problem.target_policy.probabilities(states, problem.config.radius_low)
    high = problem.target_policy.probabilities(states, problem.config.radius_high)

    assert problem.trajectories.outcomes.shape == (128, 12, 2)
    assert problem.scores.shape == (128, 12)
    assert problem.stage_grids.shape == (12, 101)
    assert not torch.equal(low, high)
