"""Default signed-feedback synthetic benchmark for SC-PCP.

Prediction radii change the target action policy.  Actions then change future
difficulty and rare outcome tails through a shared, radius-free transition
kernel.  The default is the paper's primary stress endpoint, ``gamma=-4``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from data import TrajectoryBatch


GAMMAS = (-4.0, -2.0, 0.0, 2.0, 4.0)
DEFAULT_SEEDS = tuple(range(370_000, 370_200, 10))


@dataclass(frozen=True)
class SyntheticConfig:
    gamma: float = -4.0
    horizon: int = 12
    initial_difficulty_probability: float = 0.25
    logging_exploration: float = 0.12
    radius_low: float = 1.00
    radius_high: float = 1.80
    response_sigmoid_slope: float = 4.0
    policy_tilt: float = 0.4
    policy_ratio_cap: float = 3.0
    difficulty_intercept: float = -1.60
    difficulty_persistence: float = 2.00
    difficulty_state_effect: float = 0.25
    gamma_difficulty_coefficient: float = 1.0
    tail_intercept: float = -3.20
    tail_difficulty_effect: float = 2.20
    gamma_tail_coefficient: float = 1.5
    tail_scale: float = 6.0
    outcome_correlation: float = 0.30
    state_clip: float = 8.0

    def validate(self) -> None:
        if self.gamma not in GAMMAS:
            raise ValueError(f"gamma must be one of {GAMMAS}")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")


@dataclass(frozen=True)
class SyntheticNoise:
    initial_normals: Tensor
    initial_difficulty_uniforms: Tensor
    action_uniforms: Tensor
    difficulty_uniforms: Tensor
    tail_uniforms: Tensor
    transition_normals: Tensor
    outcome_normals: Tensor


@dataclass(frozen=True)
class SyntheticTrajectory:
    states: Tensor
    actions: Tensor
    outcomes: Tensor


@dataclass(frozen=True)
class SyntheticProblem:
    trajectories: TrajectoryBatch
    scores: Tensor
    stage_grids: Tensor
    outcome_sd: Tensor
    logging_policy: "SyntheticLoggingPolicy"
    target_policy: "SyntheticTargetPolicy"
    outcome_model: "SyntheticOutcomeModel"
    config: SyntheticConfig


def make_noise(*, n: int, horizon: int, seed: int) -> SyntheticNoise:
    generator = torch.Generator().manual_seed(seed)
    options = {"generator": generator, "dtype": torch.float64}
    return SyntheticNoise(
        initial_normals=torch.randn((n, 2), **options),
        initial_difficulty_uniforms=torch.rand(n, **options),
        action_uniforms=torch.rand((n, horizon), **options),
        difficulty_uniforms=torch.rand((n, horizon), **options),
        tail_uniforms=torch.rand((n, horizon), **options),
        transition_normals=torch.randn((n, horizon, 2), **options),
        outcome_normals=torch.randn((n, horizon, 2), **options),
    )


@dataclass(frozen=True)
class SyntheticLoggingPolicy:
    config: SyntheticConfig
    n_actions: int = 3

    def probabilities(self, states: Tensor) -> Tensor:
        z1, z2, difficulty = states[:, 0], states[:, 1], states[:, 2]
        logits = torch.stack(
            (
                0.15 - 0.10 * z1 + 0.15 * z2 - 0.20 * difficulty,
                0.10 * z1 - 0.05 * z2,
                -0.10 + 0.25 * z1 - 0.10 * z2 + 0.35 * difficulty,
            ),
            dim=1,
        )
        base = torch.softmax(logits, dim=1)
        exploration = self.config.logging_exploration
        return (1.0 - exploration) * base + exploration / self.n_actions


@dataclass(frozen=True)
class SyntheticTargetPolicy:
    logging_policy: SyntheticLoggingPolicy

    def probabilities(self, states: Tensor, radius: float | Tensor) -> Tensor:
        reference = self.logging_policy.probabilities(states)
        response = self._response_weight(radius, like=states)
        if response.ndim == 0:
            response = response.expand(len(states))
        coordinate = states.new_tensor((-1.0, 0.0, 1.0))
        weights = torch.exp(
            -self.logging_policy.config.policy_tilt
            * response[:, None]
            * coordinate[None, :]
        )
        return _ratio_capped_tilt(
            reference,
            weights,
            self.logging_policy.config.policy_ratio_cap,
        )

    def probabilities_for_grid(self, states: Tensor, radii: Tensor) -> Tensor:
        return torch.stack(
            [self.probabilities(states, radius) for radius in radii],
            dim=1,
        )

    def _response_weight(self, radius: float | Tensor, *, like: Tensor) -> Tensor:
        config = self.logging_policy.config
        value = torch.as_tensor(radius, dtype=like.dtype, device=like.device)
        normalized = (
            (value - config.radius_low) / (config.radius_high - config.radius_low)
        ).clamp(0.0, 1.0)
        half = config.response_sigmoid_slope / 2.0
        lower = torch.sigmoid(value.new_tensor(-half))
        upper = torch.sigmoid(value.new_tensor(half))
        return (
            torch.sigmoid(config.response_sigmoid_slope * (normalized - 0.5)) - lower
        ) / (upper - lower)


@dataclass(frozen=True)
class SyntheticKernel:
    config: SyntheticConfig

    def difficulty_probability(self, states: Tensor, actions: Tensor) -> Tensor:
        interaction = self._interaction(states, actions)
        logits = (
            self.config.difficulty_intercept
            + self.config.difficulty_persistence * states[:, 2]
            + self.config.difficulty_state_effect * torch.tanh(states[:, 0])
            + self.config.gamma_difficulty_coefficient * interaction
        )
        return torch.sigmoid(logits)

    def tail_probability(self, states: Tensor, actions: Tensor) -> Tensor:
        interaction = self._interaction(states, actions)
        logits = (
            self.config.tail_intercept
            + self.config.tail_difficulty_effect * states[:, 2]
            + self.config.gamma_tail_coefficient * interaction
        )
        return torch.sigmoid(logits)

    def prediction_mean(self, states: Tensor, actions: Tensor) -> Tensor:
        difficulty_probability = self.difficulty_probability(states, actions)
        z1, z2 = states[:, 0], states[:, 1]
        return torch.stack(
            (
                0.78 * z1 + 0.12 * z2 + 0.55 * difficulty_probability,
                0.08 * z1 + 0.72 * z2 + 0.40 * difficulty_probability,
            ),
            dim=1,
        )

    def _interaction(self, states: Tensor, actions: Tensor) -> Tensor:
        coordinate = states.new_tensor((-1.0, 0.0, 1.0))[actions.to(torch.long)]
        return self.config.gamma * coordinate * states[:, 2]


class SyntheticOutcomeModel:
    def __init__(self, kernel: SyntheticKernel, scale: Tensor) -> None:
        self.kernel = kernel
        self.scale = scale

    def __call__(self, states: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        mean = self.kernel.prediction_mean(states, actions)
        return mean, self.scale.to(states)[None, :].expand(len(states), -1)


def rollout(
    kernel: SyntheticKernel,
    policy: SyntheticLoggingPolicy | SyntheticTargetPolicy,
    noise: SyntheticNoise,
    *,
    radii: Tensor | None = None,
) -> SyntheticTrajectory:
    config = kernel.config
    difficulty = noise.initial_difficulty_uniforms.lt(
        config.initial_difficulty_probability
    ).to(noise.initial_normals)
    state = torch.cat(
        (
            noise.initial_normals,
            difficulty[:, None],
            torch.zeros_like(difficulty)[:, None],
        ),
        dim=1,
    )
    states = [state]
    actions = []
    outcomes = []
    for stage in range(config.horizon):
        if isinstance(policy, SyntheticTargetPolicy):
            if radii is None:
                raise ValueError("target rollout requires one radius per stage")
            probabilities = policy.probabilities(state, radii[stage])
        else:
            probabilities = policy.probabilities(state)
        action = _inverse_cdf_actions(probabilities, noise.action_uniforms[:, stage])
        next_difficulty = noise.difficulty_uniforms[:, stage].lt(
            kernel.difficulty_probability(state, action)
        ).to(state)
        tail = noise.tail_uniforms[:, stage].lt(
            kernel.tail_probability(state, action)
        ).to(state)
        z1, z2 = state[:, 0], state[:, 1]
        transition = noise.transition_normals[:, stage]
        next_z1 = 0.78 * z1 + 0.12 * z2 + 0.30 * next_difficulty + 0.25 * transition[:, 0]
        next_z2 = (
            0.08 * z1
            + 0.72 * z2
            + 0.20 * next_difficulty
            + 0.20 * (0.30 * transition[:, 0] + 0.9539392014 * transition[:, 1])
        )
        continuous = torch.stack((next_z1, next_z2), dim=1).clamp(
            -config.state_clip,
            config.state_clip,
        )
        outcome_noise = noise.outcome_normals[:, stage]
        correlated_noise = torch.stack(
            (
                outcome_noise[:, 0],
                config.outcome_correlation * outcome_noise[:, 0]
                + (1.0 - config.outcome_correlation**2) ** 0.5 * outcome_noise[:, 1],
            ),
            dim=1,
        )
        multiplier = 1.0 + (config.tail_scale - 1.0) * tail
        outcome_mean = continuous + torch.stack(
            (0.25 * next_difficulty, 0.20 * next_difficulty),
            dim=1,
        )
        outcome = (
            outcome_mean
            + multiplier[:, None]
            * state.new_tensor((0.35, 0.25))[None, :]
            * correlated_noise
        )
        next_time = torch.full_like(next_difficulty, (stage + 1) / config.horizon)
        state = torch.cat(
            (continuous, next_difficulty[:, None], next_time[:, None]),
            dim=1,
        )
        states.append(state)
        actions.append(action)
        outcomes.append(outcome)
    return SyntheticTrajectory(
        states=torch.stack(states, dim=1),
        actions=torch.stack(actions, dim=1),
        outcomes=torch.stack(outcomes, dim=1),
    )


def build_synthetic_problem(
    *,
    seed: int,
    samples: int = 3_000,
    horizon: int = 12,
    gamma: float = -4.0,
) -> SyntheticProblem:
    config = SyntheticConfig(gamma=gamma, horizon=horizon)
    config.validate()
    logging_policy = SyntheticLoggingPolicy(config)
    target_policy = SyntheticTargetPolicy(logging_policy)
    kernel = SyntheticKernel(config)
    trajectory = rollout(
        kernel,
        logging_policy,
        make_noise(n=samples, horizon=horizon, seed=seed + 101),
    )
    current_states = trajectory.states[:, :-1]
    flat_states = current_states.reshape(-1, current_states.shape[-1])
    flat_actions = trajectory.actions.reshape(-1)
    predicted_mean = kernel.prediction_mean(flat_states, flat_actions).reshape_as(
        trajectory.outcomes
    )
    residual = trajectory.outcomes - predicted_mean
    global_scale = residual.square().mean(dim=(0, 1)).sqrt().clamp_min(1e-6)
    scores = (residual.abs() / global_scale[None, None, :]).amax(dim=2)
    probabilities = torch.linspace(0.75, 0.995, 101, dtype=scores.dtype)
    stage_grids = torch.stack(
        [torch.quantile(scores[:, stage], probabilities) for stage in range(horizon)]
    )
    batch = TrajectoryBatch(
        states=trajectory.states,
        actions=trajectory.actions,
        outcomes=trajectory.outcomes,
        patient_ids=torch.arange(samples),
    )
    return SyntheticProblem(
        trajectories=batch,
        scores=scores,
        stage_grids=stage_grids,
        outcome_sd=trajectory.outcomes.reshape(-1, 2).std(dim=0),
        logging_policy=logging_policy,
        target_policy=target_policy,
        outcome_model=SyntheticOutcomeModel(kernel, global_scale),
        config=config,
    )


def _inverse_cdf_actions(probabilities: Tensor, uniforms: Tensor) -> Tensor:
    return (uniforms[:, None] > probabilities.cumsum(dim=1)).sum(dim=1).clamp_max(2)


def _ratio_capped_tilt(reference: Tensor, weights: Tensor, cap: float) -> Tensor:
    lower = torch.zeros_like(weights[:, :1])
    upper = weights.amax(dim=1, keepdim=True).clamp_min(1e-12)
    for _ in range(64):
        normalizer = (lower + upper) / 2.0
        ratio = torch.minimum(weights / normalizer, torch.full_like(weights, cap))
        mass = (reference * ratio).sum(dim=1, keepdim=True)
        lower = torch.where(mass > 1.0, normalizer, lower)
        upper = torch.where(mass > 1.0, upper, normalizer)
    ratio = torch.minimum(
        weights / ((lower + upper) / 2.0),
        torch.full_like(weights, cap),
    )
    return reference * ratio


__all__ = [
    "DEFAULT_SEEDS",
    "GAMMAS",
    "SyntheticConfig",
    "SyntheticProblem",
    "build_synthetic_problem",
]
