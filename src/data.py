"""Public trajectory container used by SC-PCP calibration."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor


@dataclass(frozen=True)
class TrajectoryBatch:
    r"""Aligned sequential treatment observations.

    ``states[i, t]`` is \(S_t\), ``actions[i, t]`` is \(A_t\), and
    ``outcomes[i, t]`` is \(Y_{t+1}\).  The class intentionally has no path
    maximum: the main SC-PCP formulation is strictly per-step.
    """

    states: Tensor
    actions: Tensor
    outcomes: Tensor
    patient_ids: Tensor

    def __post_init__(self) -> None:
        if self.states.ndim != 3 or self.actions.ndim != 2 or self.outcomes.ndim != 3:
            raise ValueError("states, actions, and outcomes must be [N,T+1,D], [N,T], [N,T,Y]")
        n, horizon = self.actions.shape
        if self.states.shape[:2] != (n, horizon + 1):
            raise ValueError("states must have one more time point than actions")
        if self.outcomes.shape[:2] != (n, horizon):
            raise ValueError("outcomes must align with actions")
        if self.patient_ids.shape != (n,):
            raise ValueError("patient_ids must have shape [N]")

    @property
    def n(self) -> int:
        return self.actions.shape[0]

    @property
    def horizon(self) -> int:
        return self.actions.shape[1]

    @property
    def state_dim(self) -> int:
        return self.states.shape[-1]

    @property
    def outcome_dim(self) -> int:
        return self.outcomes.shape[-1]

    def current_states(self) -> Tensor:
        return self.states[:, :-1]

    def flat_transitions(self) -> tuple[Tensor, Tensor, Tensor]:
        return (
            self.current_states().reshape(-1, self.state_dim),
            self.actions.reshape(-1),
            self.outcomes.reshape(-1, self.outcome_dim),
        )

    def subset(self, indices: Tensor) -> "TrajectoryBatch":
        return TrajectoryBatch(
            **{field.name: getattr(self, field.name)[indices] for field in fields(self)}
        )

    def prefix(self, horizon: int) -> "TrajectoryBatch":
        if not 1 <= horizon <= self.horizon:
            raise ValueError("prefix horizon is out of range")
        return TrajectoryBatch(
            states=self.states[:, : horizon + 1],
            actions=self.actions[:, :horizon],
            outcomes=self.outcomes[:, :horizon],
            patient_ids=self.patient_ids,
        )

    def to(self, device: str | torch.device) -> "TrajectoryBatch":
        return TrajectoryBatch(
            **{field.name: getattr(self, field.name).to(device) for field in fields(self)}
        )
