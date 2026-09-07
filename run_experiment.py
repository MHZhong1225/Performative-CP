"""Run SC-PCP from one concise, dataset-level entry point.

Example
-------
python run_experiment.py --dataset synthetic
python run_experiment.py --dataset mimic_iv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data import TrajectoryBatch
from experiments import PerStepCalibrationInputs, calibrate_per_step_marginal


DEFAULT_SEEDS = tuple(range(20))
DATASETS = ("synthetic", "mimic_iv", "eicu", "inspire", "mimic_cxr")


class _UniformLoggingPolicy:
    def probabilities(self, states: torch.Tensor) -> torch.Tensor:
        return torch.full((len(states), 2), 0.5, device=states.device)


class _SyntheticTargetPolicy:
    def probabilities_for_grid(
        self,
        states: torch.Tensor,
        radii: torch.Tensor,
    ) -> torch.Tensor:
        state_effect = 0.20 * torch.tanh(states[:, :1])
        radius_effect = 0.10 * radii[None, :]
        probability_one = torch.sigmoid(-0.35 + state_effect + radius_effect)
        return torch.stack((1.0 - probability_one, probability_one), dim=-1)


class _UnitScaleOutcomeModel:
    def __call__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros((len(states), 1), device=states.device),
            torch.ones((len(states), 1), device=states.device),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the public SC-PCP example")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="run one seed; omit it to run the default 20-seed experiment",
    )
    parser.add_argument("--samples", type=int, default=1_000)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--workers-per-device", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.dataset != "synthetic":
        _run_private_dataset(args, parser)
        return
    if args.samples < 100:
        parser.error("--samples must be at least 100")
    if args.horizon < 1:
        parser.error("--horizon must be positive")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be nonnegative")

    seeds = (args.seed,) if args.seed is not None else DEFAULT_SEEDS
    records = [
        _run_seed(seed=seed, samples=args.samples, horizon=args.horizon)
        for seed in seeds
    ]
    result = _single_seed_result(records[0]) if args.seed is not None else _summary_result(records)
    result["dataset"] = args.dataset
    result["samples_per_seed"] = args.samples
    result["horizon"] = args.horizon

    text = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(args.output)


def _run_private_dataset(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Dispatch authorized clinical data to the local experiment implementation.

    The public entry point intentionally retains the same command shape for all
    five datasets.  Clinical presets and data loaders remain local because the
    underlying data are restricted, so a source-only checkout reports the
    missing private components explicitly instead of silently substituting
    synthetic data.
    """

    internal_runner = ROOT / "internal" / "run_paper_experiment.py"
    if not internal_runner.is_file():
        parser.error(
            f"{args.dataset} requires the authorized clinical-data extension; "
            "synthetic remains fully self-contained"
        )

    import subprocess

    command = [sys.executable, str(internal_runner), "--dataset", args.dataset]
    if args.seed is not None:
        command.extend(("--seeds", str(args.seed)))
    command.extend(("--devices", args.devices))
    command.extend(("--workers-per-device", str(args.workers_per_device)))
    if args.output_dir is not None:
        command.extend(("--output-dir", str(args.output_dir)))
    if args.resume:
        command.append("--resume")
    subprocess.run(command, check=True)


def _run_seed(*, seed: int, samples: int, horizon: int) -> dict[str, torch.Tensor | int | float]:
    inputs = _synthetic_inputs(seed=seed, samples=samples, horizon=horizon)
    selection = calibrate_per_step_marginal(
        inputs,
        target_policy=_SyntheticTargetPolicy(),
        logging_policy=_UniformLoggingPolicy(),
        outcome_model=_UnitScaleOutcomeModel(),
    )
    if not selection.selection_available:
        raise RuntimeError(f"no feasible radius at stage {selection.failure_stage}")
    return {
        "seed": seed,
        "target_coverage": inputs.target_coverage,
        "radii": selection.radii,
        "coverage": selection.estimated_coverage,
        "effective_sample_size": selection.effective_sample_size,
    }


def _single_seed_result(record: dict[str, torch.Tensor | int | float]) -> dict[str, object]:
    return {
        "seed": record["seed"],
        "target_coverage": record["target_coverage"],
        "radii": _rounded(record["radii"], digits=6),
        "estimated_coverage": _rounded(record["coverage"], digits=6),
        "effective_sample_size": _rounded(record["effective_sample_size"], digits=3),
    }


def _summary_result(records: list[dict[str, torch.Tensor | int | float]]) -> dict[str, object]:
    coverage = torch.stack([record["coverage"] for record in records])
    radii = torch.stack([record["radii"] for record in records])
    effective_size = torch.stack([record["effective_sample_size"] for record in records])
    mean_coverage = coverage.mean(dim=0)
    return {
        "seeds": [int(record["seed"]) for record in records],
        "n_seeds": len(records),
        "target_coverage": records[0]["target_coverage"],
        "worst_step_coverage": round(float(mean_coverage.min()), 6),
        "mean_stagewise_coverage": _rounded(mean_coverage, digits=6),
        "mean_stagewise_radii": _rounded(radii.mean(dim=0), digits=6),
        "mean_stagewise_effective_sample_size": _rounded(
            effective_size.mean(dim=0),
            digits=3,
        ),
    }


def _rounded(value: torch.Tensor | int | float, *, digits: int) -> list[float]:
    tensor = torch.as_tensor(value)
    return [round(float(item), digits) for item in tensor]


def _synthetic_inputs(*, seed: int, samples: int, horizon: int) -> PerStepCalibrationInputs:
    generator = torch.Generator().manual_seed(seed)
    states = torch.randn((samples, horizon + 1, 1), generator=generator)
    actions = torch.randint(0, 2, (samples, horizon), generator=generator)
    noise = 0.75 * torch.randn((samples, horizon, 1), generator=generator)
    outcomes = 0.35 * states[:, 1:] + 0.25 * actions[..., None] + noise
    scores = outcomes.abs().squeeze(-1)
    stage_grid = torch.linspace(0.05, 4.0, 80)
    return PerStepCalibrationInputs(
        trajectories=TrajectoryBatch(
            states=states,
            actions=actions,
            outcomes=outcomes,
            patient_ids=torch.arange(samples),
        ),
        scores=scores,
        stage_grids=stage_grid.repeat(horizon, 1),
        outcome_sd=torch.ones(1),
    )


if __name__ == "__main__":
    main()
