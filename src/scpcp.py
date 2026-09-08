"""Public entry point for Self-Consistent Performative Conformal Prediction."""

from experiments import PerStepCalibrationInputs, calibrate_per_step_marginal
from marginal_prefix import MarginalPrefixSelection
from synthetic import SyntheticConfig, SyntheticProblem, build_synthetic_problem

__all__ = [
    "MarginalPrefixSelection",
    "PerStepCalibrationInputs",
    "SyntheticConfig",
    "SyntheticProblem",
    "build_synthetic_problem",
    "calibrate_per_step_marginal",
]
__version__ = "0.1.0"
