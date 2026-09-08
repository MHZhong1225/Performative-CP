# **Self-Consistent Performative Conformal Prediction**

This project implements **SC-PCP**.

## **Installation**

SC-PCP requires Python 3.11 or later.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

PyTorch is a required dependency. For GPU use, install the PyTorch build that matches the local CUDA environment before installing SC-PCP.

## **Quick Start**

Run any dataset from the same entry point:

```bash
python run_experiment.py --dataset synthetic
# with authorized local data:
python run_experiment.py --dataset mimic_iv
python run_experiment.py --dataset eicu
python run_experiment.py --dataset inspire
python run_experiment.py --dataset mimic_cxr
```

The default command runs 20 seeds. To run one seed instead, pass `--seed`:

```bash
python run_experiment.py --dataset synthetic --seed 7
python run_experiment.py --dataset synthetic --gamma -2
```

The four clinical options require their authorized local data extension. A source-only checkout remains fully runnable on `synthetic`; it will state the missing clinical requirement clearly rather than silently changing datasets.

Run all public tests with:

```bash
python -m pytest -q tests
```

The public calibration interface is:

```python
from data import TrajectoryBatch
from scpcp import PerStepCalibrationInputs, calibrate_per_step_marginal

inputs = PerStepCalibrationInputs(
    trajectories=trajectories,
    scores=scores,
    stage_grids=stage_grids,
    outcome_sd=outcome_sd,
    target_coverage=0.90,
)

result = calibrate_per_step_marginal(
    inputs,
    target_policy=target_policy,
    logging_policy=logging_policy,
    outcome_model=outcome_model,
)
```

## **Paper Experiments**



```bash
conda run -n ucp python internal/run_synthetic_design.py
conda run -n ucp python internal/run_synthetic_experiment.py --all-gammas
conda run -n ucp python internal/run_synthetic_mfcs.py --all-gammas
conda run -n ucp python internal/render_synthetic_results.py
conda run -n ucp python internal/render_current_paper_results.py
```


## **Datasets**

| Dataset | Sequential response | Horizon | Action space |
|---|---|---|---|
| Synthetic | Two correlated outcomes under signed action--difficulty feedback | 12 stages | Three categorical actions |
| MIMIC-IV | Hypotension and tachycardia burden | `12 x 4 h` | Fluid--vasopressor intensity grid |
| eICU | Hypotension and tachycardia burden | `12 x 4 h` | Fluid--vasopressor intensity grid |
| INSPIRE | Hypotension and hypertension burden | `12 x 10 min` | None, fluid only, or vasopressor-containing treatment |
| MIMIC-CXR | Hypoxemia and tachypnea burden | `6 x 6 h` | No support, non-invasive support, or invasive ventilation |

The clinical tasks are patient-informed controlled deployment benchmarks, not prospective clinical evaluations. The MIMIC-CXR state additionally includes a frozen DenseNet-121 embedding of the index radiograph.

## **Data Layout**

The method consumes a `TrajectoryBatch`:

- `states`: `[N,T+1,D]`, with one state before each action and one terminal state;
- `actions`: `[N,T]`;
- `outcomes`: `[N,T,Y]`, where entry `t` is the post-action response `Y_(t+1)`;
- `patient_ids`: `[N]`.

Calibration scores have shape `[N,T]`, stage grids have shape `[T,K]`, and `outcome_sd` has shape `[Y]`. The logging policy implements `probabilities(states) -> [N,A]`; the radius-responsive target policy implements `probabilities_for_grid(states, radii) -> [N,K,A]`; and the frozen outcome model returns coordinate-wise means and positive scales.

All clinical roles are split by patient identifier. A patient appearing in more than one role is not allowed.

## **Experimental Protocol**

The target per-stage coverage is `0.90`. The primary feedback setting is `gamma=-4`; `gamma` in `{-2, 0, 2, 4}` forms the prespecified sensitivity analysis. Here `gamma` belongs to the deployment environment, not to SC-PCP.

Clinical patients are split into:

```text
D_pred / D_fid / D_env = 40% / 20% / 40%
```

`D_pred` fits and freezes the outcome and logging-policy models. `D_fid` fixes the radius-response range and supplies the native SPCI training stream. `D_env` constructs the frozen controlled evaluator and is never used for calibration. The synthetic benchmark instead uses known logging and target propensities and a known transition kernel.



## **Methods and Metrics**

The canonical comparison contains exactly:

- Standard CP
- ACI
- MFCS
- SPCI
- PRC
- SC-PCP

Continuous Causal CP is evaluated separately as a diagnostic because its native output is an individualized interval, not a fixed pre-deployment stagewise schedule.

The reported metrics are:

- **Marginal worst-step coverage (WSC):** `min_t mean_seed(C_seed,t)`, the primary coverage metric. For a `mean ± SD (n=20)` table entry, let `t* = argmin_t mean_seed(C_seed,t)` and report `mean_seed(C_seed,t*) ± SD_seed(C_seed,t*)`. Do not replace WSC with `mean_seed(min_t C_seed,t)`.
- **MeanCov:** for each seed, average coverage across stages, then report the mean and standard deviation across seeds.
- **Normalized prediction-set size:** use normalized box area for box methods and native normalized ellipsoid area for SPCI. The current Synthetic records also retain normalized coordinate width as a box-only diagnostic.



## **Data Availability**

The source release does not include restricted clinical data, patient-derived caches, paper result bundles, or generated figures. MIMIC-IV, eICU, INSPIRE, and MIMIC-CXR must be obtained from their respective custodians under the applicable data-use terms. Synthetic trajectories are generated by the included public example.

## **Source Layout**

- `run_experiment.py`: concise public synthetic entry point;
- `src/marginal_prefix.py`: committed-prefix SC-PCP selector;
- `src/experiments.py`: public calibration wrapper;
- `src/synthetic.py`: default signed-feedback Synthetic benchmark;
- `src/data.py`: public trajectory container;
- `tests/`: public API and method tests;
- `internal/`: local paper runners, baseline adapters, clinical pipelines, and renderers;
- `results/`: generated artifacts; intentionally not source-controlled.
