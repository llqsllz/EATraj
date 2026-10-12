# EATraj: Interaction-Aware Multi-Aircraft Trajectory Prediction

This repository contains the reproducible implementation of a multi-aircraft
trajectory-prediction framework for open-airspace scenes. The public package
contains source code, runnable entry points, tests, and a short dataset
interface description. **The dataset, trained weights, predictions, reports,
manuscript files, and private credentials are intentionally not included.**

The model is organized around three mechanisms rather than a simple network
concatenation:

1. a long-context temporal encoder for each aircraft;
2. a directed geometric interaction module for cross-aircraft history; and
3. a scene-conditioned joint decoder that produces scored future candidates.

The default motion head predicts future acceleration proposals and integrates
them from the last observed position and velocity. All relation and velocity
features must be derived causally from the history window.

## Repository layout

```text
src/trajectory_prediction/   data adapters, models, geometry and metrics
scripts/                     training, evaluation and audit entry points
tests/                       unit and contract tests
docs/                        public data and release documentation
```

Local-only directories such as `data/`, `.experiments/`, `output/`,
`reports/`, `models/`, `figures/`, `tables/`, `literature/`, and `manuscript/`
are ignored by Git. They may remain on a developer workstation but are not
part of the public code release.

## Environment

Python 3.10+ is recommended. Install the lightweight dependencies first:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-data.txt
python -m pip install -r requirements-annotation.txt
```

The Mamba-2 implementation is optional. If `mamba-ssm` is unavailable, use
the Transformer temporal encoder or install the CUDA-compatible package for
your platform. The repository does not vendor CUDA, model weights, or data.

## Data interface

The loader expects a local dataset directory with a manifest, a scene index,
and feature-cache files. The required fields and causality rules are described
in [the public dataset overview](docs/PUBLIC_DATASET_OVERVIEW.md). A dataset
is supplied by the user at runtime; it is never downloaded or committed by
this repository.

## Example training commands

The commands below illustrate the model interfaces. Replace the dataset path
with a local, private dataset prepared according to the public data contract.

```powershell
# Transformer temporal baseline
python scripts/train_trajectory.py `
  --data C:\data\trajectory_dataset `
  --name transformer-relation-joint `
  --temporal transformer `
  --interaction --joint --modes 5 `
  --motion-decoder acceleration `
  --epochs 120 --seed 17

# Mamba-2 temporal encoder with the same relation and joint heads
python scripts/train_trajectory.py `
  --data C:\data\trajectory_dataset `
  --name mamba2-relation-joint `
  --temporal mamba2 `
  --interaction --joint --modes 5 `
  --motion-decoder acceleration `
  --epochs 120 --seed 17
```

The training script writes run metadata and checkpoints to the configured
local experiment directory. Those generated artifacts are ignored by Git.
Use `--help` for the complete model and evaluation interface.

## Testing

Run the code-level tests that use synthetic fixtures:

```powershell
python -m pytest tests/test_models.py tests/test_metrics.py `
  tests/test_public_benchmark.py tests/test_cache_integrity.py `
  tests/test_trajectory_reference_cache.py -q
```

The code package does not include a dataset. Tests that require a prepared
local dataset are intentionally outside the default release command.

## Scope and data responsibility

This code is released for research use. The original project data are
simulation-derived multi-aircraft trajectories and are not redistributed here.
Users must ensure that any local data satisfy applicable access, privacy,
licensing, and export-control requirements. Renaming scene roles does not
change the provenance of the source data.

## Citation

The accompanying manuscript and experimental results are maintained outside
this code-only release. Cite the associated paper when it becomes available.
