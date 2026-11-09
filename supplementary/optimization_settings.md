# Optimization Settings

This document records the data, model, and optimization settings used to train the final EATraj model (the Stage-2 warm-start joint model, seed 17). The machine-readable counterpart is `optimization_settings.json`.

## Data and windowing

| Item | Value |
|---|---|
| Dataset interface | `data/trajectory_joint_v5/common50_hist30_20260923` |
| Sample rate | 10 Hz |
| Observation window | 30 s (301 points) |
| Prediction horizon | 5 s (50 points) |
| Training windows | 42,850 |
| Validation windows | 3,510 |
| Test windows | 3,556 eligible (3,558 total; 2 ineligible) |

## Model configuration

| Item | Value |
|---|---|
| Temporal backbone | Mamba-2 (selective state-space encoder) |
| Hidden width | 64 |
| Encoder layers | 2 |
| Attention heads | 4 |
| Scene hypotheses (modes) | 5 |
| Relational interaction | Enabled |
| Joint decoding | Enabled, shared scene modes |
| Actor-local modes | 1 |
| Edge anchors | 3 |
| Edge geometry | Enabled |
| Motion decoder | Acceleration-based kinematic integration |

## Optimization

| Item | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Learning-rate schedule | 5-epoch linear warmup, then cosine decay |
| Minimum learning rate | 1e-6 |
| Batch size | 32 |
| Epoch budget | 120 |
| Early-stopping patience | 20 |
| Samples per group | 16 |
| Candidate-score loss weight | 0.1 |
| Gradient clipping | Max norm 1.0 |
| Seed | 17 |
| Precision | float32 |

## Initialization and checkpoint selection

The Stage-2 joint model is initialized with a weights-only warm start from the Stage-1 relational checkpoint (Mamba-2 temporal encoder with relation modelling, `joint=False`). Compatible tensors are loaded, while the optimizer state, RNG state, and epoch count are not reset or restored. Joint response-dimension tensors that do not exist in the Stage-1 checkpoint use their Stage-2 initialization.

Checkpoints are selected by the minimum validation family-macro scene-minADE at the 5-second horizon.
