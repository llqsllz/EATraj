# Public Dataset Interface Overview

This document describes the **interface** required by the code. It does not
contain the project dataset or any generated samples.

## Task

Given synchronized history from `N` aircraft, predict their future 3-D
positions. The history is the only model input. Future positions and future
validity masks are supervision/evaluation targets and must not be passed to a
normal forward call.

The recommended sample contract is:

```text
history_position    [N, T, 3]
history_time        [T] or [N, T]
history_mask        [N, T]
actor_mask          [N]
forecast_times      [H]
future_position     [N, H, 3]
future_valid_mask   [N, H]
metadata: sample_id, replay_id, actor_id_map, split, coordinate_reference
```

`N` may vary across scenes. Padding actors are identified by `actor_mask` and
must not contribute to relation aggregation or losses.

## Causal feature construction

The model may derive velocity, local coordinates, relative position, relative
velocity, distance, closure rate, and time-to-closest-approach proxies from
the history. Every derived value at time `t` must use observations no later
than `t`. Future positions must not be used to select actors, update a local
frame, estimate velocity, or construct relation edges.

The implementation supports actor-centred local coordinates. The translation
and rotation used for a sample must be fixed at the prediction endpoint and
stored in the local sample context so predicted positions can be restored to
the shared frame. Units, time origin, axis convention, and missing-value
semantics must be declared by the dataset adapter.

## Dataset directory

At runtime, the loader expects a dataset directory containing a dataset
configuration, a scene index, and the feature-cache split requested by the
training command. A typical private layout is:

```text
<dataset_root>/
  dataset_config.json
  scene_index.jsonl
  feature_cache/
    train/
    validation/
    test/
```

The exact cache manifest keys are validated by
`src/trajectory_prediction/packed_data.py`. Do not commit these files to the
repository; keep them in a private data location and pass that path through
`--data`.

## Recommended split policy

Split by replay or near-duplicate family before creating windows. Adjacent
windows from one replay must not be distributed independently across train,
validation, and test. The split manifest should record the split, source
family, coordinate convention, and preprocessing version.

## Scope

The code does not require intent, action, threat, platform, or cooperation
labels. A latent future mode is a trajectory hypothesis, not an automatically
interpretable intent class. Any semantic labels used in a downstream study
must be documented as an optional extension and must not leak future data into
the history-only task.

## Supplementary materials

The repository additionally provides the optimization settings and additional
stratified results under `supplementary/`. These are summary tables and
configuration documentation; they do not contain the dataset, generated
samples, or checkpoints.
