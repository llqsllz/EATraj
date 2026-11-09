# Publication package

The repository releases the EATraj source code together with curated,
non-confidential documentation. It does not include the project dataset,
generated samples, or trained checkpoints.

## Included

- `src/trajectory_prediction/`: model, data, geometry, metrics, and adapters;
- `scripts/`: training, evaluation, preprocessing, and audit entry points;
- `tests/`: unit and contract tests;
- public runtime and dataset-interface documentation;
- `supplementary/`: optimization settings and additional stratified results,
  with a dedicated index in `supplementary/README.md`.

## Excluded

- all private datasets and feature caches;
- raw replay files and source exports;
- checkpoints, optimizer state, and trained parameters;
- raw prediction arrays, generated plots, and experiment logs;
- private manuscript, literature, and review material;
- credentials, tokens, local environment files, and machine-specific paths.

The exclusion is enforced by `.gitignore`. Before publishing, run `git status`
and inspect the staged file list. Never use `git add -f` for a file under an
excluded data, result, checkpoint, or credential path.
