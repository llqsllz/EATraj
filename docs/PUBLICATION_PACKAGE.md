# Code-only publication package

The GitHub repository is intentionally a code-only release.

## Included

- `src/trajectory_prediction/`: model, data, geometry, metrics, and adapters;
- `scripts/`: training, evaluation, preprocessing, and audit entry points;
- `tests/`: unit and contract tests;
- public runtime and dataset-interface documentation.

## Excluded

- all private datasets and feature caches;
- raw replay files and source exports;
- checkpoints, optimizer state, and trained parameters;
- predictions, metrics, plots, tables, reports, and experiment logs;
- private manuscript, literature, and review material;
- credentials, tokens, local environment files, and machine-specific paths.

The exclusion is enforced by `.gitignore`. Before publishing, run `git status`
and inspect the staged file list. Never use `git add -f` for a file under an
excluded data, result, checkpoint, or credential path.
