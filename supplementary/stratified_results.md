# Additional Stratified Results

This document reports scene-level errors across pre-defined strata for the three model stages used in the manuscript: the Mamba-2 temporal encoder, the relation-augmented model, and the warm-start joint model. The full numeric table (including scene-minFDE, top-score errors, and threshold coverage) is provided in `stratified_results.csv`.

## Window eligibility

The test partition contains 3,558 windows in total, of which 3,556 satisfy the scoring eligibility rules; two windows are excluded because they do not meet the minimum scored-actor requirement. All reported errors use the eligible-window set. The CSV retains both `n_windows` and `n_eligible_windows` so the two counts remain distinguishable.

## Stratum definitions

- **Four- / ten-aircraft scenes:** the two aircraft-count configurations, with 2,260 and 1,296 eligible windows.
- **Close / middle / far distance:** tertiles of the frozen historical edge range, with boundaries at 12,666.339 and 18,412.511 m. These are proximity strata.
- **High / low manoeuvre:** subsets separated by the frozen historical motion-score median (2.504753646). The motion score is the maximum over aircraft of three normalized median rates—absolute heading rate divided by 6 degrees/s, relative speed-change rate divided by 0.06/s, and absolute vertical acceleration divided by 2.5 m/s², using 2-s least-squares velocity estimates over the 50-s source history. Only the last 30 s enter the predictor, and no future target or model residual contributes to the grouping.

## Scene-minADE by stratum (metres)

| Stratum | Eligible / replays / groups | Mamba-2 | + Relation | Warm-start joint |
|---|---:|---:|---:|---:|
| All scenes | 3556 / 38 / 37 | 19.175 | 16.813 | 16.492 |
| Four-aircraft | 2260 / 26 / 25 | 17.068 | 14.622 | 14.700 |
| Ten-aircraft | 1296 / 12 / 12 | 23.565 | 21.379 | 20.225 |
| Close distance | 1184 / 34 / 34 | 18.225 | 16.115 | 16.363 |
| Middle distance | 1186 / 38 / 37 | 20.727 | 17.959 | 17.400 |
| Far distance | 1186 / 38 / 37 | 18.389 | 15.928 | 15.485 |
| High manoeuvre | 1779 / 38 / 37 | 21.119 | 18.445 | 18.158 |
| Low manoeuvre | 1777 / 38 / 37 | 17.601 | 15.398 | 15.073 |

## Reading the results

Relation modelling lowers scene-minADE in every reported stratum; in the high-manoeuvre subset it reduces scene-minADE from 21.119 to 18.445 m. The warm-start joint decoder adds a smaller, configuration-dependent effect: it reduces scene-minADE from 21.379 to 20.225 m in ten-aircraft scenes, while in four-aircraft scenes and the closest-distance tertile the value is essentially unchanged. The CSV additionally reports threshold coverage, where the joint model trades a small reduction at the 10-m threshold for a gain at the 20-m threshold.

## CSV column guide

- `n_windows` / `n_eligible_windows`: total and scoring-eligible window counts.
- `n_replays` / `n_families`: independence units within each stratum.
- `scene_minADE_m` / `scene_minFDE_m`: oracle scene-level minimum errors.
- `top_score_ADE_m` / `top_score_FDE_m`: errors of the highest-scoring candidate.
- `coverage_10m_*` / `coverage_20m_*`: candidate coverage at fixed scene-ADE thresholds, with window-weighted and group-macro weighting reported separately.
