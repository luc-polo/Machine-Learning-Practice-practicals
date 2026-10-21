# Rain forecast CRPS audit — 21 October 2026

The local CRPS calculation is correct. The exported and deployed model files match, and replaying the saved model independently reproduces the notebook's historical held-out score. The current leaderboard score is based on only **one** matured forecast, whereas most higher-ranked entries average around 100 forecasts. The first forecast encountered a 10.0 mm observation in Bucharest at +6 h. These findings explain why the historical average and the first live score can differ so sharply; they do not establish that future live performance will match the backtest.

## Formula and provenance

For equally weighted ensemble members, after clipping negative amounts to zero, the challenge defines

```
CRPS = mean_i |x_i - y| - (1 / (2 M^2)) sum_i sum_j |x_i - x_j|
```

The notebook sorts the members and calculates the second term as

```
sum_k x_(k) * (2*k - M - 1) / M^2       [k = 1,...,M]
```

These expressions are algebraically identical: the full pairwise sum is twice the sorted weighted sum. There is no missing factor of two. The ordinary empirical ensemble CRPS uses M^2 here. An alternative fair estimator uses a different correction for sampling a continuous distribution; it is not the score specified by this challenge.

The entire cells defining `crps`, `evaluate`, and `issue_times` are byte-for-byte identical to the original course notebook fetched during this audit. The CRPS function is in the original section 3, before the original baseline and random forest; it was not introduced by the added hurdle models in section 6.1 onward. The live challenge's overview independently specifies the same pairwise formula and averaging over 45 cities and the two horizons.

Independent numerical verification used the direct pairwise definition on arrays with 1, 2, 3, 30, 50, and 100 members, including negative members subject to clipping. Maximum absolute discrepancy: **8.88 × 10^-16**.

Sources: [original course notebook](https://github.com/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-s3-rain-model.ipynb), [ensemble CRPS documentation](https://scoringrules.readthedocs.io/en/latest/generated/scoringrules.crps_ensemble.html), and [challenge overview](https://ml-arena.com/challenges/177).

## Reproduced results

| Evaluation | Forecast issue times | +6 h CRPS | +48 h CRPS | Mean CRPS |
|---|---:|---:|---:|---:|
| Saved XGBoost, historical held-out replay | 1,700 | 0.067438 | 0.074747 | 0.071093 |
| Climatology, same historical replay | 1,700 | 0.076039 | 0.075750 | 0.075895 |
| Deployed XGBoost, first matured live forecast | 1 | 0.220323 | 0.177468 | 0.198896 |

All scores are in mm. The 0.071093 result is **not a training score**. Hyperparameters were selected on 2024, using earlier years for fitting/calibration. The final model was fitted on 2020–2023 and calibrated on 2024, then evaluated on forecasts issued from 1 January 2025 to 1 March 2026. The code excludes training/calibration targets that would cross the relevant evaluation boundaries. No temporal leakage was found in those boundaries.

The agent loaded directly from `rain_submission/` reproduced 0.07109284058351123. Direct request predictions agreed with batched replay predictions. Both local files match their saved submission-receipt SHA-256 checksums; the remotely retrieved agent source and model bytes also match the local files. Thus the discrepancy is not explained by accidentally uploading a different saved model.

The historical relative gain was 6.33% overall: 11.31% at +6 h and only 1.32% at +48 h. The notebook's weekly-block uncertainty describes a paired historical comparison, not the uncertainty of a future individual forecast or future leaderboard rank.

## What the live result actually measures

Submission 9439 (`rain-xgboost-best`) has **one scored run and 90 city/horizon outcomes** in the retrieved leaderboard snapshot. Its first forecast was issued for **19 October 2026 at 11:00 UTC** and scored on 21 October after its +48 h outcome became available. The remaining early runs report no matured forecasts, rather than zero scores. The live API confirms the advertised 48-hour scoring delay.

In the same snapshot, the leading starter has 114 scored runs and the tied city-month climatologies have 106. Their overall averages cover different weather dates. A same-period comparison cannot be inferred from this ranking. Those submissions are private, so this audit did not access their individual runs.

The first live replay image reports **Bucharest 10.0 mm** at +6 h. Even with every other city dry, an all-zero forecast would have +6 h CRPS of **10/45 = 0.222222 mm**, already close to the model's 0.220323. This is evidence that a localized heavy-rain event dominates that horizon's panel score; it is not evidence of a scaling error in CRPS. The +48 h image also shows rain including London 4.0 mm, Oslo 1.2 mm, and Frankfurt am Main 1.1 mm.

The historical replay itself had **37 of 1,700 individual runs** at or above 0.198896, and a maximum of 0.381331. This frequency describes the historical sample; it is not a probability estimate for the current forecast. Season also matters: the 124 October 2025 forecasts averaged 0.105227 for this model and 0.113264 for climatology, above their averages across the entire replay.

The leaderboard's displayed ±0.0000 does not establish certainty: the API reports a zero interval with only one scored run. The 90 outcomes share one weather situation and are not 90 independent issue dates.

## What remains unresolved

One live forecast is insufficient to establish live calibration or a stable ranking. The API exposed a rendered replay, but this audit did not recover the raw historical live request, all ensemble members, and all verifying city observations. Therefore the live score has not been independently recomputed city by city, and a live feed/feature mismatch or distribution shift has not been ruled out. Neither should be asserted as the cause without those data.

The appropriate next comparison is the model and baselines on **identical live issue times**, with per-city errors and occurrence/amount calibration checked when the raw inputs and outcomes are available. The original historical results remain reproducible, but they are not a guarantee of future superiority. Presenting the historical gain without that qualification would overstate the evidence.

## Reproduce and inspect

From the repository root:

```sh
.venv/bin/python "S3 rain-model/crps_audit.py"
```

This reads existing notebook definitions and saved model files, checks the CRPS identity, and writes the historical replay results. It does not retrain or submit anything.

Evidence is in `crps_audit_results/`: the two replay CSVs, numerical summary, a sanitized live score snapshot, and `first_scored_forecast.png`. No model, notebook, or live submission was modified by this audit.
