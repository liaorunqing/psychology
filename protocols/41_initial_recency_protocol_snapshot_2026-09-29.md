# Two-report, training-only recency baseline

Status: **post-hoc development analysis** on four repeatedly inspected
cohorts. It is not covered by the earlier OSF registration and cannot serve
as independent confirmation. This protocol is recorded before running the
new recency calculation; the previous fair H/HS and personal-mean results
have already been seen.

## Question and common policy

Does a transparent recency rule improve on the running prior-report mean,
and does a complex history/sensing model still provide value after comparing
against this stronger low-burden rule? Every method starts after exactly
two observed target reports for each held-out participant. For each later
target, the simple methods may use all *strictly earlier* observed outcomes,
including those revealed after the first two; the current outcome becomes
available only after that prediction. This is a common **startup budget**,
not a fixed total assessment budget, and the cohorts have different temporal
cadences and outcome constructs.

## Split, scale, and tuning

Reuse the current five-fold participant-held-out fold map and future-target
keys saved by `fair_history_adapter.py`; do not resplit or retune those
models. For each outer fold, reconstruct the target using only its training
participants: the CES distress factor uses its training-only outcome scaler,
PCA direction, and factor scale; the other three use a training-only scaler
for their original primary outcome. Assert reconstructed held-out actuals
match the saved fair comparison row by row.

For each eligible outer-training participant, compute future-target rows
after their first two observed reports. Let `m_it` be the mean of all earlier
reports and `l_it` the immediately preceding report. Predict with
`m_it + w * (l_it - m_it)`, where a single cohort-and-fold-specific
`w in [0,1]` minimizes mean participant-level squared error across
outer-training eligible people. Use the analytic quadratic minimizer,
clipped to `[0,1]`; when the denominator is zero, use `w=0`. Never use
held-out labels to choose `w`, features, or the objective. On held-out
people, compare the tuned recency rule with the running mean, last report,
history-only online model, and history-plus-sensing online model from the
already saved fair analysis, always on identical future targets.

Primary descriptive estimand is participant-balanced RMSE. Report each
dataset's participant and record counts, fold-specific `w`, matched RMSE,
and participant-bootstrap (2,000 draws; seed 20260918) paired relative
RMSE intervals for recency versus running mean and H/HS online versus
recency. Negative relative change denotes lower RMSE of the challenger.
Secondary horizon descriptions separate the first eligible future target
from subsequent targets; do not treat those strata or four cohorts as
independent replications or select a winning stratum. All comparisons are
exploratory, with no confirmatory multiplicity-adjusted claim.

## Validity gates

- Each outer-test participant belongs to one fold and each predicted target
  key appears exactly once.
- For every target, simple forecasts use only earlier outcomes; perturbing
  the current outcome must leave its current prediction unchanged.
- Full held-out target keys and standardized actuals agree with the saved
  fair comparison (numeric tolerance `1e-10`).
- Every fold's tuning sample uses only that fold's training participants.
- If any target reconstruction or merge fails, stop and report the mismatch
  rather than filling missing rows or changing the endpoint.

If this simple rule outperforms the current complex models, narrow the
scientific interpretation to the predictive information in repeated
self-report. Do not describe a retrospective win on these inspected data as
new algorithmic validation.
