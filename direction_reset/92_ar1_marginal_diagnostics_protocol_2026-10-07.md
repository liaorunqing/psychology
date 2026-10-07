# Post hoc AR(1) marginal-diagnostics protocol

Date fixed: 2026-10-07, before running this diagnostic script.

## Purpose

Quantify how closely the fitted Gaussian AR(1) reference generators reproduce basic features of the observed complete-report sequences. This is a model check requested during pre-submission review; it is not a new confirmatory test and does not alter the previously reported B8--L8 contrasts.

## Data and units

Use the same eligible complete-report sequences and score bounds as the published AR(1) challenges. The independent summary unit is the participant (or anonymised Corona Health user identifier). Report participant-balanced averages.

## Fixed procedure

- Seed: 20260918.
- Simulated sequences per participant/outcome: 200.
- Development datasets use the generator in `external_peer_review_robustness.py`, including clipping to the admissible outcome bounds.
- Corona Health PHQ-9 uses the generator in `corona_health_external_extension.py`; the original null generator is Gaussian and is not clipped to the PHQ-9 bounds.
- For observed and simulated sequences report: sequence mean, within-sequence SD (population denominator), floor fraction, ceiling fraction, and lag-1 correlation. Also report the simulated fraction outside the admissible scale and the simulated fraction on the integer score grid.
- Aggregate first within sequence and then equally across participants. Simulated values are additionally averaged over draws.

## Interpretation

These quantities diagnose the fitted reference model. They are not independent goodness-of-fit tests, do not validate stationarity, and do not imply that discrepancies explain the observed prediction contrast. A poor marginal match restricts conclusions to the examined fitted reference process.

## Outputs

Only aggregate CSV and JSON manifest files may enter the public release. No participant identifiers or participant-level diagnostics may be exported.
