# Reviewer-revision robustness protocol

## Status and purpose

This is a **post-review robustness analysis** written after the primary B8-versus-L8 results were known. It cannot replace or be presented as prospectively preregistered confirmation. Its purpose is to answer four reporting questions raised during pre-submission review without changing the primary outcome family:

1. What are the absolute participant-balanced prediction errors of B8 and L8?
2. Does L8 remain informative relative to transparent leakage-safe alternatives?
3. For the bounded CES PHQ-2 score and four-category Marian items, does the conclusion survive category-level scoring?
4. What source-to-analysis attrition, score distribution, and available CES demographic information should be reported?

## Frozen inputs

- `dynamic_baseline_challenge_predictions.parquet`
- `cross_dataset_equal_information_predictions.parquet`
- CES constructed PHQ-2 panel produced by the frozen benchmark builder
- CES public demographics table
- Dejonckheere and Marian public longitudinal tables

The target records and participant identities must match the existing equal-weight sensitivity: 100 Dejonckheere participants and 7,902 targets per outcome; 105 CES participants and 4,541 targets; and 145 Marian participants and 7,071 targets per outcome.

## Comparator definitions

Every prediction is formed strictly before the target report.

- **B8:** arithmetic mean of the first eight complete reports.
- **L8:** arithmetic mean of the eight most recent complete reports.
- **Last observation:** most recent complete report.
- **Cumulative mean:** arithmetic mean of all complete reports available before the target, initialised by the first eight reports.
- **EWM8:** exponentially weighted mean initialised at B8 and updated only after each target becomes observed, with fixed smoothing parameter `alpha = 2/(8+1)`; no outcome-specific tuning is permitted.

The comparator analysis is descriptive and post hoc. It will not create a new confirmatory hypothesis family.

## Metrics and uncertainty

The independent resampling unit is the participant. For each dataset, outcome, and method, report:

- participant-balanced MAE and its 95% percentile-bootstrap interval;
- participant-balanced RMSE and its 95% percentile-bootstrap interval;
- MAE divided by the outcome range;
- relative MAE change of L8 against every alternative, with a participant bootstrap interval.

Use 2,000 participant bootstrap resamples and random seed `20260918`.

## Bounded and ordinal score sensitivity

For CES PHQ-2 and both Marian outcomes, clip predictions to the legal scale and round to the nearest category/score. Report participant-balanced:

- exact-score accuracy;
- accuracy within one score/category;
- rounded ordinal MAE.

Report L8-minus-B8 differences for accuracy and L8-minus-B8 change in rounded ordinal MAE with participant-bootstrap intervals. This analysis does not establish latent-scale measurement invariance; it only checks whether the B8-versus-L8 conclusion depends on evaluating continuous averages.

## Descriptive reporting audit

For every outcome, report target-score mean, standard deviation, minimum, maximum, and proportions at the scale floor and ceiling. Report source opportunities/observations, constructed panel size, eligibility after eight calibration reports, and evaluated targets. For CES, report self-described gender and race for the 105-person analytic sample using the public demographics file. No demographic field unavailable in the public release will be inferred.

## Interpretation rules

- The primary claim remains the B8-versus-L8 equal-information result.
- Comparator results will be described as post-review context, not confirmation.
- Category-level results will be described as bounded/ordinal scoring sensitivity, not an ordinal latent-variable model.
- If L8 does not outperform last observation, cumulative mean, or EWM8, the paper must not call L8 the optimal updating rule.
- The manuscript will use “start-anchored early baseline” rather than imply that the analysis identifies a unique causal ageing mechanism.

