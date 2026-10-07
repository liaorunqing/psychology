# Corona Health external-extension protocol (locked before B8--L8 comparison)

Date locked: 2026-10-05 (Asia/Shanghai)  
Status: prospective for the analyses specified below, but post hoc relative to the manuscript's original analyses and not preregistered  
Random seed: `20260918`

## Purpose

This extension asks whether the manuscript's central measurement result--a recent personal history predicts the next self-report more accurately than an equally sized history collected near entry--appears in an independent, international smartphone-app cohort. It is an external extension of a post hoc study, not a new confirmatory trial. It will not be described as preregistered or as validation of a clinical decision rule.

## Data frozen before outcome comparison

Dataset: *Longitudinal mental health data collected via the Corona Health smartphone app during COVID-19*, B2SHARE version 2, DOI `10.23728/b2share.cgf63-kme28`.

Files used:

- `EMA.csv`, SHA-256 `FBFF34486AC2D4FF61D5E92B2CAA7276CEC51719973FE232DF3096D3357F8D87`;
- `Codebook_Baseline_EMA.xlsx`, SHA-256 `551B65FDB308EDBA9809B2B78FD30D99EED8AC9B6964BBC8594400E256CA96B1`.

The structural audit found 11,479 complete PHQ-9 records from 1,487 participants. After exact duplicate participant--timestamp records are collapsed, 325 participants have at least nine complete reports and 246 have at least 12. Median spacing between consecutive complete PHQ-9 reports is approximately seven days. No B8--L8 errors, benefits, or age gradients were inspected before this protocol was written.

## Frozen outcome and eligibility

- Primary outcome: PHQ-9 total, the sum of `phq9_a` through `phq9_i`, with all nine items required and each item restricted to 0--3 (total 0--27).
- Secondary outcome: GAD-7 total, the sum of `gad7_a` through `gad7_g`, with all seven items required and each item restricted to 0--3 (total 0--21). It is a conceptual generalization check and will not replace PHQ-9 if its result is more favourable.
- The analysis unit for inference is the participant.
- Records are ordered by parsed `collected_at` within participant.
- Exact participant--timestamp duplicates with identical scale items are collapsed to one record. Conflicting duplicate timestamps, if present, are excluded. The structural audit found 47 exact duplicate timestamp groups and no conflicts for complete PHQ-9 records.
- The prediction sample requires at least nine complete reports: eight calibration reports plus at least one later target.
- The age-gradient sample requires at least 13 complete reports, at least five post-calibration targets, and nonzero variation in elapsed age.
- No outcome-value outlier exclusions will be made after scale-range validation.

## Frozen predictors and estimands

For participant (i), let (y_{ij}) be the ordered complete score.

- `B8`: arithmetic mean of reports 1--8, fixed for every later target.
- `L8`: arithmetic mean of the eight most recent strictly prior complete reports.
- Target-level update benefit: `abs(y - B8) - abs(y - L8)`; positive values favour L8.
- Primary estimand: participant-balanced MAE difference, B8 minus L8, with participant-level percentile bootstrap 95% CI (2,000 resamples).
- Secondary estimands: relative participant-balanced MAE change, participant-balanced RMSE change, proportion of participants with lower L8 MAE, and the mean participant-specific gradient of update benefit on within-person-centred `log1p(elapsed days since report 8)`, scaled by the target-level SD of that age variable.
- The same target rows are used for paired B8 and L8 errors. No current or future report enters either predictor.

## Frozen challenges

1. **Stationary AR(1) challenge.** A participant-matched Gaussian AR(1) parametric bootstrap will use the same construction as the manuscript: participant means, variances, and shrunk lag-1 coefficients; observed complete-report counts and elapsed-age covariates retained; 5,000 draws. This tests the fitted Gaussian first-order stationary process, not all stationary processes.
2. **Duplicate sensitivity.** Because audited duplicates are identical, retaining the first versus collapsing the group must yield identical scores and ordering after zero-gap duplicates are removed; the pipeline will assert this.
3. **Long-gap sensitivity.** The primary analysis uses complete-report order, matching the other datasets. A sensitivity analysis excludes targets whose immediately preceding complete report is more than 30 days earlier. B8 and L8 are then reconstructed within the retained sequence rather than deleting target rows after feature construction.
4. **Ordinal sensitivity.** B8 and L8 predictions are rounded to the nearest admissible integer and clipped to the scale range. Exact accuracy, within-two-point accuracy, and absolute ordinal error will be reported descriptively. Median8 is an additional descriptive comparator.

## Decision and reporting rules

- The extension supports generalization if the PHQ-9 participant-bootstrap interval for B8-minus-L8 MAE lies above zero and the direction is retained in the long-gap sensitivity. The magnitude, not only the sign, will be reported.
- A null or adverse PHQ-9 result remains in the manuscript or supplement and will not trigger switching the primary outcome to GAD-7.
- The AR(1) result will be used only to judge whether stationary persistence can account for the observed magnitude.
- This dataset will be labelled an independently sourced, protocol-locked external extension. Because the research question and analysis family were developed using the first three datasets, it will not be called a fully independent confirmation.
- All exclusions, participant counts, record counts, hashes, software versions, and deviations from this file will be exported.

