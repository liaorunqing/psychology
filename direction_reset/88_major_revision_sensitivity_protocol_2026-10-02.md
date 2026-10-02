# Major-revision sensitivity protocol (frozen before execution)

Date frozen: 2026-10-02 (Asia/Shanghai)

Status: post-review sensitivity work. This document is not a prospective
preregistration and must not be described as one.

## Purpose

The analyses below address four reviewer concerns about the equal-information
comparison between an early eight-report mean (B8) and the immediately prior
eight-report mean (L8). No result from these analyses had been calculated when
this protocol was written.

## Frozen analyses

1. **CES recall separation.** Repeat the B8--L8 comparison after dividing
   targets by the elapsed time since the immediately preceding complete PHQ-2
   assessment: `<14 days` versus `>=14 days`. The latter guarantees that the
   target's 14-day recall period does not overlap the nearest history report.
2. **Marian observation-process sensitivity.** Merge the already generated,
   participant-disjoint response probabilities from the Marian observation
   model with the equal-weight B8/L8 target set. Re-estimate participant-level
   MAE and baseline-age slopes using inverse response-probability weights,
   clipped to 0.10--0.99 and normalized within participant. This analysis is
   conditional on observed-history MAR and cannot identify unobserved states.
3. **Window-length sensitivity.** On identical targets beginning after 12
   complete reports, compare early and recent equal-weight means for
   `K = 4, 6, 8, 12`. A direction is robust when recent history has lower
   participant-balanced MAE for each K; magnitude need not be monotonic.
4. **Repeated pseudo-origins.** For each participant with sufficient follow-up,
   create early, middle, and late pseudo-origins. Each origin uses eight reports
   for calibration, skips the next eight reports, and evaluates all later
   targets (minimum five). This prevents the mechanically identical first
   B8/L8 prediction from determining the slope. Report participant-balanced
   MAE change and participant-specific benefit slopes by origin.

All predictors are constructed from strictly prior complete reports. The
independent unit is the participant. Uncertainty uses 2,000 participant-level
bootstrap resamples with seed `20260918`. Results are descriptive sensitivity
evidence and do not redefine the original primary outcomes or multiplicity
family.

