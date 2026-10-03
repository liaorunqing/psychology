# External peer-review robustness protocol (2026-10-02)

## Status and purpose

This protocol was written after receipt of external peer-review comments and before running the analyses below. Every analysis is post hoc and must be labelled as such. It does not convert the staged secondary study into a preregistered or confirmatory study.

## Questions

1. Does the observed advantage of an eight-report recent mean over an eight-report study-start mean exceed the advantage expected under a participant-matched stationary AR(1) process?
2. How much serial persistence, between-person heterogeneity, and linear trend is present in each outcome?
3. Do median-, mode-, cumulative-, exponentially weighted-, and online AR(1)-based predictors change the substantive comparator conclusion?
4. In the EMA datasets, do removal of day 1 and adjustment for prompt-of-day alter the result?
5. Does temporal generalizability of participant-level update benefit remain after division by each participant's observed outcome standard deviation?

## Frozen analysis decisions

- Random seed: `20260918`.
- The independent resampling unit is the participant.
- B8 is the mean of the first eight complete outcome reports. L8 is the mean of the eight complete reports strictly preceding a target.
- The observed error statistic is the participant-balanced mean of `MAE(B8) - MAE(L8)`; positive values favour L8.
- The age-gradient statistic is the participant-balanced mean within-person slope of `|y-B8|-|y-L8|` on log-transformed baseline age, using the original report-count or elapsed-day scale.
- Stationary null models use participant means and innovation variances plus outcome-level pooled lag-1 persistence. Participant lag-1 coefficients are shrunk toward the pooled coefficient with weight `(n-1)/(n-1+10)` and clipped to `[-0.95, 0.95]`. Simulated scores are clipped to the released scale bounds. We use 5,000 parametric-bootstrap draws and report the null mean, 95% interval, one-sided Monte Carlo p value, and standardized distance from the null mean.
- Comparator predictions are strictly prior except the full-series mean, which is explicitly labelled an oracle diagnostic and is never treated as deployable. The online AR(1) predictor estimates its mean and lag coefficient from prior observations only and shrinks the coefficient toward zero using the same 10-pair constant. EWM8 is treated as an exponentially weighted local-level heuristic, not as a novel model.
- Rolling medians are evaluated for all outcomes. Rolling modes are evaluated only for CES and Marian's bounded integer-valued outcomes, with ties resolved toward the most recent tied value.
- Participant-standardized errors divide each participant's MAE by the standard deviation of that participant's complete outcome series; zero-variance series are excluded only from standardized summaries.
- EMA day-1 exclusion re-anchors B8 after removal of all day-1 observations. Prompt adjustment subtracts the pooled prompt-number mean deviation from the outcome before constructing B8 and L8; this is a descriptive nuisance-adjusted sensitivity, not a prospective predictor.
- Confidence intervals use 2,000 participant bootstrap resamples. No multiplicity-controlled confirmatory family is declared.

## Interpretation rule

If the observed B8--L8 advantage or age gradient does not clearly exceed the stationary AR(1) null, the manuscript title, abstract, and conclusions must not claim that early personal baselines generally lose relevance. The result will instead be described as a comparison of early and recent histories that is compatible with short-range persistence. Individual temporal correlations near zero will be described as insufficient evidence for stable person-specific gradients, not evidence that such heterogeneity is absent.
