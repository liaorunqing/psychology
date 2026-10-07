# Corona Health protocol-locked external-extension result

Date completed: 2026-10-05  
Protocol: `90_corona_health_external_extension_protocol_2026-10-05.md`  
Script: `analysis/revision_2026_09_29/corona_health_external_extension.py`  
Output directory: `analysis/outputs/revision_2026_09_29/corona_health_extension/`

## Decision

**Classification: direction and long-gap sensitivity supported.** The independently sourced Corona Health cohort strengthens the evidence that the manuscript's result is not peculiar to the original three datasets. It does not convert the full study into a preregistered confirmation: the research question and B8--L8 comparison were developed using the earlier datasets, and the extension protocol was frozen only before the new outcome comparison.

## Primary PHQ-9 result

After range validation and collapse of 47 identical duplicate participant--timestamp groups, 324 participants contributed 5,618 post-calibration targets. Participant-balanced MAE was 2.196 for B8 and 1.831 for L8. The paired difference was 0.365 PHQ-9 points (95% participant-bootstrap CI 0.276 to 0.462), equivalent to a 16.60% reduction (95% CI 13.18% to 20.11%). L8 had lower participant-specific MAE for 58.0% of participants.

The age-gradient subset comprised 232 participants and 5,426 targets. Mean within-person age gradient was 0.291 (95% CI 0.168 to 0.411) per one-SD increase in centred log baseline age; 57.8% of participant slopes were positive.

## Frozen challenges

- **Long-gap sensitivity:** after splitting sequences at gaps longer than 30 days and reconstructing both histories, 274 participants and 4,978 targets remained. B8-minus-L8 MAE was 0.404 (95% CI 0.303 to 0.518), a relative MAE reduction of 18.93%.
- **Ordinal sensitivity:** after rounding and clipping predictions, participant-balanced PHQ-9 MAE was 2.181 for B8, 1.807 for L8, and 1.797 for Median8. Exact accuracy was 21.0%, 25.4%, and 29.2%, respectively; within-two-point accuracy was 69.6%, 75.7%, and 76.2%.
- **Stationary AR(1) challenge:** the observed PHQ-9 MAE advantage (0.365) exceeded every one of 5,000 participant-matched Gaussian AR(1) draws after Monte Carlo correction (null mean 0.175; central 95% interval 0.119 to 0.236; one-sided `p_MC=0.0002`). The observed age gradient (0.291) likewise exceeded the null distribution (mean 0.073; central 95% interval 0.021 to 0.129; `p_MC=0.0002`). This rejects that fitted null, not all stationary psychological processes.

## Secondary GAD-7 result

GAD-7 was fixed as secondary before comparison. Among 327 participants and 5,663 targets, participant-balanced MAE fell from 1.946 to 1.619, a 16.82% reduction (95% CI 13.54% to 20.35%). The direction was also retained in the long-gap analysis. This result supports construct generalization but must not replace the PHQ-9 result or be presented as an additional independent confirmation.

## What this repairs

1. **Independent-data gap:** the manuscript now includes a protocol-locked analysis of a fourth, independently sourced cohort not used to develop the original result.
2. **Construct gap:** PHQ-9 adds a full depression symptom scale and GAD-7 adds an anxiety scale, while preserving the claim that replication concerns a temporal measurement pattern rather than one latent disorder.
3. **Ordinal/floor gap:** integer-rounded accuracy and a recent median reproduce the direction without treating the scale as fully continuous.
4. **Mechanism gap:** the effect exceeds a fitted stationary AR(1) expectation and survives removal of long gaps. This narrows, but does not identify, the mechanism.

## Remaining limits

- The extension is protocol-locked but not preregistered and was added after the manuscript's core hypothesis existed.
- Assessment was self-selected and irregular; sequence order does not imply a controlled schedule.
- The Gaussian AR(1) challenge ignores irregular continuous-time dynamics and is not a universal stationarity test.
- PHQ-9 and GAD-7 report intervals may overlap when assessments occur close together; the median gap was about seven days.
- A lower next-report error is a measurement result, not evidence of diagnostic validity, clinical benefit, or causal psychological change.

