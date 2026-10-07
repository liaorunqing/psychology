# Version 1.3.1 — synchronized revised-manuscript release

This release candidate synchronizes the public code and aggregate evidence with the revised four-dataset manuscript.

## Added

- Corona Health parsing, cohort construction, PHQ-9/GAD-7 analysis, long-gap checks, stationary reference analysis, figure generation, dated protocol, aggregate outputs, and manifest.
- Empirical stationary-block sensitivity and aggregate outputs.
- Participant-balanced observed-versus-simulated AR(1) marginal diagnostics and dated protocol.
- Revised manuscript and supplementary sources, including corrected cross-references and readable split tables.

## Corrected

- Marian response-probability matching now uses the recorded `target_occasion` rather than reconstructing the opportunity from baseline age.
- A synthetic regression test verifies correct matching after an early missed opportunity.
- The CES AR(1) age-gradient eligibility and the Figure 5 common-eligibility sources are synchronized with the manuscript.
- IPW documentation now distinguishes clipped response probabilities from reciprocal, within-participant-normalised weights.

No raw or participant-level source data are included.
