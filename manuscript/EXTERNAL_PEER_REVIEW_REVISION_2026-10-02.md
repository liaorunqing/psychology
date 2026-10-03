# External peer-review revision record

Date: 2 October 2026

This record maps the principal external-review concerns to the revised manuscript and reproducibility files. All new analyses are labelled post hoc unless they belonged to the outcome-association-blind Dejonckheere protocol.

1. **Stationary autocorrelation alternative.** Added a participant-matched stationary Gaussian AR(1) parametric bootstrap with 5,000 draws. The revised paper reports lag-1 autocorrelation, descriptive between-person variance shares, null means, 95% null intervals, standardized distances, and Monte Carlo values. The observed primary B8-minus-L8 advantages exceeded every fitted stationary null distribution.
2. **Short EMA calibration period.** Added calendar-span reporting, removal of all first-day observations followed by re-anchoring, and prompt-position adjustment. Primary directions were unchanged.
3. **Evidential labels.** Replaced confirmatory language with a staged secondary-analysis chronology. Only Dejonckheere is described as outcome-association-blind; the AR(1), comparator, timing, and re-anchoring analyses are explicitly post hoc.
4. **Replication counting.** Replaced language implying eight independent replications with three selected samples and eight correlated measured outcomes.
5. **Comparator strength.** Added recent median, mode where admissible, online AR(1), and a diagnostic oracle mean, alongside last observation, cumulative mean, and fixed EWM8. Median or mode performed best for several bounded or floor-concentrated outcomes.
6. **Person-specific claims.** Replaced “stable personal characteristic” language with temporal generalizability of noisy person-specific estimates. Added participant-SD-standardized mean-benefit correlations and discussed mathematical coupling and regression to the mean.
7. **Semi-synthetic exercise.** Removed it from the abstract and main figures, reduced its role in the main text, and retained full details only in the Supplement as a deterministic illustration.
8. **Dataset selection and OSF.** Added dataset-selection criteria and the embargoed OSF URL, while stating that the registration addressed a separate question.
9. **Absolute and standardized effects.** Added absolute MAE differences to the main table and participant-SD-standardized comparator results to the Supplement.
10. **Presentation and reproducibility.** Retitled the manuscript neutrally, revised the abstract and Discussion, reordered references by first citation, removed line numbering and forced double spacing, regenerated Figures 2 and 5, added Supplementary Tables S16--S19, and expanded automated leakage and simulation tests.

Verification:

- Focused automated tests: 4 passed.
- Local MiKTeX compilation: main manuscript and supplement compiled without fatal errors, undefined references/citations, or overfull boxes.
- Visual PDF review: all 22 main-manuscript pages and 11 supplement pages inspected; no clipping, overlap, or unreadable table was observed.
- The built-in editor compiler could not initialize its standard directories on this Windows host; successful local compilation provides the compilation check for this revision.

Remaining author action before submission:

- tag and archive the revised repository version, then insert the final commit identifier and DOI in the Data Availability statement.
