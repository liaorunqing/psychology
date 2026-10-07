# Early versus recent personal baselines in repeated psychological self-reports

Reproducibility materials for **“Early versus recent personal baselines in repeated psychological self-reports: a cross-dataset prediction study.”**

This repository contains versioned analysis scripts, validation tests, dated protocols, and non-identifying aggregate outputs used to compare an early personal reference with recent and continuously updated histories across three development datasets and a Corona Health external extension. Protocol timing and post hoc status are stated explicitly; the complete study was not preregistered.

## Scope

The release covers:

- report-budget and baseline-age analyses;
- external confirmation in openESM 0012;
- equal-information and nonstationarity challenges;
- participant-level reliability and temporal-generalizability analyses;
- nested prediction of future updating benefit;
- the sustained-deterioration safety experiment;
- a participant-matched stationary AR(1) falsification analysis;
- expanded leakage-safe comparators, including median, mode, and online AR(1);
- EMA first-day and prompt-timing sensitivities;
- participant-SD-standardized temporal-generalizability analyses;
- reviewer-requested comparator and bounded-score analyses;
- window-length, CES recall-gap, Marian inverse-observation-weighting, and
  repeated pseudo-origin sensitivities; and
- a protocol-locked Corona Health PHQ-9 external extension with a GAD-7 sensitivity;
- empirical stationary-block and AR(1) marginal-model diagnostics; and
- generation of the five main manuscript figures.

The repository does **not** contain raw or participant-level derived data. Dataset access remains governed by the original repositories and licenses.

The `results/aggregate` and `figures/source_data` directories contain only
group-level, participant-balanced summaries used in the manuscript. They do
not contain participant identifiers or participant-level rows.

## Repository map

```text
analysis/revision_2026_09_29/  frozen analysis and figure scripts
direction_reset/               dated analysis protocols
tests/                         leakage and construction unit tests
results/aggregate/             non-identifying aggregate result tables
figures/source_data/           machine-readable source values for figures
figures/rendered/              rendered PDF figures
manuscript/                    LaTeX manuscript and supplement sources
```

## Data layout

Set `DPT_DATA_ROOT` to the directory containing the source datasets. If the variable is omitted, scripts look under `./data`.

```text
$DPT_DATA_ROOT/
├── dejonckheere_openesm/
│   └── 0012_dejonckheere_ts.tsv
├── marian_openesm/
│   └── 0052_marian_ts.tsv
└── Demographics/
    └── demographics.csv

repository sibling directory/
└── cgf63-kme28/
    ├── EMA.csv
    └── Codebook_Baseline_EMA.xlsx
```

CES-derived analysis tables are expected under `analysis/outputs/exploration/` because redistribution is not authorized by this repository. The Corona Health script resolves `cgf63-kme28` beside the repository by default. See `DATA_ACCESS.md` for source identifiers, hashes, and restrictions.

## Environment

The analyses used Python 3.12.3 with the versions pinned in `requirements.txt`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:DPT_DATA_ROOT = "D:\path\to\data"
python -m pytest tests -q
```

The full analysis is intentionally staged: later scripts consume frozen outputs from earlier stages. The protocol file paired with each script records the estimand, stopping rule, multiplicity treatment, and expected sample invariants.

## Reproducibility and privacy

- Random seed: `20260918`.
- Resampling is performed at the participant level where applicable.
- Temporal ordering and equal-information constraints are tested automatically.
- Raw data, participant identifiers, exact locations, caches, and local machine paths are excluded from the release.
- Results should not be interpreted as causal effects or clinical diagnostic performance.

## Citation

Please cite the accompanying article and this repository. Machine-readable citation metadata are provided in `CITATION.cff`.

## Archiving status

The live code is available at <https://github.com/liaorunqing/psychology>.
Version `v1.3.1` is the synchronized release for the revised manuscript.
The version-specific Zenodo DOI is <https://doi.org/10.5281/zenodo.23216022>.

## License

Code is distributed under the repository's Apache-2.0 license. Third-party datasets are not covered by that license.
