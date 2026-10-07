# Data access and provenance

No raw or participant-level derived data are redistributed in this repository.

| Dataset | Role | Access |
|---|---|---|
| College Experience Study (CES) | Cross-timescale PHQ-2 analysis | [Kaggle dataset page](https://www.kaggle.com/datasets/subigyanepal/college-experience-dataset), CC BY-NC-SA 4.0; access and reuse follow the repository terms. Required participant-level derived tables are not redistributed here. |
| Dejonckheere / openESM 0012 | External intensive-EMA confirmation | Zenodo DOI: [10.5281/zenodo.17347569](https://doi.org/10.5281/zenodo.17347569), CC BY 4.0. |
| Marian / openESM 0052 | Cross-dataset intensive-EMA replication | Zenodo DOI: [10.5281/zenodo.17348267](https://doi.org/10.5281/zenodo.17348267), CC BY 4.0. |
| Corona Health version 2 | Protocol-locked PHQ-9 external extension and GAD-7 sensitivity | B2SHARE DOI: [10.23728/b2share.cgf63-kme28](https://doi.org/10.23728/b2share.cgf63-kme28). Formal descriptor: Winter et al., *Scientific Data* (2026), DOI [10.1038/s41597-026-07015-7](https://doi.org/10.1038/s41597-026-07015-7). |

Users are responsible for downloading each dataset from its authoritative source, accepting its terms, and placing the files in the layout documented in `README.md`. File hashes and cohort-size invariants are checked by the analysis scripts and frozen protocols.

The Corona Health descriptor notes that reinstalling the app or changing phones could generate a new anonymised user identifier. Analyses therefore cluster and resample by identifier but do not assume that every identifier necessarily represents a distinct natural person.

The GitHub repository is a code and protocol release, not a redistribution endpoint for the source datasets.
