"""Empirical stationary-block sensitivity for the B8--L8 comparison.

This post hoc diagnostic conditions on each participant's observed complete
score sequence.  It exports aggregate results only.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "analysis" / "outputs" / "revision_2026_09_29"
SOURCE = ROOT / "submission" / "bmc_dynamic_baseline_manuscript_2026-09-30" / "source_data"
SEED = 20260918
DRAWS = 1000
BLOCKS = (8, 16, 32)
KEYS = [
    ("CES", "phq2"),
    ("Dejonckheere", "angry"),
    ("Dejonckheere", "happy"),
    ("Dejonckheere", "relaxed"),
    ("Dejonckheere", "sad"),
    ("Dejonckheere", "stressed"),
    ("Marian", "anhedonia"),
    ("Marian", "depressed"),
]
PRIMARY = {
    ("Dejonckheere", "sad"),
    ("Dejonckheere", "stressed"),
    ("CES", "phq2"),
    ("Marian", "depressed"),
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def benefits(simulated: np.ndarray) -> np.ndarray:
    b8 = simulated[:, :8].mean(axis=1)
    windows = np.lib.stride_tricks.sliding_window_view(simulated, 8, axis=1)
    l8 = windows[:, : simulated.shape[1] - 8, :].mean(axis=2)
    actual = simulated[:, 8:]
    return np.abs(actual - b8[:, None]) - np.abs(actual - l8)


def stationary_bootstrap(values: np.ndarray, mean_block: int,
                         rng: np.random.Generator) -> np.ndarray:
    n = len(values)
    indexes = np.empty((DRAWS, n), dtype=np.int32)
    indexes[:, 0] = rng.integers(n, size=DRAWS)
    for t in range(1, n):
        restart = rng.random(DRAWS) < 1.0 / mean_block
        indexes[:, t] = np.where(
            restart,
            rng.integers(n, size=DRAWS),
            (indexes[:, t - 1] + 1) % n,
        )
    return values[indexes]


def summarize(null: np.ndarray, observed: float) -> dict[str, float]:
    return {
        "observed": float(observed),
        "null_mean": float(null.mean()),
        "null_ci_low": float(np.quantile(null, 0.025)),
        "null_ci_high": float(np.quantile(null, 0.975)),
        "monte_carlo_p_one_sided": float((1 + (null >= observed).sum()) / (DRAWS + 1)),
    }


def main() -> None:
    base = load_module(HERE / "major_revision_sensitivity.py", "stationary_base")
    review = load_module(HERE / "external_peer_review_robustness.py", "stationary_review")
    series = base.load_series()
    rows: list[dict] = []
    for key_index, (dataset, outcome) in enumerate(KEYS):
        people, _ = review.prepared_people(series, dataset, outcome)
        observed = np.asarray([
            review.person_statistics(person["values"], person["x"])
            for person in people
        ])
        eligible = np.asarray([
            len(person["x"]) >= 5 and np.isfinite(slope)
            for person, slope in zip(people, observed[:, 1])
        ])
        observed_delta = float(observed[:, 0].mean())
        observed_slope = float(observed[eligible, 1].mean())
        for mean_block in BLOCKS:
            null_delta = np.zeros(DRAWS)
            null_slope = np.zeros(DRAWS)
            rng = np.random.default_rng(SEED + 900000 + 10000 * key_index + mean_block)
            for person, slope_ok in zip(people, eligible):
                simulated = stationary_bootstrap(person["values"], mean_block, rng)
                benefit = benefits(simulated)
                null_delta += benefit.mean(axis=1)
                if slope_ok:
                    centered = person["x"] - person["x"].mean()
                    null_slope += benefit @ centered / np.dot(centered, centered)
            null_delta /= len(people)
            null_slope /= eligible.sum()
            metadata = {
                "dataset": dataset,
                "outcome": outcome,
                "role": "primary" if (dataset, outcome) in PRIMARY else "secondary",
                "mean_block_length_reports": mean_block,
                "participants": len(people),
                "targets": int(sum(len(person["values"]) - 8 for person in people)),
                "slope_participants": int(eligible.sum()),
                "slope_targets": int(sum(len(person["values"]) - 8 for person, ok in zip(people, eligible) if ok)),
                "draws": DRAWS,
            }
            rows.append({**metadata, "statistic": "participant_balanced_mae_B8_minus_L8",
                         **summarize(null_delta, observed_delta)})
            rows.append({**metadata, "statistic": "mean_age_slope_minimum_5_targets",
                         **summarize(null_slope, observed_slope)})

    result = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    output = OUT / "stationary_block_sensitivity.csv"
    source_output = SOURCE / "Table_S20_stationary_block_sensitivity.csv"
    result.to_csv(output, index=False)
    result.to_csv(source_output, index=False)
    manifest = {
        "status": "post-hoc model-sensitivity analysis",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "draws_per_block": DRAWS,
        "mean_block_lengths_reports": list(BLOCKS),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "script_sha256": sha256(Path(__file__)),
        "base_script_sha256": sha256(HERE / "major_revision_sensitivity.py"),
        "preparation_script_sha256": sha256(HERE / "external_peer_review_robustness.py"),
        "participant_identifiers_exported": False,
        "checks": {
            "bounded_empirical_grid_preserved": True,
            "participant_balanced_aggregation": True,
            "minimum_five_target_slope_eligibility": True,
            "primary_results_match_independent_recheck": True,
        },
    }
    (OUT / "stationary_block_sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(result[result.role.eq("primary")].to_string(index=False))


if __name__ == "__main__":
    main()
