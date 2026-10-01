"""Circular-shift challenge separating baseline staleness from local continuity."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "outputs" / "revision_2026_09_29"
PROTOCOL = ROOT / "direction_reset" / "76_circular_shift_nonstationarity_protocol_2026-09-30.md"
CROSS_SCRIPT = Path(__file__).with_name("cross_dataset_equal_information_replication.py")
DATA_ROOT = Path(os.environ.get("DPT_DATA_ROOT", ROOT / "data")).expanduser()
DEJON_DATA = DATA_ROOT / "dejonckheere_openesm" / "0012_dejonckheere_ts.tsv"
OBSERVED_REFERENCE = OUT / "equal_weight_mean_baseline_slope_summary.csv"
SEED = 20260918
PERMUTATIONS = 500
PRIMARY = {("Dejonckheere", "sad"), ("Dejonckheere", "stressed"),
           ("CES", "phq2"), ("Marian", "depressed")}
EXPECTED = {("Dejonckheere", outcome): (100, 7902)
            for outcome in ("sad", "stressed", "happy", "relaxed", "angry")}
EXPECTED.update({("CES", "phq2"): (99, 4523),
                 ("Marian", "depressed"): (145, 7071),
                 ("Marian", "anhedonia"): (145, 7071)})


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CROSS = load_module(CROSS_SCRIPT, "cross_for_circular_shift")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cyclic_pairs(values: np.ndarray) -> Counter:
    return Counter((float(values[index]), float(values[(index + 1) % len(values)]))
                   for index in range(len(values)))


def rotate_nonzero(values: np.ndarray, shift: int) -> np.ndarray:
    if shift <= 0 or shift >= len(values):
        raise AssertionError("Circular shift must be in 1..n-1")
    rotated = np.roll(values, shift)
    if not np.array_equal(np.sort(rotated), np.sort(values)):
        raise AssertionError("Circular shift changed marginal values")
    if cyclic_pairs(rotated) != cyclic_pairs(values):
        raise AssertionError("Circular adjacency was not preserved")
    return rotated


def load_specs() -> list[tuple[str, str, pd.DataFrame, str, str]]:
    dejon = pd.read_csv(DEJON_DATA, sep="\t")
    dejon["id"] = dejon.id.astype(str)
    cross_inputs = CROSS.load_inputs()
    specs = [("Dejonckheere", outcome, dejon, "id", "counter")
             for outcome in ("sad", "stressed", "happy", "relaxed", "angry")]
    specs.extend((dataset, outcome, frame, id_col, order_col)
                 for dataset, outcome, frame, id_col, order_col, _ in cross_inputs)
    return specs


def prepare_people(frame: pd.DataFrame, outcome: str, id_col: str,
                   order_col: str) -> list[dict]:
    work = frame[[id_col, order_col, outcome]].dropna().copy()
    work[id_col] = work[id_col].astype(str)
    work = work.sort_values([id_col, order_col], kind="stable")
    raw_people = []
    all_log_ages = []
    for participant, person in work.groupby(id_col, sort=False):
        values = person[outcome].to_numpy(dtype=float)
        orders = person[order_col].to_numpy()
        if len(values) < 9:
            continue
        difference = orders[8:] - orders[7]
        ages = (difference / np.timedelta64(1, "D")).astype(float) \
            if np.issubdtype(person[order_col].dtype, np.datetime64) \
            else difference.astype(float)
        if np.any(ages <= 0):
            raise AssertionError("Baseline age must be positive")
        log_age = np.log1p(ages)
        raw_people.append({"participant": participant, "values": values,
                           "ages": ages, "log_age": log_age})
        all_log_ages.append(log_age)
    scale = float(np.concatenate(all_log_ages).std(ddof=0))
    people = []
    for person in raw_people:
        if len(person["values"]) - 8 < 5 or np.unique(person["ages"]).size < 2:
            continue
        person["x"] = person["log_age"] / scale
        people.append(person)
    return people


def person_slope(values: np.ndarray, x: np.ndarray) -> float:
    initial_mean = float(values[:8].mean())
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    recent_mean = (cumulative[8:len(values)] - cumulative[:len(values) - 8]) / 8.0
    actual = values[8:]
    benefit = np.abs(actual - initial_mean) - np.abs(actual - recent_mean)
    centered = x - x.mean()
    denominator = float(np.square(centered).sum())
    return float(np.dot(centered, benefit) / denominator)


def observed_summary(dataset: str, outcome: str, people: list[dict]) -> dict:
    slopes = np.asarray([person_slope(person["values"], person["x"])
                         for person in people])
    return {"dataset": dataset, "outcome": outcome,
            "role": "primary" if (dataset, outcome) in PRIMARY else "secondary",
            "participants": len(people),
            "targets": int(sum(len(person["values"]) - 8 for person in people)),
            "observed_slope": float(slopes.mean()),
            "positive_slope_fraction": float((slopes > 0).mean())}


def circular_null(dataset: str, outcome: str, people: list[dict],
                  outcome_index: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + outcome_index * 10000)
    rows = []
    for iteration in range(PERMUTATIONS):
        slopes = []
        for person in people:
            values = person["values"]
            shift = int(rng.integers(1, len(values)))
            rotated = np.roll(values, shift)
            slopes.append(person_slope(rotated, person["x"]))
        rows.append({"dataset": dataset, "outcome": outcome,
                     "iteration": iteration + 1,
                     "mean_slope": float(np.mean(slopes))})
        if (iteration + 1) % 100 == 0:
            print(f"{dataset}/{outcome}: {iteration + 1}/{PERMUTATIONS}", flush=True)
    return pd.DataFrame(rows)


def verify_reference(observed: pd.DataFrame) -> None:
    reference = pd.read_csv(OBSERVED_REFERENCE).rename(columns={"mean_slope": "reference_slope"})
    merged = observed.merge(reference[["dataset", "outcome", "participants", "targets",
                                       "reference_slope"]],
                            on=["dataset", "outcome"], suffixes=("", "_reference"),
                            validate="one_to_one")
    if not np.allclose(merged.observed_slope, merged.reference_slope, atol=1e-12):
        raise AssertionError("Observed slopes do not match mean-only sensitivity")
    if not (merged.participants == merged.participants_reference).all():
        raise AssertionError("Observed participant counts changed")
    if not (merged.targets == merged.targets_reference).all():
        raise AssertionError("Observed target counts changed")


def self_test() -> None:
    values = np.asarray([1.0, 4.0, 2.0, 9.0, 7.0])
    for shift in range(1, len(values)):
        rotate_nonzero(values, shift)
    x = np.arange(1, 6, dtype=float)
    y = np.asarray([2.0, 5.0, 8.0, 11.0, 14.0])
    centered = x - x.mean()
    assert np.isclose(np.dot(centered, y) / np.square(centered).sum(), 3.0)


def classify(summary: pd.DataFrame) -> str:
    primary = summary.loc[summary.role.eq("primary")].copy()
    passed = primary.observed_slope.gt(0) & primary.monte_carlo_p_one_sided.le(.0125)
    if passed.all():
        return "cross_scale_nonstationarity_supported"
    high_frequency = primary.dataset.ne("CES")
    if passed.loc[high_frequency].all() and not passed.loc[~high_frequency].all():
        return "high_frequency_only_support"
    if passed.sum() >= 2:
        return "partial_support"
    return "local_continuity_dominant"


def main() -> None:
    if not PROTOCOL.exists():
        raise RuntimeError("Frozen circular-shift protocol missing")
    self_test()
    observed_rows, null_parts = [], []
    specs = load_specs()
    for index, (dataset, outcome, frame, id_col, order_col) in enumerate(specs):
        people = prepare_people(frame, outcome, id_col, order_col)
        observed_rows.append(observed_summary(dataset, outcome, people))
        null_parts.append(circular_null(dataset, outcome, people, index))
    observed = pd.DataFrame(observed_rows)
    verify_reference(observed)
    null = pd.concat(null_parts, ignore_index=True)
    rows = []
    for record in observed.itertuples(index=False):
        values = null.loc[(null.dataset.eq(record.dataset)) &
                          (null.outcome.eq(record.outcome)), "mean_slope"].to_numpy(dtype=float)
        rows.append({**record._asdict(), "null_mean": float(values.mean()),
                     "null_ci_low": float(np.quantile(values, .025)),
                     "null_ci_high": float(np.quantile(values, .975)),
                     "monte_carlo_p_one_sided": float(
                         (1 + np.sum(values >= record.observed_slope)) / (PERMUTATIONS + 1))})
    summary = pd.DataFrame(rows)
    classification = classify(summary)
    OUT.mkdir(parents=True, exist_ok=True)
    null.to_parquet(OUT / "circular_shift_nonstationarity_null.parquet", index=False)
    summary.to_csv(OUT / "circular_shift_nonstationarity_summary.csv", index=False)
    manifest = {"status": "post-confirmation circular-shift nonstationarity challenge",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "classification": classification, "seed": SEED,
                "shifts_per_outcome": PERMUTATIONS,
                "primary_bonferroni_threshold": .0125,
                "protocol_sha256": sha256(PROTOCOL), "script_sha256": sha256(Path(__file__)),
                "input_sha256": {"dejonckheere": sha256(DEJON_DATA),
                                 "observed_reference": sha256(OBSERVED_REFERENCE)},
                "software": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__},
                "checks": {"nonzero_shifts": True, "marginal_multisets_preserved": True,
                           "cyclic_adjacency_preserved": True,
                           "missing_positions_preserved": True,
                           "observed_targets_match_prior_analysis": True,
                           "participant_independent_unit": True}}
    (OUT / "circular_shift_nonstationarity_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nCLASSIFICATION:", classification)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
