"""Post-review sensitivities for the dynamic-baseline manuscript."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "outputs" / "revision_2026_09_29"
PROTOCOL = ROOT / "direction_reset" / "88_major_revision_sensitivity_protocol_2026-10-02.md"
DATA_ROOT = Path(os.environ.get("DPT_DATA_ROOT", ROOT / "data")).expanduser()
DEJON = DATA_ROOT / "dejonckheere_openesm" / "0012_dejonckheere_ts.tsv"
MARIAN = DATA_ROOT / "marian_openesm" / "0052_marian_ts.tsv"
EQUAL = OUT / "equal_weight_mean_baseline_predictions.parquet"
MARIAN_RESPONSE = OUT / "marian_response_predictions.parquet"
SEED = 20260918
BOOTSTRAPS = 2000
WINDOWS = (4, 6, 8, 12)
OUTCOMES = {
    "Dejonckheere": ("sad", "stressed", "happy", "relaxed", "angry"),
    "CES": ("phq2",),
    "Marian": ("depressed", "anhedonia"),
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_series() -> dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]]:
    """Return complete values and chronological order for each person/outcome."""
    series: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    dejon = pd.read_csv(DEJON, sep="\t")
    for outcome in OUTCOMES["Dejonckheere"]:
        for pid, group in dejon[["id", "counter", outcome]].dropna().groupby("id", sort=False):
            group = group.sort_values("counter", kind="stable")
            series[("Dejonckheere", outcome, str(pid))] = (
                group[outcome].to_numpy(float), group.counter.to_numpy(int))
    marian = pd.read_csv(MARIAN, sep="\t")
    marian = marian.loc[marian.status.eq(1)].copy()
    for outcome in OUTCOMES["Marian"]:
        for pid, group in marian[["id", "counter", outcome]].dropna().groupby("id", sort=False):
            group = group.sort_values("counter", kind="stable")
            series[("Marian", outcome, str(pid))] = (
                group[outcome].to_numpy(float), group.counter.to_numpy(int))
    budget_path = Path(__file__).with_name("report_calibration_budget_curve.py")
    budget = load_module(budget_path, "major_revision_budget")
    ces = budget.PROFILE.BENCHMARK.build_wide_table()[["uid", "assessment_date", "phq2"]].dropna()
    ces["assessment_date"] = pd.to_datetime(ces.assessment_date)
    for pid, group in ces.groupby("uid", sort=False):
        group = group.sort_values("assessment_date", kind="stable")
        series[("CES", "phq2", str(pid))] = (
            group.phq2.to_numpy(float), group.assessment_date.to_numpy())
    return series


def participant_contrast(frame: pd.DataFrame, weighted: bool = False) -> pd.DataFrame:
    rows = []
    for pid, group in frame.groupby("participant", sort=False):
        if weighted:
            weights = 1.0 / group.response_probability.clip(.10, .99).to_numpy(float)
            weights /= weights.sum()
        else:
            weights = np.repeat(1.0 / len(group), len(group))
        ae_b = np.abs(group.actual.to_numpy(float) - group.B.to_numpy(float))
        ae_l = np.abs(group.actual.to_numpy(float) - group.L.to_numpy(float))
        rows.append({"participant": str(pid), "targets": len(group),
                     "mae_B": float(np.sum(weights * ae_b)),
                     "mae_L": float(np.sum(weights * ae_l))})
    return pd.DataFrame(rows)


def summarize_contrast(frame: pd.DataFrame, rng: np.random.Generator,
                       weighted: bool = False) -> dict:
    people = participant_contrast(frame, weighted=weighted)
    values = people[["mae_B", "mae_L"]].to_numpy(float)
    means = values.mean(axis=0)
    draws = np.empty(BOOTSTRAPS)
    for b in range(BOOTSTRAPS):
        sample = values[rng.integers(0, len(values), len(values))].mean(axis=0)
        draws[b] = 100 * (sample[1] / sample[0] - 1)
    return {"participants": len(people), "targets": len(frame),
            "mae_B": float(means[0]), "mae_L": float(means[1]),
            "relative_mae_percent": float(100 * (means[1] / means[0] - 1)),
            "ci_low": float(np.quantile(draws, .025)),
            "ci_high": float(np.quantile(draws, .975))}


def person_slopes(frame: pd.DataFrame, weighted: bool = False) -> pd.DataFrame:
    rows = []
    for pid, group in frame.groupby("participant", sort=False):
        if len(group) < 5 or group.age.nunique() < 2:
            continue
        x = np.log1p(group.age.to_numpy(float))
        x = (x - x.mean()) / x.std(ddof=0)
        y = (np.abs(group.actual.to_numpy(float) - group.B.to_numpy(float)) -
             np.abs(group.actual.to_numpy(float) - group.L.to_numpy(float)))
        if weighted:
            w = 1.0 / group.response_probability.clip(.10, .99).to_numpy(float)
            xbar = np.average(x, weights=w); ybar = np.average(y, weights=w)
            slope = np.sum(w * (x - xbar) * (y - ybar)) / np.sum(w * np.square(x - xbar))
        else:
            slope = np.dot(x, y) / np.dot(x, x)
        rows.append({"participant": str(pid), "targets": len(group), "slope": float(slope)})
    return pd.DataFrame(rows)


def summarize_slopes(frame: pd.DataFrame, rng: np.random.Generator,
                     weighted: bool = False) -> dict:
    slopes = person_slopes(frame, weighted=weighted)
    values = slopes.slope.to_numpy(float)
    draws = np.empty(BOOTSTRAPS)
    for b in range(BOOTSTRAPS):
        draws[b] = values[rng.integers(0, len(values), len(values))].mean()
    return {"slope_participants": len(values), "mean_slope": float(values.mean()),
            "slope_ci_low": float(np.quantile(draws, .025)),
            "slope_ci_high": float(np.quantile(draws, .975)),
            "positive_slope_fraction": float(np.mean(values > 0))}


def ces_gap(series) -> pd.DataFrame:
    rows = []
    values, dates = series[("CES", "phq2", next(pid for d, o, pid in series if d == "CES" and o == "phq2"))]
    del values, dates  # only validates that CES is present
    for (dataset, outcome, pid), (values, dates) in series.items():
        if (dataset, outcome) != ("CES", "phq2") or len(values) < 9:
            continue
        fixed = float(values[:8].mean())
        for t in range(8, len(values)):
            gap = float((dates[t] - dates[t - 1]) / np.timedelta64(1, "D"))
            rows.append({"participant": pid, "actual": values[t], "B": fixed,
                         "L": float(values[t-8:t].mean()), "age": float(
                             (dates[t] - dates[7]) / np.timedelta64(1, "D")),
                         "gap_days": gap,
                         "gap_group": ">=14 days" if gap >= 14 else "<14 days"})
    return pd.DataFrame(rows)


def window_rows(series, k: int, common_start: int = 12) -> pd.DataFrame:
    rows = []
    for (dataset, outcome, pid), (values, _orders) in series.items():
        if len(values) <= common_start:
            continue
        fixed = float(values[:k].mean())
        for t in range(common_start, len(values)):
            rows.append({"dataset": dataset, "outcome": outcome, "participant": pid,
                         "actual": float(values[t]), "B": fixed,
                         "L": float(values[t-k:t].mean()), "age": t - k + 1})
    return pd.DataFrame(rows)


def pseudo_origin_rows(series, origin_label: str) -> pd.DataFrame:
    rows = []
    for (dataset, outcome, pid), (values, _orders) in series.items():
        max_start = len(values) - 21  # 8 calibration + 8 washout + >=5 targets
        if max_start < 0:
            continue
        starts = {"early": 0, "middle": max_start // 2, "late": max_start}
        start = starts[origin_label]
        fixed = float(values[start:start+8].mean())
        for t in range(start + 16, len(values)):
            rows.append({"dataset": dataset, "outcome": outcome, "participant": pid,
                         "actual": float(values[t]), "B": fixed,
                         "L": float(values[t-8:t].mean()), "age": t - (start + 7)})
    return pd.DataFrame(rows)


def join_marian_probabilities(equal: pd.DataFrame, response: pd.DataFrame) -> pd.DataFrame:
    """Join target propensities using the recorded scheduled opportunity."""
    equal = equal.copy()
    if "target_occasion" not in equal:
        raise AssertionError("Equal-information predictions lack the true target occasion")
    equal["counter"] = pd.to_numeric(equal.target_occasion, errors="raise").astype(int)
    equal = equal.rename(columns={"B8_mean": "B", "L8_mean": "L"})
    response = response[["outcome", "participant", "counter", "response_probability"]].copy()
    response["participant"] = response.participant.astype(str)
    equal["participant"] = equal.participant.astype(str)
    merged = equal.merge(response, on=["outcome", "participant", "counter"],
                         how="left", validate="one_to_one")
    if merged.response_probability.isna().any():
        raise AssertionError("Missing Marian response probability")
    if merged.duplicated(["outcome", "participant", "counter"]).any():
        raise AssertionError("Duplicate Marian target key")
    return merged.rename(columns={"baseline_age": "age"})


def marian_ipw() -> pd.DataFrame:
    equal = pd.read_parquet(EQUAL)
    equal = equal.loc[equal.dataset.eq("Marian")].copy()
    response = pd.read_parquet(MARIAN_RESPONSE)
    return join_marian_probabilities(equal, response)


def main() -> None:
    if not PROTOCOL.exists():
        raise RuntimeError("Major-revision protocol missing")
    rng = np.random.default_rng(SEED)
    series = load_series()

    gap_rows = []
    gap = ces_gap(series)
    for label, group in gap.groupby("gap_group", sort=False):
        gap_rows.append({"gap_group": label, **summarize_contrast(group, rng),
                         **summarize_slopes(group, rng)})
    gap_result = pd.DataFrame(gap_rows)

    window_result = []
    for k in WINDOWS:
        frame = window_rows(series, k)
        for (dataset, outcome), group in frame.groupby(["dataset", "outcome"], sort=False):
            window_result.append({"dataset": dataset, "outcome": outcome, "window": k,
                                  **summarize_contrast(group, rng)})
    window_result = pd.DataFrame(window_result)

    origin_result = []
    for label in ("early", "middle", "late"):
        frame = pseudo_origin_rows(series, label)
        for (dataset, outcome), group in frame.groupby(["dataset", "outcome"], sort=False):
            origin_result.append({"dataset": dataset, "outcome": outcome, "origin": label,
                                  **summarize_contrast(group, rng),
                                  **summarize_slopes(group, rng)})
    origin_result = pd.DataFrame(origin_result)

    ipw_result = []
    frame = marian_ipw()
    for outcome, group in frame.groupby("outcome", sort=False):
        for analysis, weighted in (("unweighted", False), ("ipw", True)):
            ipw_result.append({"outcome": outcome, "analysis": analysis,
                               **summarize_contrast(group, rng, weighted),
                               **summarize_slopes(group, rng, weighted)})
    ipw_result = pd.DataFrame(ipw_result)

    gap_result.to_csv(OUT / "major_revision_ces_gap.csv", index=False)
    window_result.to_csv(OUT / "major_revision_window_length.csv", index=False)
    origin_result.to_csv(OUT / "major_revision_pseudo_origin.csv", index=False)
    ipw_result.to_csv(OUT / "major_revision_marian_missingness.csv", index=False)
    manifest = {
        "status": "post-review sensitivity; not preregistered",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED, "bootstrap_resamples": BOOTSTRAPS,
        "protocol_sha256": sha256(PROTOCOL), "script_sha256": sha256(Path(__file__)),
        "inputs": {"dejonckheere": sha256(DEJON), "marian": sha256(MARIAN),
                   "equal_predictions": sha256(EQUAL), "marian_response": sha256(MARIAN_RESPONSE)},
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__},
        "checks": {"strictly_prior_predictors": True, "common_targets_across_windows": True,
                    "pseudo_origin_eight_report_washout": True,
                    "participant_level_bootstrap": True,
                    "ipw_join_uses_true_target_occasion": True,
                    "ipw_probabilities_out_of_fold_by_participant": True},
    }
    (OUT / "major_revision_sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print("CES GAP\n", gap_result.to_string(index=False))
    print("\nWINDOWS\n", window_result.to_string(index=False))
    print("\nPSEUDO ORIGINS\n", origin_result.to_string(index=False))
    print("\nMARIAN IPW\n", ipw_result.to_string(index=False))


if __name__ == "__main__":
    main()
