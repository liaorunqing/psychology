"""Baseline-age mechanism analysis for frozen B8 versus online updating."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "outputs" / "revision_2026_09_29"
INPUT = OUT / "report_calibration_budget_predictions.parquet"
PROTOCOL = ROOT / "direction_reset" / "63_baseline_age_mechanism_protocol_2026-09-30.md"
SEED = 20260918
BOOTSTRAPS = 2000
EXPECTED = {("CES", "phq2"): (105, 4541),
            ("Marian", "anhedonia"): (145, 7071),
            ("Marian", "depressed"): (145, 7071)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assign_strata(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    ces = result.dataset.eq("CES")
    result["age_stratum"] = ""
    result.loc[ces, "age_stratum"] = pd.cut(
        result.loc[ces, "distance_8"], bins=[0, 90, 365, 730, np.inf],
        labels=["(0,90]", "(90,365]", "(365,730]", ">730"],
        right=True, include_lowest=False,
    ).astype("string")
    result.loc[~ces, "age_stratum"] = pd.cut(
        result.loc[~ces, "distance_8"], bins=[0, 7, 21, 42, np.inf],
        labels=["1-7", "8-21", "22-42", ">=43"],
        right=True, include_lowest=False,
    ).astype("string")
    if result.age_stratum.isna().any() or result.age_stratum.eq("").any():
        raise AssertionError("Unassigned age stratum")
    return result


def validate(frame: pd.DataFrame) -> None:
    key = ["dataset", "outcome", "participant", "occasion"]
    if frame.duplicated(key).any():
        raise AssertionError("Duplicate input target")
    for group_key, group in frame.groupby(["dataset", "outcome"], sort=False):
        people, targets = EXPECTED[group_key]
        if group.participant.nunique() != people or len(group) != targets:
            raise AssertionError(f"Input sample changed: {group_key}")
    if frame.distance_8.le(0).any():
        raise AssertionError("Baseline age must be positive")
    ordered = frame.sort_values(["dataset", "outcome", "participant", "report_rank"])
    decreases = ordered.groupby(["dataset", "outcome", "participant"]).distance_8.diff().dropna().lt(0)
    if decreases.any():
        raise AssertionError("Baseline age decreases within participant")
    frozen = frame.groupby(["dataset", "outcome", "participant"]).B8.nunique()
    if not frozen.eq(1).all():
        raise AssertionError("B8 prediction is not frozen")
    expected_ae = np.abs(frame.actual - frame.B8)
    if not np.isfinite(expected_ae).all():
        raise AssertionError("Non-finite outcome or prediction")


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    result = assign_strata(frame)
    result["ae_B8"] = np.abs(result.actual - result.B8)
    result["ae_R"] = np.abs(result.actual - result.R_online)
    result["se_B8"] = np.square(result.actual - result.B8)
    result["se_R"] = np.square(result.actual - result.R_online)
    result["ae_benefit"] = result.ae_B8 - result.ae_R
    result["se_benefit"] = result.se_B8 - result.se_R
    result["log_age_scaled"] = np.nan
    for dataset, group in result.groupby("dataset", sort=False):
        raw = np.log1p(group.distance_8.to_numpy(dtype=float))
        scale = float(raw.std(ddof=0))
        if scale <= 0:
            raise AssertionError(f"Degenerate log-age: {dataset}")
        result.loc[group.index, "log_age_scaled"] = raw / scale
    return result


def person_slopes(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, outcome, participant), group in frame.groupby(
            ["dataset", "outcome", "participant"], sort=False):
        if len(group) < 5:
            continue
        x = group.log_age_scaled.to_numpy(dtype=float)
        centered = x - x.mean()
        denominator = float(np.square(centered).sum())
        if denominator <= 0:
            continue
        record = {"dataset": dataset, "outcome": outcome,
                  "participant": participant, "targets": len(group),
                  "age_min": float(group.distance_8.min()),
                  "age_max": float(group.distance_8.max())}
        for column in ("ae_B8", "ae_R", "ae_benefit"):
            y = group[column].to_numpy(dtype=float)
            record[f"slope_{column}"] = float(np.dot(centered, y) / denominator)
        rows.append(record)
    return pd.DataFrame(rows)


def validate_slope_counts(result: pd.DataFrame, frame: pd.DataFrame) -> None:
    eligibility = frame.groupby(["dataset", "outcome", "participant"]).agg(
        targets=("actual", "size"), age_values=("distance_8", "nunique"))
    expected_ids = set(eligibility[(eligibility.targets >= 5) &
                                   (eligibility.age_values >= 2)].index)
    observed_ids = set(result[["dataset", "outcome", "participant"]]
                       .itertuples(index=False, name=None))
    if observed_ids != expected_ids:
        raise AssertionError("Individual slope eligibility does not match the frozen rule")


def summarize_slopes(slopes: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    columns = ("slope_ae_B8", "slope_ae_R", "slope_ae_benefit")
    for (dataset, outcome), group in slopes.groupby(["dataset", "outcome"], sort=False):
        values = group[list(columns)].to_numpy()
        draws = np.empty((BOOTSTRAPS, len(columns)))
        for iteration in range(BOOTSTRAPS):
            draws[iteration] = values[rng.integers(0, len(values), len(values))].mean(axis=0)
        for index, column in enumerate(columns):
            low, high = np.quantile(draws[:, index], [.025, .975])
            rows.append({"dataset": dataset, "outcome": outcome,
                         "quantity": column.removeprefix("slope_"),
                         "participants": len(group),
                         "targets": int(group.targets.sum()),
                         "mean_slope": float(values[:, index].mean()),
                         "ci_low": float(low), "ci_high": float(high),
                         "positive_slope_fraction": float((values[:, index] > 0).mean())})
    return pd.DataFrame(rows)


def stratum_metrics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED + 1)
    counts = frame.groupby(["dataset", "outcome", "participant"]).age_stratum.nunique()
    common = set(counts[counts.eq(4)].index)
    metric_rows, contrast_rows = [], []
    populations = [("available", frame),
                   ("all_four_strata", frame.loc[
                       [tuple(row) in common for row in frame[["dataset", "outcome", "participant"]]
                        .itertuples(index=False, name=None)]])]
    for population, table in populations:
        for (dataset, outcome, stratum), group in table.groupby(
                ["dataset", "outcome", "age_stratum"], sort=False):
            if group.empty:
                continue
            by_person = group.groupby("participant")[["ae_B8", "ae_R", "se_B8", "se_R"]].mean()
            b8_mae, r_mae = by_person[["ae_B8", "ae_R"]].mean()
            b8_rmse, r_rmse = np.sqrt(by_person[["se_B8", "se_R"]].mean())
            metric_rows.extend([
                {"dataset": dataset, "outcome": outcome, "population": population,
                 "age_stratum": stratum, "method": "B8", "participants": len(by_person),
                 "targets": len(group), "mae_pb": float(b8_mae), "rmse_pb": float(b8_rmse)},
                {"dataset": dataset, "outcome": outcome, "population": population,
                 "age_stratum": stratum, "method": "R_online", "participants": len(by_person),
                 "targets": len(group), "mae_pb": float(r_mae), "rmse_pb": float(r_rmse)},
            ])
            values = by_person.to_numpy()
            mae_draws, rmse_draws = np.empty(BOOTSTRAPS), np.empty(BOOTSTRAPS)
            for iteration in range(BOOTSTRAPS):
                sample = values[rng.integers(0, len(values), len(values))]
                means = sample.mean(axis=0)
                mae_draws[iteration] = 100 * (means[1] / means[0] - 1)
                rmse_draws[iteration] = 100 * (np.sqrt(means[3]) / np.sqrt(means[2]) - 1)
            contrast_rows.append({
                "dataset": dataset, "outcome": outcome, "population": population,
                "age_stratum": stratum, "participants": len(by_person), "targets": len(group),
                "R_vs_B8_mae_percent": float(100 * (r_mae / b8_mae - 1)),
                "mae_ci_low": float(np.quantile(mae_draws, .025)),
                "mae_ci_high": float(np.quantile(mae_draws, .975)),
                "R_vs_B8_rmse_percent": float(100 * (r_rmse / b8_rmse - 1)),
                "rmse_ci_low": float(np.quantile(rmse_draws, .025)),
                "rmse_ci_high": float(np.quantile(rmse_draws, .975)),
            })
    return pd.DataFrame(metric_rows), pd.DataFrame(contrast_rows)


def main() -> None:
    if not PROTOCOL.exists():
        raise RuntimeError("Frozen baseline-age protocol missing")
    source = pd.read_parquet(INPUT)
    validate(source)
    frame = prepare(source)
    slopes = person_slopes(frame)
    validate_slope_counts(slopes, frame)
    slope_summary = summarize_slopes(slopes)
    metrics, contrasts = stratum_metrics(frame)
    frame.to_parquet(OUT / "baseline_age_predictions.parquet", index=False)
    slopes.to_parquet(OUT / "baseline_age_person_slopes.parquet", index=False)
    slope_summary.to_csv(OUT / "baseline_age_slope_summary.csv", index=False)
    metrics.to_csv(OUT / "baseline_age_stratum_metrics.csv", index=False)
    contrasts.to_csv(OUT / "baseline_age_stratum_contrasts.csv", index=False)
    manifest = {
        "status": "post-inspection exploratory baseline-age mechanism audit",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED, "bootstrap_resamples": BOOTSTRAPS,
        "input_sha256": sha256(INPUT), "protocol_sha256": sha256(PROTOCOL),
        "script_sha256": sha256(Path(__file__)),
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__},
        "expected": {"|".join(key): {"participants": value[0], "targets": value[1]}
                     for key, value in EXPECTED.items()},
        "checks": {"positive_age": True, "nondecreasing_age": True,
                   "B8_frozen": True, "same_target_for_paired_errors": True,
                   "within_person_centered_slopes": True,
                   "participant_bootstrap": True},
    }
    (OUT / "baseline_age_manifest.json").write_text(json.dumps(manifest, indent=2),
                                                     encoding="utf-8")
    print("\nSLOPE SUMMARY")
    print(slope_summary.to_string(index=False))
    print("\nSTRATUM CONTRASTS")
    print(contrasts.to_string(index=False))


if __name__ == "__main__":
    main()
