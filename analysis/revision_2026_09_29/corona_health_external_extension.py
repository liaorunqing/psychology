"""Protocol-locked Corona Health B8--L8 external extension."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parents[1] / "cgf63-kme28" / "EMA.csv"
CODEBOOK = ROOT.parents[1] / "cgf63-kme28" / "Codebook_Baseline_EMA.xlsx"
PROTOCOL = ROOT / "direction_reset" / "90_corona_health_external_extension_protocol_2026-10-05.md"
OUT = ROOT / "analysis" / "outputs" / "revision_2026_09_29" / "corona_health_extension"
SEED = 20260918
BOOTSTRAPS = 2000
NULL_DRAWS = 5000
EXPECTED_HASHES = {
    DATA: "FBFF34486AC2D4FF61D5E92B2CAA7276CEC51719973FE232DF3096D3357F8D87",
    CODEBOOK: "551B65FDB308EDBA9809B2B78FD30D99EED8AC9B6964BBC8594400E256CA96B1",
}
SCALES = {"PHQ-9": ([f"phq9_{x}" for x in "abcdefghi"], 27),
          "GAD-7": ([f"gad7_{x}" for x in "abcdefg"], 21)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_raw() -> pd.DataFrame:
    with DATA.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        index = {name: position for position, name in enumerate(header)}
        columns = ["user_id", "collected_at"] + SCALES["PHQ-9"][0] + SCALES["GAD-7"][0]
        last = max(index[column] for column in columns)
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if len(row) <= last:
                raise AssertionError(f"Truncated required fields at CSV line {line_number}")
            rows.append([row[index[column]] for column in columns])
    frame = pd.DataFrame(rows, columns=columns)
    frame["participant"] = frame.pop("user_id").astype(str)
    frame["date"] = pd.to_datetime(frame.pop("collected_at"), dayfirst=True, errors="coerce")
    if frame.date.isna().any():
        raise AssertionError("Unparseable assessment timestamp")
    for column in columns[2:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def scale_records(raw: pd.DataFrame, outcome: str, long_gap: bool = False) -> pd.DataFrame:
    items, maximum = SCALES[outcome]
    valid = raw[items].notna().all(axis=1) & raw[items].apply(lambda x: x.between(0, 3)).all(axis=1)
    work = raw.loc[valid, ["participant", "date"] + items].copy()
    conflicts = (work.groupby(["participant", "date"])[items]
                 .nunique(dropna=False).max(axis=1).gt(1))
    if conflicts.any():
        bad = set(conflicts[conflicts].index)
        work = work.loc[~pd.MultiIndex.from_frame(work[["participant", "date"]]).isin(bad)]
    work = work.drop_duplicates(["participant", "date"] + items, keep="first")
    work["score"] = work[items].sum(axis=1)
    if work.score.lt(0).any() or work.score.gt(maximum).any():
        raise AssertionError("Scale total outside admissible range")
    work = work.sort_values(["participant", "date"], kind="stable")
    if work.duplicated(["participant", "date"]).any():
        raise AssertionError("Duplicate participant timestamp remains")
    if long_gap:
        gap = work.groupby("participant").date.diff().dt.total_seconds().div(86400)
        work["segment"] = gap.gt(30).groupby(work.participant).cumsum().astype(int)
        work["series_id"] = work.participant + "::" + work.segment.astype(str)
    else:
        work["series_id"] = work.participant
    return work[["participant", "series_id", "date", "score"]]


def build_predictions(records: pd.DataFrame, outcome: str) -> pd.DataFrame:
    rows = []
    for series_id, group in records.groupby("series_id", sort=False):
        group = group.sort_values("date", kind="stable")
        if len(group) < 9:
            continue
        participant = str(group.participant.iloc[0])
        values = group.score.to_numpy(dtype=float)
        dates = group.date.to_numpy()
        b8 = float(values[:8].mean())
        for index in range(8, len(group)):
            age = float((dates[index] - dates[7]) / np.timedelta64(1, "D"))
            if age <= 0:
                raise AssertionError("Nonpositive baseline age")
            recent = values[index - 8:index]
            rows.append({"outcome": outcome, "participant": participant,
                         "series_id": series_id, "target_index": index + 1,
                         "date": dates[index], "actual": values[index],
                         "baseline_age_days": age, "B8": b8,
                         "L8": float(recent.mean()), "Median8": float(np.median(recent))})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    if not result.groupby("series_id").B8.nunique().eq(1).all():
        raise AssertionError("B8 changed within a series")
    result["ae_B8"] = (result.actual - result.B8).abs()
    result["ae_L8"] = (result.actual - result.L8).abs()
    result["se_B8"] = (result.actual - result.B8).pow(2)
    result["se_L8"] = (result.actual - result.L8).pow(2)
    result["benefit"] = result.ae_B8 - result.ae_L8
    return result


def bootstrap_summary(predictions: pd.DataFrame, rng: np.random.Generator) -> dict:
    per = predictions.groupby("participant").agg(
        ae_B8=("ae_B8", "mean"), ae_L8=("ae_L8", "mean"),
        se_B8=("se_B8", "mean"), se_L8=("se_L8", "mean"),
        targets=("actual", "size"))
    values = per[["ae_B8", "ae_L8", "se_B8", "se_L8"]].to_numpy()
    draws = np.empty((BOOTSTRAPS, 5))
    for iteration in range(BOOTSTRAPS):
        sample = values[rng.integers(0, len(values), len(values))]
        means = sample.mean(axis=0)
        draws[iteration] = [means[0], means[1], means[0] - means[1],
                            100 * (means[1] / means[0] - 1),
                            100 * (np.sqrt(means[3]) / np.sqrt(means[2]) - 1)]
    means = values.mean(axis=0)
    return {"participants": int(len(per)), "targets": int(len(predictions)),
            "B8_MAE": float(means[0]),
            "B8_MAE_ci_low": float(np.quantile(draws[:, 0], .025)),
            "B8_MAE_ci_high": float(np.quantile(draws[:, 0], .975)),
            "L8_MAE": float(means[1]),
            "L8_MAE_ci_low": float(np.quantile(draws[:, 1], .025)),
            "L8_MAE_ci_high": float(np.quantile(draws[:, 1], .975)),
            "mae_difference_B8_minus_L8": float(means[0] - means[1]),
            "mae_difference_ci_low": float(np.quantile(draws[:, 2], .025)),
            "mae_difference_ci_high": float(np.quantile(draws[:, 2], .975)),
            "relative_mae_percent": float(100 * (means[1] / means[0] - 1)),
            "relative_mae_ci_low": float(np.quantile(draws[:, 3], .025)),
            "relative_mae_ci_high": float(np.quantile(draws[:, 3], .975)),
            "relative_rmse_percent": float(100 * (np.sqrt(means[3]) / np.sqrt(means[2]) - 1)),
            "relative_rmse_ci_low": float(np.quantile(draws[:, 4], .025)),
            "relative_rmse_ci_high": float(np.quantile(draws[:, 4], .975)),
            "participant_fraction_L8_better": float((per.ae_L8 < per.ae_B8).mean())}


def slope_summary(predictions: pd.DataFrame, rng: np.random.Generator) -> tuple[dict, pd.DataFrame]:
    scaled = np.log1p(predictions.baseline_age_days.to_numpy(dtype=float))
    scaled = scaled / scaled.std(ddof=0)
    work = predictions.assign(age_scaled=scaled)
    rows = []
    for participant, group in work.groupby("participant", sort=False):
        if len(group) < 5 or group.age_scaled.nunique() < 2:
            continue
        x = group.age_scaled.to_numpy(); x -= x.mean()
        slope = float(np.dot(x, group.benefit.to_numpy()) / np.dot(x, x))
        rows.append({"participant": participant, "targets": len(group), "slope": slope})
    slopes = pd.DataFrame(rows)
    values = slopes.slope.to_numpy()
    draws = np.array([values[rng.integers(0, len(values), len(values))].mean()
                      for _ in range(BOOTSTRAPS)])
    return ({"slope_participants": int(len(slopes)), "slope_targets": int(slopes.targets.sum()),
             "mean_age_gradient": float(values.mean()),
             "age_gradient_ci_low": float(np.quantile(draws, .025)),
             "age_gradient_ci_high": float(np.quantile(draws, .975)),
             "positive_gradient_fraction": float((values > 0).mean())}, slopes)


def ordinal_summary(predictions: pd.DataFrame, maximum: int) -> dict:
    result = {}
    for method in ("B8", "L8", "Median8"):
        work = predictions[["participant", "actual", method]].copy()
        work["estimate"] = np.clip(np.rint(work[method].to_numpy()), 0, maximum)
        work["absolute_error"] = (work.actual - work.estimate).abs()
        work["exact"] = work.actual.eq(work.estimate).astype(float)
        work["within_two"] = work.absolute_error.le(2).astype(float)
        per = work.groupby("participant")[["absolute_error", "exact", "within_two"]].mean()
        result[f"{method}_rounded_MAE"] = float(per.absolute_error.mean())
        result[f"{method}_exact_accuracy"] = float(per.exact.mean())
        result[f"{method}_within_two_accuracy"] = float(per.within_two.mean())
    return result


def fit_ar1(records: pd.DataFrame) -> tuple[list[dict], float]:
    series = []
    pooled_x, pooled_y = [], []
    for participant, group in records.groupby("participant", sort=False):
        values = group.sort_values("date").score.to_numpy(dtype=float)
        if len(values) < 9:
            continue
        pooled_x.extend(values[:-1]); pooled_y.extend(values[1:])
        variance = float(values.var(ddof=1))
        phi = float(np.corrcoef(values[:-1], values[1:])[0, 1]) if variance > 0 else 0.0
        if not np.isfinite(phi): phi = 0.0
        series.append({"participant": participant, "n": len(values),
                       "mean": float(values.mean()), "sd": float(np.sqrt(max(variance, 1e-8))),
                       "phi_raw": phi})
    global_phi = float(np.corrcoef(pooled_x, pooled_y)[0, 1])
    for item in series:
        weight = max(item["n"] - 3, 0) / (max(item["n"] - 3, 0) + 10)
        item["phi"] = float(np.clip(weight * item["phi_raw"] + (1 - weight) * global_phi, -.95, .95))
    return series, global_phi


def stationary_null(records: pd.DataFrame, observed_difference: float,
                    observed_gradient: float, rng: np.random.Generator) -> tuple[dict, pd.DataFrame]:
    specifications, global_phi = fit_ar1(records)
    draws = np.empty((NULL_DRAWS, 2))
    for iteration in range(NULL_DRAWS):
        mae_person, slope_person = [], []
        for spec in specifications:
            n, phi, mean, sd = spec["n"], spec["phi"], spec["mean"], spec["sd"]
            innovation_sd = sd * np.sqrt(max(1 - phi * phi, 1e-8))
            values = np.empty(n); values[0] = rng.normal(mean, sd)
            for j in range(1, n):
                values[j] = mean + phi * (values[j - 1] - mean) + rng.normal(0, innovation_sd)
            b8 = values[:8].mean(); target = values[8:]
            l8 = np.array([values[j - 8:j].mean() for j in range(8, n)])
            benefit = np.abs(target - b8) - np.abs(target - l8)
            mae_person.append(float(benefit.mean()))
            if len(target) >= 5:
                age = np.log1p(np.arange(1, len(target) + 1, dtype=float))
                age /= age.std(ddof=0); age -= age.mean()
                slope_person.append(float(np.dot(age, benefit) / np.dot(age, age)))
        draws[iteration] = [np.mean(mae_person), np.mean(slope_person)]
    table = pd.DataFrame(draws, columns=["mae_difference_B8_minus_L8", "mean_age_gradient"])
    summary = {"global_lag1": global_phi, "draws": NULL_DRAWS,
               "mae_null_mean": float(table.iloc[:, 0].mean()),
               "mae_null_ci_low": float(table.iloc[:, 0].quantile(.025)),
               "mae_null_ci_high": float(table.iloc[:, 0].quantile(.975)),
               "mae_monte_carlo_p_one_sided": float((1 + (table.iloc[:, 0] >= observed_difference).sum()) / (NULL_DRAWS + 1)),
               "gradient_null_mean": float(table.iloc[:, 1].mean()),
               "gradient_null_ci_low": float(table.iloc[:, 1].quantile(.025)),
               "gradient_null_ci_high": float(table.iloc[:, 1].quantile(.975)),
               "gradient_monte_carlo_p_one_sided": float((1 + (table.iloc[:, 1] >= observed_gradient).sum()) / (NULL_DRAWS + 1))}
    return summary, table


def leakage_test() -> None:
    dates = pd.date_range("2020-01-01", periods=12, freq="7D")
    base = pd.DataFrame({"participant": "x", "series_id": "x", "date": dates,
                         "score": np.arange(12, dtype=float)})
    changed = base.copy(); changed.loc[changed.index >= 8, "score"] += 1000
    first = build_predictions(base, "toy").iloc[0]
    second = build_predictions(changed, "toy").iloc[0]
    assert first.B8 == second.B8 and first.L8 == second.L8


def main() -> None:
    if not PROTOCOL.exists(): raise RuntimeError("Frozen protocol missing")
    for path, expected in EXPECTED_HASHES.items():
        if sha256(path).upper() != expected: raise AssertionError(f"Hash mismatch: {path}")
    leakage_test()
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_raw(); rng = np.random.default_rng(SEED)
    summaries, slopes_all, ordinal_all, predictions_all = [], [], [], []
    primary_null = None
    for outcome, (_, maximum) in SCALES.items():
        records = scale_records(raw, outcome)
        predictions = build_predictions(records, outcome)
        overall = bootstrap_summary(predictions, rng)
        slope, slopes = slope_summary(predictions, rng)
        ordinal = ordinal_summary(predictions, maximum)
        sensitivity_predictions = build_predictions(scale_records(raw, outcome, long_gap=True), outcome)
        sensitivity = bootstrap_summary(sensitivity_predictions, rng)
        row = {"outcome": outcome, "role": "primary" if outcome == "PHQ-9" else "secondary",
               **overall, **slope,
               "long_gap_participants": sensitivity["participants"],
               "long_gap_targets": sensitivity["targets"],
               "long_gap_mae_difference": sensitivity["mae_difference_B8_minus_L8"],
               "long_gap_mae_ci_low": sensitivity["mae_difference_ci_low"],
               "long_gap_mae_ci_high": sensitivity["mae_difference_ci_high"],
               "long_gap_relative_mae_percent": sensitivity["relative_mae_percent"]}
        summaries.append(row); ordinal_all.append({"outcome": outcome, **ordinal})
        slopes.insert(0, "outcome", outcome); slopes_all.append(slopes)
        predictions_all.append(predictions)
        if outcome == "PHQ-9":
            primary_null, null_draws = stationary_null(
                records, overall["mae_difference_B8_minus_L8"],
                slope["mean_age_gradient"], rng)
            null_draws.to_parquet(OUT / "corona_health_phq9_stationary_null.parquet", index=False)
    summary = pd.DataFrame(summaries); ordinal = pd.DataFrame(ordinal_all)
    summary.to_csv(OUT / "corona_health_extension_summary.csv", index=False)
    ordinal.to_csv(OUT / "corona_health_ordinal_sensitivity.csv", index=False)
    pd.concat(slopes_all, ignore_index=True).to_csv(OUT / "corona_health_person_slopes.csv", index=False)
    pd.concat(predictions_all, ignore_index=True).to_parquet(OUT / "corona_health_predictions.parquet", index=False)
    (OUT / "corona_health_phq9_stationary_summary.json").write_text(
        json.dumps(primary_null, indent=2), encoding="utf-8")
    phq = summary.loc[summary.outcome.eq("PHQ-9")].iloc[0]
    classification = ("direction_and_long_gap_supported" if
                      phq.mae_difference_ci_low > 0 and phq.long_gap_mae_ci_low > 0
                      else "direction_only" if phq.mae_difference_ci_low > 0 else "not_supported")
    manifest = {"status": "completed protocol-locked post hoc external extension",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "classification": classification, "seed": SEED,
                "bootstrap_resamples": BOOTSTRAPS, "stationary_null_draws": NULL_DRAWS,
                "input_hashes": {k.name: sha256(k) for k in EXPECTED_HASHES},
                "protocol_sha256": sha256(PROTOCOL), "script_sha256": sha256(Path(__file__)),
                "software": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__},
                "checks": {"scale_ranges": True, "exact_duplicates_collapsed": True,
                           "current_and_future_excluded": True, "equal_eight_report_budget": True,
                           "participant_balanced": True, "participant_bootstrap": True}}
    (OUT / "corona_health_extension_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False)); print("\nORDINAL\n", ordinal.to_string(index=False))
    print("\nSTATIONARY\n", json.dumps(primary_null, indent=2)); print("\n", classification)


if __name__ == "__main__":
    main()
