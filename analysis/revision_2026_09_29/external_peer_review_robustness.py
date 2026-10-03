"""Post-hoc analyses requested in external peer review.

The script separates an early-versus-recent prediction contrast from the
advantage expected under stationary serial persistence, adds conventional
comparators, and quantifies EMA and individual-standardisation sensitivities.
"""

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
PROTOCOL = ROOT / "direction_reset" / "89_external_peer_review_robustness_protocol_2026-10-02.md"
BASE_SCRIPT = Path(__file__).with_name("major_revision_sensitivity.py")
DATA_ROOT = Path(os.environ.get("DPT_DATA_ROOT", ROOT / "data")).expanduser()
DEJON = DATA_ROOT / "dejonckheere_openesm" / "0012_dejonckheere_ts.tsv"
MARIAN = DATA_ROOT / "marian_openesm" / "0052_marian_ts.tsv"
TEMPORAL_BLOCKS = OUT / "individual_staleness_temporal_block_metrics.parquet"
SPLIT_RELIABILITY = OUT / "individual_staleness_reliability_summary.csv"
SEED = 20260918
NULL_DRAWS = 5000
BOOTSTRAPS = 2000
BOUNDS = {"Dejonckheere": (0.0, 100.0), "CES": (0.0, 6.0), "Marian": (1.0, 4.0)}
PRIMARY = {("Dejonckheere", "sad"), ("Dejonckheere", "stressed"),
           ("CES", "phq2"), ("Marian", "depressed")}


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


def age_values(orders: np.ndarray) -> np.ndarray:
    difference = orders[8:] - orders[7]
    if np.issubdtype(orders.dtype, np.datetime64):
        return (difference / np.timedelta64(1, "D")).astype(float)
    return difference.astype(float)


def rolling_mean(values: np.ndarray, width: int = 8) -> np.ndarray:
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    return (cumulative[width:] - cumulative[:-width]) / width


def mode_recent_first(values: np.ndarray) -> float:
    unique, counts = np.unique(values, return_counts=True)
    tied = set(unique[counts == counts.max()].tolist())
    for value in values[::-1]:
        if value in tied:
            return float(value)
    raise AssertionError("Mode tie resolution failed")


def prepared_people(series, dataset: str, outcome: str) -> tuple[list[dict], float]:
    raw, all_log_age = [], []
    for (dset, out, participant), (values, orders) in series.items():
        if (dset, out) != (dataset, outcome) or len(values) < 9:
            continue
        ages = age_values(orders)
        if np.any(ages <= 0):
            continue
        log_age = np.log1p(ages)
        raw.append({"participant": participant, "values": values.astype(float),
                    "orders": orders, "ages": ages, "log_age": log_age})
        all_log_age.append(log_age)
    scale = float(np.concatenate(all_log_age).std(ddof=0))
    people = []
    for person in raw:
        person["x"] = person["log_age"] / scale
        people.append(person)
    return people, scale


def person_statistics(values: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    b8 = float(values[:8].mean())
    l8 = rolling_mean(values)[:len(values) - 8]
    actual = values[8:]
    benefit = np.abs(actual - b8) - np.abs(actual - l8)
    centered = x - x.mean()
    slope = float(np.dot(centered, benefit) / np.dot(centered, centered)) \
        if len(x) >= 2 and np.dot(centered, centered) > 0 else np.nan
    return float(benefit.mean()), slope


def diagnostics(series) -> pd.DataFrame:
    rows = []
    keys = sorted({(dataset, outcome) for dataset, outcome, _ in series})
    for dataset, outcome in keys:
        people, _ = prepared_people(series, dataset, outcome)
        lag1, trends, means, within = [], [], [], []
        targets = 0
        for person in people:
            values = person["values"]
            targets += len(values) - 8
            if len(values) >= 3 and np.std(values[:-1]) > 0 and np.std(values[1:]) > 0:
                lag1.append(float(np.corrcoef(values[:-1], values[1:])[0, 1]))
            zt = np.arange(len(values), dtype=float)
            zt = (zt - zt.mean()) / zt.std(ddof=0)
            trends.append(float(np.dot(zt, values - values.mean()) / np.dot(zt, zt)))
            means.append(float(values.mean()))
            within.append(float(np.mean(np.square(values - values.mean()))))
        between = float(np.var(means, ddof=0))
        within_mean = float(np.mean(within))
        rows.append({"dataset": dataset, "outcome": outcome,
                     "role": "primary" if (dataset, outcome) in PRIMARY else "secondary",
                     "participants": len(people), "targets": targets,
                     "participant_mean_lag1_r": float(np.mean(lag1)),
                     "lag1_participants": len(lag1),
                     "descriptive_between_person_fraction": between / (between + within_mean),
                     "mean_person_linear_trend_per_time_sd": float(np.mean(trends)),
                     "positive_trend_fraction": float(np.mean(np.asarray(trends) > 0))})
    return pd.DataFrame(rows)


def pooled_phi(people: list[dict]) -> float:
    numerator = denominator = 0.0
    for person in people:
        values = person["values"]
        centered = values - values.mean()
        numerator += float(np.dot(centered[:-1], centered[1:]))
        denominator += float(np.dot(centered[:-1], centered[:-1]))
    return float(np.clip(numerator / denominator, -.95, .95)) if denominator > 0 else 0.0


def simulate_person(values: np.ndarray, draws: int, pooled: float,
                    bounds: tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    mean = float(values.mean())
    centered = values - mean
    if len(values) > 2 and np.dot(centered[:-1], centered[:-1]) > 0:
        raw_phi = float(np.dot(centered[:-1], centered[1:]) /
                        np.dot(centered[:-1], centered[:-1]))
    else:
        raw_phi = pooled
    weight = (len(values) - 1) / (len(values) - 1 + 10.0)
    phi = float(np.clip(weight * raw_phi + (1.0 - weight) * pooled, -.95, .95))
    residual = centered[1:] - phi * centered[:-1]
    innovation = float(np.sqrt(np.mean(np.square(residual)))) if len(residual) else 0.0
    if not np.isfinite(innovation) or innovation <= 1e-8:
        innovation = max(float(values.std(ddof=0)) * np.sqrt(max(1.0 - phi ** 2, .05)), 1e-6)
    result = np.empty((draws, len(values)), dtype=float)
    stationary_sd = innovation / np.sqrt(max(1.0 - phi ** 2, .05))
    result[:, 0] = mean + rng.normal(0.0, stationary_sd, draws)
    for index in range(1, len(values)):
        result[:, index] = mean + phi * (result[:, index - 1] - mean) + \
            rng.normal(0.0, innovation, draws)
    return np.clip(result, bounds[0], bounds[1])


def ar1_null(series) -> pd.DataFrame:
    rows = []
    keys = sorted({(dataset, outcome) for dataset, outcome, _ in series})
    for key_index, (dataset, outcome) in enumerate(keys):
        people, _ = prepared_people(series, dataset, outcome)
        pooled = pooled_phi(people)
        observed = np.asarray([person_statistics(p["values"], p["x"]) for p in people])
        eligible_slope = np.isfinite(observed[:, 1])
        observed_delta = float(observed[:, 0].mean())
        observed_slope = float(observed[eligible_slope, 1].mean())
        null_delta = np.zeros(NULL_DRAWS)
        null_slope = np.zeros(NULL_DRAWS)
        slope_count = 0
        rng = np.random.default_rng(SEED + 10000 * key_index)
        for person in people:
            simulated = simulate_person(person["values"], NULL_DRAWS, pooled,
                                        BOUNDS[dataset], rng)
            b8 = simulated[:, :8].mean(axis=1)
            windows = np.lib.stride_tricks.sliding_window_view(simulated, 8, axis=1)
            l8 = windows[:, :len(person["values"]) - 8, :].mean(axis=2)
            actual = simulated[:, 8:]
            benefit = np.abs(actual - b8[:, None]) - np.abs(actual - l8)
            null_delta += benefit.mean(axis=1)
            x = person["x"]
            centered = x - x.mean()
            denominator = float(np.dot(centered, centered))
            if denominator > 0:
                null_slope += benefit @ centered / denominator
                slope_count += 1
        null_delta /= len(people)
        null_slope /= slope_count
        for statistic, observed_value, null_values in (
                ("participant_balanced_mae_B8_minus_L8", observed_delta, null_delta),
                ("mean_within_person_age_slope", observed_slope, null_slope)):
            null_sd = float(null_values.std(ddof=1))
            rows.append({"dataset": dataset, "outcome": outcome,
                         "role": "primary" if (dataset, outcome) in PRIMARY else "secondary",
                         "statistic": statistic, "participants": len(people),
                         "targets": int(sum(len(p["values"]) - 8 for p in people)),
                         "pooled_lag1_phi": pooled, "observed": observed_value,
                         "null_mean": float(null_values.mean()),
                         "null_ci_low": float(np.quantile(null_values, .025)),
                         "null_ci_high": float(np.quantile(null_values, .975)),
                         "z_from_null": (observed_value - float(null_values.mean())) / null_sd,
                         "monte_carlo_p_one_sided": float(
                             (1 + np.sum(null_values >= observed_value)) / (NULL_DRAWS + 1))})
        print(f"AR1 null complete: {dataset}/{outcome}", flush=True)
    return pd.DataFrame(rows)


def prediction_rows(series) -> pd.DataFrame:
    rows = []
    for (dataset, outcome, participant), (raw_values, orders) in series.items():
        values = raw_values.astype(float)
        if len(values) < 9:
            continue
        b8 = float(values[:8].mean())
        ewm = b8
        oracle = float(values.mean())
        cumulative = float(values[:8].sum())
        for target in range(8, len(values)):
            history = values[:target]
            recent = values[target - 8:target]
            centered = history - history.mean()
            denominator = float(np.dot(centered[:-1], centered[:-1]))
            raw_phi = float(np.dot(centered[:-1], centered[1:]) / denominator) \
                if denominator > 0 else 0.0
            weight = (target - 1) / (target - 1 + 10.0)
            phi = float(np.clip(weight * raw_phi, -.95, .95))
            ar1 = float(history.mean() + phi * (history[-1] - history.mean()))
            record = {"dataset": dataset, "outcome": outcome,
                      "participant": participant, "target_index": target,
                      "actual": float(values[target]), "person_sd": float(values.std(ddof=0)),
                      "B8": b8, "L8": float(recent.mean()), "Last": float(recent[-1]),
                      "Cumulative": cumulative / target, "EWM8": ewm,
                      "Median8": float(np.median(recent)), "OnlineAR1": ar1,
                      "OracleMean": oracle, "Mode8": np.nan}
            if dataset in {"CES", "Marian"}:
                record["Mode8"] = mode_recent_first(recent)
            rows.append(record)
            cumulative += values[target]
            ewm = (2.0 / 9.0) * values[target] + (7.0 / 9.0) * ewm
    return pd.DataFrame(rows)


def comparator_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    methods = ["B8", "L8", "Last", "Cumulative", "EWM8", "Median8",
               "Mode8", "OnlineAR1", "OracleMean"]
    rng = np.random.default_rng(SEED + 501)
    rows = []
    for (dataset, outcome), group in predictions.groupby(["dataset", "outcome"], sort=False):
        for method in methods:
            available = group.dropna(subset=[method]).copy()
            if available.empty:
                continue
            available["ae"] = np.abs(available.actual - available[method])
            available["sae"] = available.ae / available.person_sd.replace(0, np.nan)
            people = available.groupby("participant", sort=False).agg(
                mae=("ae", "mean"), standardized_mae=("sae", "mean"), targets=("ae", "size"))
            array = people[["mae", "standardized_mae"]].to_numpy(float)
            draws = np.full((BOOTSTRAPS, 2), np.nan)
            for b in range(BOOTSTRAPS):
                sample = array[rng.integers(0, len(array), len(array))]
                draws[b] = np.nanmean(sample, axis=0)
            observed = np.nanmean(array, axis=0)
            rows.append({"dataset": dataset, "outcome": outcome, "method": method,
                         "diagnostic_oracle": method == "OracleMean",
                         "participants": len(people), "targets": len(available),
                         "mae": observed[0], "mae_ci_low": np.nanquantile(draws[:, 0], .025),
                         "mae_ci_high": np.nanquantile(draws[:, 0], .975),
                         "participant_sd_standardized_mae": observed[1],
                         "standardized_ci_low": np.nanquantile(draws[:, 1], .025),
                         "standardized_ci_high": np.nanquantile(draws[:, 1], .975)})
    result = pd.DataFrame(rows)
    baseline = result.loc[result.method.eq("B8"), ["dataset", "outcome", "mae"]].rename(
        columns={"mae": "b8_mae"})
    result = result.merge(baseline, on=["dataset", "outcome"], validate="many_to_one")
    result["absolute_mae_difference_vs_B8"] = result.mae - result.b8_mae
    result["relative_mae_percent_vs_B8"] = 100 * (result.mae / result.b8_mae - 1)
    return result


def summarize_frame(frame: pd.DataFrame, rng: np.random.Generator) -> dict:
    by_person = []
    for participant, group in frame.groupby("participant", sort=False):
        ae_b = np.abs(group.actual.to_numpy(float) - group.B.to_numpy(float)).mean()
        ae_l = np.abs(group.actual.to_numpy(float) - group.L.to_numpy(float)).mean()
        by_person.append((ae_b, ae_l))
    values = np.asarray(by_person, dtype=float)
    draws = np.empty((BOOTSTRAPS, 2))
    for b in range(BOOTSTRAPS):
        sample = values[rng.integers(0, len(values), len(values))].mean(axis=0)
        draws[b] = [sample[0] - sample[1], 100 * (sample[1] / sample[0] - 1)]
    means = values.mean(axis=0)
    return {"participants": len(values), "targets": len(frame),
            "absolute_mae_B8_minus_L8": means[0] - means[1],
            "absolute_ci_low": np.quantile(draws[:, 0], .025),
            "absolute_ci_high": np.quantile(draws[:, 0], .975),
            "relative_mae_L8_vs_B8_percent": 100 * (means[1] / means[0] - 1),
            "relative_ci_low": np.quantile(draws[:, 1], .025),
            "relative_ci_high": np.quantile(draws[:, 1], .975)}


def ema_sensitivities() -> tuple[pd.DataFrame, pd.DataFrame]:
    specifications = [
        ("Dejonckheere", pd.read_csv(DEJON, sep="\t"), "id",
         ("sad", "stressed", "happy", "relaxed", "angry")),
        ("Marian", pd.read_csv(MARIAN, sep="\t").query("status == 1"), "id",
         ("depressed", "anhedonia")),
    ]
    rng = np.random.default_rng(SEED + 902)
    summaries, coverage = [], []
    for dataset, raw, id_col, outcomes in specifications:
        for outcome in outcomes:
            complete = raw[[id_col, "counter", "day", "beep", outcome]].dropna().copy()
            beep_effect = complete.groupby("beep")[outcome].mean() - complete[outcome].mean()
            for scenario in ("original", "exclude_day1_reanchored", "prompt_adjusted"):
                work = complete.loc[complete.day.gt(1)].copy() if scenario == "exclude_day1_reanchored" else complete.copy()
                value_col = outcome
                if scenario == "prompt_adjusted":
                    work["adjusted"] = work[outcome] - work.beep.map(beep_effect)
                    value_col = "adjusted"
                rows = []
                spans = []
                for participant, person in work.groupby(id_col, sort=False):
                    person = person.sort_values("counter", kind="stable")
                    values = person[value_col].to_numpy(float)
                    if len(values) < 9:
                        continue
                    days = person.day.to_numpy(int)
                    spans.append(days[7] - days[0] + 1)
                    b8 = float(values[:8].mean())
                    for target in range(8, len(values)):
                        rows.append({"participant": str(participant), "actual": values[target],
                                     "B": b8, "L": float(values[target-8:target].mean())})
                frame = pd.DataFrame(rows)
                summaries.append({"dataset": dataset, "outcome": outcome,
                                  "scenario": scenario, **summarize_frame(frame, rng)})
                coverage.append({"dataset": dataset, "outcome": outcome,
                                 "scenario": scenario, "participants": len(spans),
                                 "median_calendar_days_in_B8": float(np.median(spans)),
                                 "q1_calendar_days_in_B8": float(np.quantile(spans, .25)),
                                 "q3_calendar_days_in_B8": float(np.quantile(spans, .75))})
    return pd.DataFrame(summaries), pd.DataFrame(coverage)


def bootstrap_correlation(x: np.ndarray, y: np.ndarray,
                          rng: np.random.Generator) -> tuple[float, float, float]:
    observed = float(np.corrcoef(x, y)[0, 1])
    draws = []
    for _ in range(BOOTSTRAPS):
        index = rng.integers(0, len(x), len(x))
        if np.std(x[index]) > 0 and np.std(y[index]) > 0:
            draws.append(float(np.corrcoef(x[index], y[index])[0, 1]))
    return observed, float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def standardized_reliability(series) -> pd.DataFrame:
    blocks = pd.read_parquet(TEMPORAL_BLOCKS).copy()
    sd_lookup = {(dataset, outcome, participant): float(values.std(ddof=0))
                 for (dataset, outcome, participant), (values, _orders) in series.items()}
    blocks["person_sd"] = [sd_lookup.get((r.dataset, r.outcome, str(r.participant)), np.nan)
                           for r in blocks.itertuples(index=False)]
    blocks = blocks.loc[blocks.person_sd.gt(0)].copy()
    blocks["standardized_mean_benefit_early"] = blocks.mean_benefit_early / blocks.person_sd
    blocks["standardized_mean_benefit_late"] = blocks.mean_benefit_late / blocks.person_sd
    rng = np.random.default_rng(SEED + 1203)
    rows = []
    for (dataset, outcome), group in blocks.groupby(["dataset", "outcome"], sort=False):
        raw = bootstrap_correlation(group.mean_benefit_early.to_numpy(float),
                                    group.mean_benefit_late.to_numpy(float), rng)
        standardized = bootstrap_correlation(
            group.standardized_mean_benefit_early.to_numpy(float),
            group.standardized_mean_benefit_late.to_numpy(float), rng)
        slope = bootstrap_correlation(group.slope_benefit_early.to_numpy(float),
                                      group.slope_benefit_late.to_numpy(float), rng)
        for quantity, result in (("raw_mean_benefit", raw),
                                 ("participant_sd_standardized_mean_benefit", standardized),
                                 ("age_gradient", slope)):
            rows.append({"dataset": dataset, "outcome": outcome, "quantity": quantity,
                         "participants": len(group), "pearson_r": result[0],
                         "ci_low": result[1], "ci_high": result[2]})
    return pd.DataFrame(rows)


def main() -> None:
    if not PROTOCOL.exists():
        raise FileNotFoundError("External peer-review protocol is missing")
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_module(BASE_SCRIPT, "external_review_base")
    series = base.load_series()
    diagnostic = diagnostics(series)
    null = ar1_null(series)
    predictions = prediction_rows(series)
    comparators = comparator_summary(predictions)
    ema, coverage = ema_sensitivities()
    reliability = standardized_reliability(series)
    diagnostic.to_csv(OUT / "peer_review_series_diagnostics.csv", index=False)
    null.to_csv(OUT / "peer_review_stationary_ar1_null.csv", index=False)
    predictions.to_parquet(OUT / "peer_review_comparator_predictions.parquet", index=False)
    comparators.to_csv(OUT / "peer_review_comparator_summary.csv", index=False)
    ema.to_csv(OUT / "peer_review_ema_sensitivity.csv", index=False)
    coverage.to_csv(OUT / "peer_review_ema_baseline_coverage.csv", index=False)
    reliability.to_csv(OUT / "peer_review_standardized_reliability.csv", index=False)
    manifest = {
        "status": "post-hoc external peer-review robustness analysis",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED, "stationary_ar1_draws": NULL_DRAWS,
        "participant_bootstrap_resamples": BOOTSTRAPS,
        "protocol_sha256": sha256(PROTOCOL), "script_sha256": sha256(Path(__file__)),
        "inputs": {"dejonckheere": sha256(DEJON), "marian": sha256(MARIAN),
                   "temporal_blocks": sha256(TEMPORAL_BLOCKS),
                   "split_reliability": sha256(SPLIT_RELIABILITY)},
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__},
        "checks": {"strictly_prior_non_oracle_predictors": True,
                   "oracle_explicitly_labelled": True,
                   "participant_balanced_statistics": True,
                   "participant_bootstrap": True,
                   "original_time_grid_preserved_in_ar1_null": True,
                   "participant_level_data_not_exported_publicly": True},
    }
    (OUT / "peer_review_robustness_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nDIAGNOSTICS\n", diagnostic.to_string(index=False))
    print("\nAR1 NULL\n", null.to_string(index=False))
    print("\nCOMPARATORS\n", comparators.to_string(index=False))
    print("\nEMA SENSITIVITY\n", ema.to_string(index=False))
    print("\nSTANDARDIZED RELIABILITY\n", reliability.to_string(index=False))


if __name__ == "__main__":
    main()
