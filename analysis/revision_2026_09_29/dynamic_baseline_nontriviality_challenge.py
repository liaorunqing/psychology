"""Equal-information and within-person permutation challenge for baseline freshness."""

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
DATA_ROOT = Path(os.environ.get("DPT_DATA_ROOT", ROOT / "data")).expanduser()
DATA = DATA_ROOT / "dejonckheere_openesm" / "0012_dejonckheere_ts.tsv"
PROTOCOL = ROOT / "direction_reset" / "69_dynamic_baseline_nontriviality_challenge_protocol_2026-09-30.md"
BASE_SCRIPT = Path(__file__).with_name("dejonckheere_baseline_age_confirmation.py")
SEED = 20260918
BOOTSTRAPS = 2000
PERMUTATIONS = 500
OUTCOMES = ("sad", "stressed", "happy", "relaxed", "angry")
PRIMARY = ("sad", "stressed")
WEIGHTS = np.linspace(0.0, 1.0, 41)
METHODS = ("B8", "L8", "R_online")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_module(BASE_SCRIPT, "dejon_confirmation_for_challenge")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_rows(data: pd.DataFrame, outcome: str, fold_map: dict[str, int],
               rng: np.random.Generator | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    for raw_id, person in data.sort_values(["id", "counter"]).groupby("id", sort=False):
        participant = str(raw_id)
        complete = person.loc[person[outcome].notna(), ["counter", outcome]]
        counters = complete.counter.to_numpy(dtype=int)
        values = complete[outcome].to_numpy(dtype=float)
        if rng is not None:
            values = values[rng.permutation(len(values))]
        if len(values) < 9:
            continue
        cumulative = np.cumsum(values)
        fixed_mean, fixed_last = float(values[:8].mean()), float(values[7])
        for index in range(8, len(values)):
            prior = values[:index]
            rows.append({
                "outcome": outcome, "participant": participant,
                "fold": fold_map[participant], "counter": int(counters[index]),
                "actual": float(values[index]),
                "baseline_age": int(counters[index] - counters[7]),
                "mean_8": fixed_mean, "last_8": fixed_last,
                "last8_mean": float(values[index - 8:index].mean()),
                "last8_last": float(values[index - 1]),
                "online_mean": float(cumulative[index - 1] / index),
                "online_last": float(values[index - 1]),
            })
    result = pd.DataFrame(rows)
    if len(result) != 7902 or result.participant.nunique() != 100:
        raise AssertionError(f"Target sample changed for {outcome}")
    if result.baseline_age.le(0).any():
        raise AssertionError("Non-positive baseline age")
    if not result.groupby("participant")[["mean_8", "last_8"]].nunique().eq(1).all().all():
        raise AssertionError("Frozen baseline changed")
    return result


def tune_weight_arrays(actual: np.ndarray, mean: np.ndarray, last: np.ndarray,
                       sample_weight: np.ndarray) -> float:
    prediction = mean[:, None] + (last - mean)[:, None] * WEIGHTS[None, :]
    loss = np.average(np.abs(actual[:, None] - prediction), axis=0,
                      weights=sample_weight)
    return float(WEIGHTS[int(np.argmin(loss))])


def method_columns(method: str) -> tuple[str, str]:
    return {"B8": ("mean_8", "last_8"),
            "L8": ("last8_mean", "last8_last"),
            "R_online": ("online_mean", "online_last")}[method]


def fit_predict(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    outputs, audits = [], []
    for fold in range(1, 6):
        train = rows.loc[rows.fold.ne(fold)].copy()
        test = rows.loc[rows.fold.eq(fold)].copy()
        if set(train.participant) & set(test.participant):
            raise AssertionError("Participant crosses fold")
        weights = BASE.participant_weights(train.participant)
        audit = {"outcome": rows.outcome.iloc[0], "fold": fold,
                 "train_participants": train.participant.nunique(),
                 "test_participants": test.participant.nunique(),
                 "train_targets": len(train), "test_targets": len(test)}
        for method in METHODS:
            mean_col, last_col = method_columns(method)
            weight = tune_weight_arrays(train.actual.to_numpy(dtype=float),
                                        train[mean_col].to_numpy(dtype=float),
                                        train[last_col].to_numpy(dtype=float), weights)
            test[method] = (test[mean_col] + weight *
                            (test[last_col] - test[mean_col]))
            audit[f"weight_{method}"] = weight
        outputs.append(test)
        audits.append(audit)
    result = pd.concat(outputs, ignore_index=True)
    if result[list(METHODS)].isna().any().any():
        raise AssertionError("Missing prediction")
    return result, pd.DataFrame(audits)


def add_errors_and_age(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    raw_age = np.log1p(result.baseline_age.to_numpy(dtype=float))
    result["log_age_scaled"] = raw_age / raw_age.std(ddof=0)
    for method in METHODS:
        residual = result.actual - result[method]
        result[f"ae_{method}"] = np.abs(residual)
        result[f"se_{method}"] = np.square(residual)
    result["benefit_L8"] = result.ae_B8 - result.ae_L8
    result["benefit_R_online"] = result.ae_B8 - result.ae_R_online
    result["age_stratum"] = pd.cut(
        result.baseline_age, [0, 14, 35, 63, 90],
        labels=["1-14", "15-35", "36-63", "64-90"], right=True).astype("string")
    if result.age_stratum.isna().any():
        raise AssertionError("Unassigned stratum")
    return result


def person_slope_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for participant, group in frame.groupby("participant", sort=False):
        x = group.log_age_scaled.to_numpy(dtype=float)
        centered = x - x.mean()
        denominator = float(np.square(centered).sum())
        record = {"participant": participant, "targets": len(group)}
        for method in ("L8", "R_online"):
            y = group[f"benefit_{method}"].to_numpy(dtype=float)
            record[f"slope_benefit_{method}"] = float(np.dot(centered, y) / denominator)
        rows.append(record)
    return pd.DataFrame(rows)


def summarize_observed(outcome: str, slopes: pd.DataFrame,
                       rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for method in ("L8", "R_online"):
        values = slopes[f"slope_benefit_{method}"].to_numpy(dtype=float)
        draws = np.empty(BOOTSTRAPS)
        for iteration in range(BOOTSTRAPS):
            draws[iteration] = values[rng.integers(0, len(values), len(values))].mean()
        rows.append({"outcome": outcome, "challenger": method,
                     "participants": len(values), "targets": int(slopes.targets.sum()),
                     "mean_slope": float(values.mean()),
                     "ci_low": float(np.quantile(draws, .025)),
                     "ci_high": float(np.quantile(draws, .975)),
                     "positive_slope_fraction": float((values > 0).mean())})
    return pd.DataFrame(rows)


def overall_contrasts(outcome: str, frame: pd.DataFrame,
                      rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    columns = [f"ae_{method}" for method in METHODS] + [f"se_{method}" for method in METHODS]
    by_person = frame.groupby("participant")[columns].mean()
    for method in ("L8", "R_online"):
        b8_ae, method_ae = by_person[["ae_B8", f"ae_{method}"]].mean()
        b8_se, method_se = by_person[["se_B8", f"se_{method}"]].mean()
        values = by_person[["ae_B8", f"ae_{method}", "se_B8", f"se_{method}"]].to_numpy()
        mae_draws, rmse_draws = np.empty(BOOTSTRAPS), np.empty(BOOTSTRAPS)
        for iteration in range(BOOTSTRAPS):
            means = values[rng.integers(0, len(values), len(values))].mean(axis=0)
            mae_draws[iteration] = 100 * (means[1] / means[0] - 1)
            rmse_draws[iteration] = 100 * (np.sqrt(means[3]) / np.sqrt(means[2]) - 1)
        rows.append({
            "outcome": outcome, "challenger": method,
            "participants": len(by_person), "targets": len(frame),
            "relative_mae_percent": float(100 * (method_ae / b8_ae - 1)),
            "mae_ci_low": float(np.quantile(mae_draws, .025)),
            "mae_ci_high": float(np.quantile(mae_draws, .975)),
            "relative_rmse_percent": float(100 * (np.sqrt(method_se) / np.sqrt(b8_se) - 1)),
            "rmse_ci_low": float(np.quantile(rmse_draws, .025)),
            "rmse_ci_high": float(np.quantile(rmse_draws, .975)),
        })
    return pd.DataFrame(rows)


def stratum_contrasts(outcome: str, frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stratum, group in frame.groupby("age_stratum", sort=False):
        by_person = group.groupby("participant")[["ae_B8", "ae_L8", "ae_R_online"]].mean()
        means = by_person.mean()
        for method in ("L8", "R_online"):
            rows.append({"outcome": outcome, "age_stratum": stratum,
                         "challenger": method, "participants": len(by_person),
                         "targets": len(group),
                         "relative_mae_percent": float(
                             100 * (means[f"ae_{method}"] / means.ae_B8 - 1))})
    return pd.DataFrame(rows)


def mean_benefit_slopes(frame: pd.DataFrame) -> tuple[float, float]:
    enriched = add_errors_and_age(frame)
    slopes = person_slope_table(enriched)
    return (float(slopes.slope_benefit_L8.mean()),
            float(slopes.slope_benefit_R_online.mean()))


def permutation_null(data: pd.DataFrame, outcome: str, fold_map: dict[str, int]) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + OUTCOMES.index(outcome) * 10000)
    rows = []
    for iteration in range(PERMUTATIONS):
        permuted = build_rows(data, outcome, fold_map, rng=rng)
        predictions, _ = fit_predict(permuted)
        slope_l8, slope_online = mean_benefit_slopes(predictions)
        rows.append({"outcome": outcome, "permutation": iteration + 1,
                     "slope_benefit_L8": slope_l8,
                     "slope_benefit_R_online": slope_online})
        if (iteration + 1) % 50 == 0:
            print(f"{outcome}: completed {iteration + 1}/{PERMUTATIONS} permutations", flush=True)
    return pd.DataFrame(rows)


def summarize_permutations(observed: pd.DataFrame,
                           null: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in PRIMARY:
        for method in ("L8", "R_online"):
            observed_slope = float(observed.loc[
                observed.outcome.eq(outcome) & observed.challenger.eq(method),
                "mean_slope"].iloc[0])
            values = null.loc[null.outcome.eq(outcome),
                              f"slope_benefit_{method}"].to_numpy(dtype=float)
            rows.append({
                "outcome": outcome, "challenger": method,
                "observed_slope": observed_slope,
                "null_mean": float(values.mean()),
                "null_ci_low": float(np.quantile(values, .025)),
                "null_ci_high": float(np.quantile(values, .975)),
                "monte_carlo_p_one_sided": float((1 + np.sum(values >= observed_slope)) /
                                                  (PERMUTATIONS + 1)),
            })
    return pd.DataFrame(rows)


def invariance_self_test() -> None:
    data = pd.DataFrame({"id": ["x"] * 12, "counter": np.arange(1, 13),
                         "sad": np.arange(12, dtype=float)})
    mapping = {"x": 1}
    first = build_rows_small(data, "sad", mapping)
    changed = data.copy()
    changed.loc[changed.counter.ge(9), "sad"] += 1000
    second = build_rows_small(changed, "sad", mapping)
    cols = ["mean_8", "last_8", "last8_mean", "last8_last", "online_mean", "online_last"]
    assert np.allclose(first.loc[first.counter.eq(9), cols],
                       second.loc[second.counter.eq(9), cols])


def build_rows_small(data: pd.DataFrame, outcome: str,
                     fold_map: dict[str, int]) -> pd.DataFrame:
    rows = []
    for raw_id, person in data.groupby("id"):
        participant = str(raw_id)
        complete = person.loc[person[outcome].notna()]
        counters = complete.counter.to_numpy(dtype=int)
        values = complete[outcome].to_numpy(dtype=float)
        for index in range(8, len(values)):
            rows.append({"participant": participant, "counter": counters[index],
                         "mean_8": values[:8].mean(), "last_8": values[7],
                         "last8_mean": values[index-8:index].mean(),
                         "last8_last": values[index-1],
                         "online_mean": values[:index].mean(),
                         "online_last": values[index-1]})
    return pd.DataFrame(rows)


def main() -> None:
    if not PROTOCOL.exists():
        raise RuntimeError("Frozen nontriviality protocol missing")
    invariance_self_test()
    data = pd.read_csv(DATA, sep="\t")
    data["id"] = data.id.astype(str)
    BASE.validate_grid(data)
    fold_map = BASE.make_fold_map(data.id)
    rng = np.random.default_rng(SEED)
    prediction_parts, audit_parts, slope_parts, overall_parts, stratum_parts = [], [], [], [], []
    for outcome in OUTCOMES:
        rows = build_rows(data, outcome, fold_map)
        predictions, audit = fit_predict(rows)
        predictions = add_errors_and_age(predictions)
        slopes = person_slope_table(predictions)
        slope_summary = summarize_observed(outcome, slopes, rng)
        prediction_parts.append(predictions); audit_parts.append(audit)
        slope_parts.append(slope_summary)
        overall_parts.append(overall_contrasts(outcome, predictions, rng))
        stratum_parts.append(stratum_contrasts(outcome, predictions))
    observed = pd.concat(slope_parts, ignore_index=True)
    null = pd.concat([permutation_null(data, outcome, fold_map) for outcome in PRIMARY],
                     ignore_index=True)
    permutation_summary = summarize_permutations(observed, null)
    primary_l8 = observed.loc[(observed.outcome.isin(PRIMARY)) &
                              observed.challenger.eq("L8")].set_index("outcome")
    permutation_l8 = permutation_summary.loc[
        permutation_summary.challenger.eq("L8")].set_index("outcome")
    passed = (primary_l8.mean_slope.gt(0) & primary_l8.ci_low.gt(0) &
              permutation_l8.monte_carlo_p_one_sided.le(.025))
    classification = ("nontriviality_confirmed" if passed.all() else
                      "partial_nontriviality" if passed.sum() == 1 else
                      "nontriviality_not_confirmed")
    predictions = pd.concat(prediction_parts, ignore_index=True)
    audits = pd.concat(audit_parts, ignore_index=True)
    overall = pd.concat(overall_parts, ignore_index=True)
    strata = pd.concat(stratum_parts, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(OUT / "dynamic_baseline_challenge_predictions.parquet", index=False)
    audits.to_csv(OUT / "dynamic_baseline_challenge_fold_audit.csv", index=False)
    observed.to_csv(OUT / "dynamic_baseline_challenge_slope_summary.csv", index=False)
    overall.to_csv(OUT / "dynamic_baseline_challenge_overall_contrasts.csv", index=False)
    strata.to_csv(OUT / "dynamic_baseline_challenge_stratum_contrasts.csv", index=False)
    null.to_parquet(OUT / "dynamic_baseline_challenge_permutation_null.parquet", index=False)
    permutation_summary.to_csv(OUT / "dynamic_baseline_challenge_permutation_summary.csv", index=False)
    manifest = {
        "status": "post-confirmation frozen nontriviality challenge",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "classification": classification, "seed": SEED,
        "bootstrap_resamples": BOOTSTRAPS, "permutations": PERMUTATIONS,
        "data_sha256": sha256(DATA), "protocol_sha256": sha256(PROTOCOL),
        "script_sha256": sha256(Path(__file__)),
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__},
        "checks": {"B8_uses_exactly_8_prior_reports": True,
                   "L8_uses_exactly_8_prior_reports": True,
                   "current_and_future_excluded": True,
                   "participant_disjoint_folds": True,
                   "within_participant_permutation_only": True,
                   "missing_positions_preserved": True,
                   "weights_retuned_in_each_permutation": True,
                   "same_fold_map_all_outcomes": True},
    }
    (OUT / "dynamic_baseline_challenge_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nCLASSIFICATION:", classification)
    print("\nOBSERVED SLOPES")
    print(observed.to_string(index=False))
    print("\nOVERALL CONTRASTS")
    print(overall.to_string(index=False))
    print("\nPERMUTATION SUMMARY")
    print(permutation_summary.to_string(index=False))


if __name__ == "__main__":
    main()
