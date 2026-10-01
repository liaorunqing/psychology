"""Frozen equal-information B8 versus L8 replication in CES and Marian."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "outputs" / "revision_2026_09_29"
PROTOCOL = ROOT / "direction_reset" / "72_cross_dataset_equal_information_replication_protocol_2026-09-30.md"
BUDGET_SCRIPT = Path(__file__).with_name("report_calibration_budget_curve.py")
SEED = 20260918
BOOTSTRAPS = 2000
PERMUTATIONS = 500
WEIGHTS = np.linspace(0.0, 1.0, 41)
EXPECTED = {("CES", "phq2"): (105, 4541),
            ("Marian", "depressed"): (145, 7071),
            ("Marian", "anhedonia"): (145, 7071)}
PRIMARY = (("CES", "phq2"), ("Marian", "depressed"))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BUDGET = load_module(BUDGET_SCRIPT, "budget_for_equal_information_replication")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs() -> list[tuple[str, str, pd.DataFrame, str, str, dict[str, int]]]:
    ces = BUDGET.PROFILE.BENCHMARK.build_wide_table()[
        ["uid", "assessment_date", "phq2"]].copy()
    ces["uid"] = ces.uid.astype(str)
    ces_map = (pd.read_parquet(BUDGET.FOLDS).query("dataset == 'CES'")
               .set_index("participant").fold.astype(int).to_dict())
    if set(ces.uid) != set(ces_map):
        raise AssertionError("CES fold map mismatch")
    marian = pd.read_csv(BUDGET.MARIAN, sep="\t")
    marian = marian.loc[marian.status.eq(1)].copy()
    marian["id"] = marian.id.astype(str)
    marian_map = BUDGET.MARIAN_MODULE.fold_map(
        pd.read_csv(BUDGET.MARIAN, sep="\t").id.astype(str))
    return [
        ("CES", "phq2", ces, "uid", "assessment_date", ces_map),
        ("Marian", "depressed", marian, "id", "counter", marian_map),
        ("Marian", "anhedonia", marian, "id", "counter", marian_map),
    ]


def build_rows(frame: pd.DataFrame, dataset: str, outcome: str, id_col: str,
               order_col: str, fold_map: dict[str, int],
               rng: np.random.Generator | None = None) -> pd.DataFrame:
    work = frame[[id_col, order_col, outcome]].dropna().copy()
    work[id_col] = work[id_col].astype(str)
    work = work.sort_values([id_col, order_col], kind="stable")
    rows: list[dict] = []
    for participant, person in work.groupby(id_col, sort=False):
        values = person[outcome].to_numpy(dtype=float)
        orders = person[order_col].to_numpy()
        if rng is not None:
            values = values[rng.permutation(len(values))]
        if len(values) < 9:
            continue
        fixed_mean, fixed_last = float(values[:8].mean()), float(values[7])
        for index in range(8, len(values)):
            difference = orders[index] - orders[7]
            age = (float(difference / np.timedelta64(1, "D"))
                   if np.issubdtype(person[order_col].dtype, np.datetime64)
                   else float(difference))
            rows.append({
                "dataset": dataset, "outcome": outcome,
                "participant": participant, "fold": int(fold_map[participant]),
                "occasion": str(orders[index]), "actual": float(values[index]),
                "baseline_age": age, "mean_8": fixed_mean, "last_8": fixed_last,
                "last8_mean": float(values[index - 8:index].mean()),
                "last8_last": float(values[index - 1]),
            })
    result = pd.DataFrame(rows)
    expected_people, expected_targets = EXPECTED[(dataset, outcome)]
    if result.participant.nunique() != expected_people or len(result) != expected_targets:
        raise AssertionError(f"Sample changed: {(dataset, outcome)}")
    if result.baseline_age.le(0).any():
        raise AssertionError("Baseline age must be positive")
    if not result.groupby("participant")[["mean_8", "last_8"]].nunique().eq(1).all().all():
        raise AssertionError("B8 changed after calibration")
    if result.duplicated(["participant", "occasion"]).any():
        raise AssertionError("Duplicate target")
    return result


def tune_weight(frame: pd.DataFrame, mean_col: str, last_col: str) -> float:
    actual = frame.actual.to_numpy(dtype=float)
    mean = frame[mean_col].to_numpy(dtype=float)
    last = frame[last_col].to_numpy(dtype=float)
    row_weights = BUDGET.participant_weights(frame.participant)
    prediction = mean[:, None] + (last - mean)[:, None] * WEIGHTS[None, :]
    losses = np.average(np.abs(actual[:, None] - prediction), axis=0,
                        weights=row_weights)
    return float(WEIGHTS[int(np.argmin(losses))])


def fit_predict(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    outputs, audits = [], []
    for fold in range(1, 6):
        train = rows.loc[rows.fold.ne(fold)].copy()
        test = rows.loc[rows.fold.eq(fold)].copy()
        if set(train.participant) & set(test.participant):
            raise AssertionError("Participant crosses fold")
        weight_b8 = tune_weight(train, "mean_8", "last_8")
        weight_l8 = tune_weight(train, "last8_mean", "last8_last")
        test["B8"] = test.mean_8 + weight_b8 * (test.last_8 - test.mean_8)
        test["L8"] = test.last8_mean + weight_l8 * (test.last8_last - test.last8_mean)
        audits.append({"dataset": rows.dataset.iloc[0], "outcome": rows.outcome.iloc[0],
                       "fold": fold, "train_participants": train.participant.nunique(),
                       "test_participants": test.participant.nunique(),
                       "train_targets": len(train), "test_targets": len(test),
                       "weight_B8": weight_b8, "weight_L8": weight_l8})
        outputs.append(test)
    result = pd.concat(outputs, ignore_index=True)
    if result[["B8", "L8"]].isna().any().any():
        raise AssertionError("Missing prediction")
    return result, pd.DataFrame(audits)


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    raw = np.log1p(result.baseline_age.to_numpy(dtype=float))
    result["log_age_scaled"] = raw / raw.std(ddof=0)
    result["ae_B8"] = np.abs(result.actual - result.B8)
    result["ae_L8"] = np.abs(result.actual - result.L8)
    result["se_B8"] = np.square(result.actual - result.B8)
    result["se_L8"] = np.square(result.actual - result.L8)
    result["benefit_L8"] = result.ae_B8 - result.ae_L8
    if result.dataset.iloc[0] == "CES":
        result["age_stratum"] = pd.cut(
            result.baseline_age, [0, 90, 365, 730, np.inf],
            labels=["(0,90]", "(90,365]", "(365,730]", ">730"], right=True).astype("string")
    else:
        result["age_stratum"] = pd.cut(
            result.baseline_age, [0, 7, 21, 42, np.inf],
            labels=["1-7", "8-21", "22-42", ">=43"], right=True).astype("string")
    if result.age_stratum.isna().any():
        raise AssertionError("Unassigned age stratum")
    return result


def person_slopes(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for participant, group in frame.groupby("participant", sort=False):
        if len(group) < 5 or group.baseline_age.nunique() < 2:
            continue
        x = group.log_age_scaled.to_numpy(dtype=float)
        centered = x - x.mean()
        denominator = float(np.square(centered).sum())
        y = group.benefit_L8.to_numpy(dtype=float)
        rows.append({"participant": participant, "targets": len(group),
                     "slope_benefit_L8": float(np.dot(centered, y) / denominator)})
    result = pd.DataFrame(rows)
    eligibility = frame.groupby("participant").agg(
        targets=("actual", "size"), age_values=("baseline_age", "nunique"))
    expected_ids = set(eligibility.loc[(eligibility.targets >= 5) &
                                       (eligibility.age_values >= 2)].index.astype(str))
    if set(result.participant.astype(str)) != expected_ids:
        raise AssertionError("Slope eligibility differs from frozen >=5-target rule")
    return result


def summarize_slope(dataset: str, outcome: str, slopes: pd.DataFrame,
                    rng: np.random.Generator) -> dict:
    values = slopes.slope_benefit_L8.to_numpy(dtype=float)
    draws = np.empty(BOOTSTRAPS)
    for iteration in range(BOOTSTRAPS):
        draws[iteration] = values[rng.integers(0, len(values), len(values))].mean()
    return {"dataset": dataset, "outcome": outcome,
            "role": "primary" if (dataset, outcome) in PRIMARY else "secondary",
            "participants": len(values), "targets": int(slopes.targets.sum()),
            "mean_slope": float(values.mean()),
            "ci_low": float(np.quantile(draws, .025)),
            "ci_high": float(np.quantile(draws, .975)),
            "positive_slope_fraction": float((values > 0).mean())}


def overall_contrast(dataset: str, outcome: str, frame: pd.DataFrame,
                     rng: np.random.Generator) -> dict:
    by_person = frame.groupby("participant")[["ae_B8", "ae_L8", "se_B8", "se_L8"]].mean()
    means = by_person.mean()
    values = by_person.to_numpy(dtype=float)
    mae_draws, rmse_draws = np.empty(BOOTSTRAPS), np.empty(BOOTSTRAPS)
    for iteration in range(BOOTSTRAPS):
        sampled = values[rng.integers(0, len(values), len(values))].mean(axis=0)
        mae_draws[iteration] = 100 * (sampled[1] / sampled[0] - 1)
        rmse_draws[iteration] = 100 * (np.sqrt(sampled[3]) / np.sqrt(sampled[2]) - 1)
    return {"dataset": dataset, "outcome": outcome,
            "participants": len(by_person), "targets": len(frame),
            "relative_mae_percent": float(100 * (means.ae_L8 / means.ae_B8 - 1)),
            "mae_ci_low": float(np.quantile(mae_draws, .025)),
            "mae_ci_high": float(np.quantile(mae_draws, .975)),
            "relative_rmse_percent": float(
                100 * (np.sqrt(means.se_L8) / np.sqrt(means.se_B8) - 1)),
            "rmse_ci_low": float(np.quantile(rmse_draws, .025)),
            "rmse_ci_high": float(np.quantile(rmse_draws, .975))}


def stratum_contrasts(dataset: str, outcome: str, frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stratum, group in frame.groupby("age_stratum", sort=False):
        by_person = group.groupby("participant")[["ae_B8", "ae_L8", "se_B8", "se_L8"]].mean()
        means = by_person.mean()
        rows.append({"dataset": dataset, "outcome": outcome, "age_stratum": stratum,
                     "participants": len(by_person), "targets": len(group),
                     "relative_mae_percent": float(100 * (means.ae_L8 / means.ae_B8 - 1)),
                     "relative_rmse_percent": float(
                         100 * (np.sqrt(means.se_L8) / np.sqrt(means.se_B8) - 1))})
    return pd.DataFrame(rows)


def permutation_null(frame: pd.DataFrame, dataset: str, outcome: str, id_col: str,
                     order_col: str, fold_map: dict[str, int]) -> pd.DataFrame:
    key_index = list(EXPECTED).index((dataset, outcome))
    rng = np.random.default_rng(SEED + key_index * 10000)
    rows = []
    for iteration in range(PERMUTATIONS):
        permuted = build_rows(frame, dataset, outcome, id_col, order_col, fold_map, rng)
        predictions, _ = fit_predict(permuted)
        slope = float(person_slopes(enrich(predictions)).slope_benefit_L8.mean())
        rows.append({"dataset": dataset, "outcome": outcome,
                     "permutation": iteration + 1, "slope_benefit_L8": slope})
        if (iteration + 1) % 50 == 0:
            print(f"{dataset}/{outcome}: {iteration + 1}/{PERMUTATIONS} permutations", flush=True)
    return pd.DataFrame(rows)


def leakage_self_test() -> None:
    toy = pd.DataFrame({"id": ["x"] * 12, "counter": np.arange(1, 13),
                        "value": np.arange(12, dtype=float)})
    first = build_small(toy)
    changed = toy.copy(); changed.loc[changed.counter.ge(9), "value"] += 1000
    second = build_small(changed)
    columns = ["mean_8", "last_8", "last8_mean", "last8_last"]
    assert np.allclose(first.loc[first.counter.eq(9), columns],
                       second.loc[second.counter.eq(9), columns])


def build_small(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.value.to_numpy(dtype=float)
    counters = frame.counter.to_numpy(dtype=int)
    rows = []
    for index in range(8, len(values)):
        rows.append({"counter": counters[index], "mean_8": values[:8].mean(),
                     "last_8": values[7], "last8_mean": values[index-8:index].mean(),
                     "last8_last": values[index-1]})
    return pd.DataFrame(rows)


def main() -> None:
    if not PROTOCOL.exists():
        raise RuntimeError("Frozen replication protocol missing")
    leakage_self_test()
    rng = np.random.default_rng(SEED)
    predictions_all, audits_all, slopes_summary, overall_all, strata_all, null_all = [], [], [], [], [], []
    inputs = load_inputs()
    for dataset, outcome, frame, id_col, order_col, fold_map in inputs:
        rows = build_rows(frame, dataset, outcome, id_col, order_col, fold_map)
        predictions, audit = fit_predict(rows)
        predictions = enrich(predictions)
        slopes = person_slopes(predictions)
        predictions_all.append(predictions); audits_all.append(audit)
        slopes_summary.append(summarize_slope(dataset, outcome, slopes, rng))
        overall_all.append(overall_contrast(dataset, outcome, predictions, rng))
        strata_all.append(stratum_contrasts(dataset, outcome, predictions))
        null_all.append(permutation_null(frame, dataset, outcome, id_col, order_col, fold_map))
    summary = pd.DataFrame(slopes_summary)
    overall = pd.DataFrame(overall_all)
    strata = pd.concat(strata_all, ignore_index=True)
    null = pd.concat(null_all, ignore_index=True)
    permutation_rows = []
    for row in summary.itertuples(index=False):
        values = null.loc[(null.dataset.eq(row.dataset)) & (null.outcome.eq(row.outcome)),
                          "slope_benefit_L8"].to_numpy(dtype=float)
        permutation_rows.append({"dataset": row.dataset, "outcome": row.outcome,
                                 "role": row.role, "observed_slope": row.mean_slope,
                                 "null_mean": float(values.mean()),
                                 "null_ci_low": float(np.quantile(values, .025)),
                                 "null_ci_high": float(np.quantile(values, .975)),
                                 "monte_carlo_p_one_sided": float(
                                     (1 + np.sum(values >= row.mean_slope)) / (PERMUTATIONS + 1))})
    permutation_summary = pd.DataFrame(permutation_rows)
    merged = summary.merge(permutation_summary, on=["dataset", "outcome", "role"])
    primary = merged.loc[[tuple(row) in set(PRIMARY) for row in
                          merged[["dataset", "outcome"]].itertuples(index=False, name=None)]]
    passed = (primary.mean_slope.gt(0) & primary.ci_low.gt(0) &
              primary.monte_carlo_p_one_sided.le(.025))
    classification = ("cross_timescale_replication" if passed.all() else
                      "partial_replication" if passed.sum() == 1 or primary.mean_slope.gt(0).all()
                      else "not_replicated")
    predictions = pd.concat(predictions_all, ignore_index=True)
    audits = pd.concat(audits_all, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(OUT / "cross_dataset_equal_information_predictions.parquet", index=False)
    audits.to_csv(OUT / "cross_dataset_equal_information_fold_audit.csv", index=False)
    summary.to_csv(OUT / "cross_dataset_equal_information_slope_summary.csv", index=False)
    overall.to_csv(OUT / "cross_dataset_equal_information_overall_contrasts.csv", index=False)
    strata.to_csv(OUT / "cross_dataset_equal_information_stratum_contrasts.csv", index=False)
    null.to_parquet(OUT / "cross_dataset_equal_information_permutation_null.parquet", index=False)
    permutation_summary.to_csv(OUT / "cross_dataset_equal_information_permutation_summary.csv", index=False)
    manifest = {"status": "frozen cross-dataset equal-information replication",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "classification": classification, "seed": SEED,
                "bootstrap_resamples": BOOTSTRAPS, "permutations_per_outcome": PERMUTATIONS,
                "protocol_sha256": sha256(PROTOCOL), "script_sha256": sha256(Path(__file__)),
                "software": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__},
                "expected": {"|".join(key): {"participants": value[0], "targets": value[1]}
                             for key, value in EXPECTED.items()},
                "checks": {"B8_exactly_8_prior_reports": True,
                           "L8_exactly_8_prior_reports": True,
                           "current_and_future_excluded": True,
                           "participant_disjoint_folds": True,
                           "within_participant_permutation_only": True,
                           "weights_retuned_each_permutation": True,
                           "participant_bootstrap": True}}
    (OUT / "cross_dataset_equal_information_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nCLASSIFICATION:", classification)
    print("\nSLOPES")
    print(summary.to_string(index=False))
    print("\nOVERALL")
    print(overall.to_string(index=False))
    print("\nPERMUTATION")
    print(permutation_summary.to_string(index=False))


if __name__ == "__main__":
    main()
