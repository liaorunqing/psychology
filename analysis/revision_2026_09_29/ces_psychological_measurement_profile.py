"""Development-only CES profile of subjective-state stability and feedback."""

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
import sklearn
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "analysis" / "outputs" / "revision_2026_09_29"
EXPLORATION = ROOT / "analysis" / "outputs" / "exploration"
PROTOCOL = ROOT / "direction_reset" / "49_ces_psychological_measurement_profile_protocol_2026-09-29.md"
FOLDS = OUTPUT / "fair_history_adapter_fold_map.parquet"
SEED = 20260918
BOOTSTRAPS = 2000
OUTCOMES = ("stress", "gad2", "phq2", "low_self_esteem", "perceived_isolation")
METHODS = ("F", "M", "L", "R")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("ces_profile_" + path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = load_module(ROOT / "analysis" / "exploration" / "17_construct_aware_benchmark.py")
RECENCY = load_module(Path(__file__).with_name("training_only_recency_baseline.py"))


def measurement_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    gaps = data[["uid", "assessment_date"]].sort_values(["uid", "assessment_date"]).copy()
    gaps["adjacent_gap_days"] = gaps.groupby("uid").assessment_date.diff().dt.days
    positive = gaps.adjacent_gap_days.dropna()
    if not positive.gt(0).all():
        raise AssertionError("Nonpositive gap between retained assessments")
    for name in OUTCOMES:
        source = data[["uid", name]].copy()
        person_mean = source.groupby("uid")[name].mean()
        source["person_mean"] = source.uid.map(person_mean)
        within = source.groupby("uid").apply(
            lambda x: np.square(x[name] - x.person_mean).mean(), include_groups=False
        )
        grand_mean = float(person_mean.mean())
        between = float(np.square(person_mean - grand_mean).mean())
        within_mean = float(within.mean())
        if between + within_mean <= 0:
            raise AssertionError(f"Degenerate outcome: {name}")
        rows.append({"outcome": name, "participants": data.uid.nunique(),
                     "retained_assessments": len(data),
                     "observed_min": float(source[name].min()),
                     "observed_max": float(source[name].max()),
                     "zero_fraction": float(source[name].eq(0).mean()),
                     "person_equal_grand_mean": grand_mean,
                     "between_person_component": between,
                     "within_person_component": within_mean,
                     "descriptive_between_share": between / (between + within_mean),
                     "adjacent_gap_median_days": float(positive.median()),
                     "adjacent_gap_p10_days": float(positive.quantile(.1)),
                     "adjacent_gap_p90_days": float(positive.quantile(.9)),
                     "adjacent_gap_max_days": float(positive.max()),
                     "adjacent_gap_over_30_count": int(positive.gt(30).sum())})
    return pd.DataFrame(rows)


def trajectory_rows(data: pd.DataFrame, values: np.ndarray,
                    outcome: str, fold: int) -> pd.DataFrame:
    if len(data) != len(values):
        raise ValueError("Target array not aligned with rows")
    frame = data[["uid", "assessment_date"]].copy()
    frame["actual"] = values
    frame = frame.sort_values(["uid", "assessment_date"], kind="stable")
    rows = []
    for uid, person in frame.groupby("uid", sort=False):
        if len(person) <= 2:
            continue
        initial = person.actual.iloc[:2].to_numpy(dtype=float)
        first_two_mean = float(initial.mean())
        second_value = float(initial[-1])
        second_date = person.assessment_date.iloc[1]
        observed = list(initial)
        for rank, row in enumerate(person.iloc[2:].itertuples(index=False), start=1):
            age = float((row.assessment_date - second_date).total_seconds() / 86400)
            if age <= 0:
                raise AssertionError("Nonpositive age since second visible assessment")
            rows.append({"outcome": outcome, "fold": fold, "participant": str(uid),
                         "occasion": str(row.assessment_date), "future_rank": rank,
                         "actual": float(row.actual), "F_mean": first_two_mean,
                         "F_last": second_value, "M": float(np.mean(observed)),
                         "L": float(observed[-1]), "days_since_second": age})
            # This becomes visible only at subsequent targets for M/L/R.
            observed.append(float(row.actual))
    return pd.DataFrame(rows)


def tune_weight(rows: pd.DataFrame, mean_column: str,
                last_column: str) -> float:
    target = rows[["participant", "occasion", "actual", mean_column, last_column]].rename(
        columns={mean_column: "personal_running_mean", last_column: "last_report"}
    )
    weight, _, _ = RECENCY.tuning_weight(target)
    return weight


def confirm_current_label_invariance(data: pd.DataFrame, y: np.ndarray,
                                     outcome: str, fold: int) -> None:
    original = trajectory_rows(data, y, outcome, fold)
    changed = np.asarray(y, dtype=float).copy()
    ordering = data[["uid", "assessment_date"]].copy()
    rank = ordering.sort_values(["uid", "assessment_date"]).groupby("uid").cumcount()
    first_later_index = rank[rank.eq(2)].index
    if len(first_later_index) == 0:
        return
    # Perturb each person's first target only. Predictions at that very
    # occasion must not change; later predictions may change legitimately.
    for position in first_later_index:
        changed[data.index.get_loc(position)] += 1000.0
    altered = trajectory_rows(data, changed, outcome, fold)
    first = original[original.future_rank.eq(1)].reset_index(drop=True)
    other = altered[altered.future_rank.eq(1)].reset_index(drop=True)
    for column in ("F_mean", "F_last", "M", "L", "days_since_second"):
        np.testing.assert_allclose(first[column], other[column], atol=0, rtol=0)


def assemble(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_map = pd.read_parquet(FOLDS)
    mapping = fold_map[fold_map.dataset.eq("CES")].set_index("participant").fold
    data = data.copy()
    data["uid"] = data.uid.astype(str)
    if mapping.index.has_duplicates or set(data.uid) != set(mapping.index):
        raise AssertionError("CES fold assignment does not match current analytic table")
    data["outer_fold"] = data.uid.map(mapping).astype(int)
    pieces, audits = [], []
    for name in OUTCOMES:
        for fold in range(1, 6):
            train = data.loc[data.outer_fold.ne(fold)].copy()
            test = data.loc[data.outer_fold.eq(fold)].copy()
            if set(train.uid) & set(test.uid):
                raise AssertionError("Participant overlap")
            scaler = StandardScaler().fit(train[[name]])
            y_train = scaler.transform(train[[name]]).ravel()
            y_test = scaler.transform(test[[name]]).ravel()
            training = trajectory_rows(train, y_train, name, fold)
            testing = trajectory_rows(test, y_test, name, fold)
            confirm_current_label_invariance(test, y_test, name, fold)
            w_fixed = tune_weight(training, "F_mean", "F_last")
            w_online = tune_weight(training, "M", "L")
            testing["F"] = testing.F_mean + w_fixed * (testing.F_last - testing.F_mean)
            testing["R"] = testing.M + w_online * (testing.L - testing.M)
            pieces.append(testing[["outcome", "fold", "participant", "occasion",
                                   "future_rank", "actual", "days_since_second",
                                   "F", "M", "L", "R"]])
            audits.append({"outcome": name, "fold": fold,
                           "train_participants": training.participant.nunique(),
                           "train_targets": len(training),
                           "test_participants": testing.participant.nunique(),
                           "test_targets": len(testing),
                           "w_fixed": w_fixed, "w_online": w_online,
                           "current_target_invariance": True,
                           "participant_overlap": False})
            print(f"{name} fold {fold}/5 complete", flush=True)
    result = pd.concat(pieces, ignore_index=True)
    key = ["fold", "participant", "occasion", "future_rank"]
    reference = result[result.outcome.eq(OUTCOMES[0])][key].sort_values(key).reset_index(drop=True)
    for name in OUTCOMES[1:]:
        candidate = result[result.outcome.eq(name)][key].sort_values(key).reset_index(drop=True)
        pd.testing.assert_frame_equal(candidate, reference, check_exact=True)
    if result.duplicated(["outcome", *key]).any():
        raise AssertionError("Duplicate outcome target")
    return result, pd.DataFrame(audits)


def score(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    metric_rows, contrasts = [], []
    for name, group in predictions.groupby("outcome", sort=False):
        squared = pd.DataFrame({"participant": group.participant})
        absolute = pd.DataFrame({"participant": group.participant})
        for method in METHODS:
            residual = group.actual - group[method]
            squared[method] = np.square(residual)
            absolute[method] = np.abs(residual)
        person_mse = squared.groupby("participant")[list(METHODS)].mean()
        person_mae = absolute.groupby("participant")[list(METHODS)].mean()
        for method in METHODS:
            metric_rows.append({"outcome": name, "method": method,
                                "participants": len(person_mse), "targets": len(group),
                                "rmse_pb": float(np.sqrt(person_mse[method].mean())),
                                "mae_pb": float(person_mae[method].mean())})
        for reference, comparator in (("F", "R"), ("M", "R")):
            paired = person_mse[[reference, comparator]].to_numpy()
            old, new = np.sqrt(paired.mean(axis=0))
            draws = np.empty(BOOTSTRAPS)
            for iteration in range(BOOTSTRAPS):
                sample = paired[rng.integers(0, len(paired), len(paired))]
                ref, current = np.sqrt(sample.mean(axis=0))
                draws[iteration] = 100 * (current / ref - 1)
            lo, hi = np.quantile(draws, (.025, .975))
            contrasts.append({"outcome": name, "reference": reference,
                              "comparator": comparator,
                              "participants": len(paired), "targets": len(group),
                              "reference_rmse_pb": float(old),
                              "comparator_rmse_pb": float(new),
                              "relative_rmse_percent": float(100 * (new / old - 1)),
                              "ci_low_percent": float(lo), "ci_high_percent": float(hi)})
    return pd.DataFrame(metric_rows), pd.DataFrame(contrasts)


def age_strata(predictions: pd.DataFrame) -> pd.DataFrame:
    work = predictions.copy()
    work["stratum"] = pd.cut(work.days_since_second,
                             bins=[0, 90, 365, 730, np.inf],
                             labels=["[0,90)", "[90,365)", "[365,730)", "[730,+inf)"],
                             right=False)
    if work.stratum.isna().any():
        raise AssertionError("Age stratum not assigned")
    rows = []
    for name, outcome_rows in work.groupby("outcome", sort=False):
        counts = outcome_rows.groupby("participant", observed=True).stratum.nunique()
        complete_people = set(counts[counts.eq(4)].index)
        for population, table in (("available", outcome_rows),
                                  ("all_four_strata", outcome_rows.loc[
                                      outcome_rows.participant.isin(complete_people)])):
            for stratum, group in table.groupby("stratum", observed=True):
                participants = group.participant.nunique()
                by_person = group.groupby("participant")
                rmse_f = float(np.sqrt(by_person.apply(
                    lambda x: np.square(x.actual - x.F).mean(), include_groups=False
                ).mean()))
                rmse_r = float(np.sqrt(by_person.apply(
                    lambda x: np.square(x.actual - x.R).mean(), include_groups=False
                ).mean()))
                rows.append({"outcome": name, "population": population,
                             "age_days": str(stratum), "participants": participants,
                             "targets": len(group), "F_rmse_pb": rmse_f,
                             "R_rmse_pb": rmse_r,
                             "R_vs_F_relative_rmse_percent": 100 * (rmse_r / rmse_f - 1),
                             "minimum_10_participants": participants >= 10})
    return pd.DataFrame(rows)


def main() -> None:
    if not PROTOCOL.exists():
        raise RuntimeError("Protocol missing")
    data = BENCHMARK.build_wide_table()
    if data.uid.nunique() != 110 or len(data) != 5407:
        raise AssertionError("CES development panel size has changed")
    measurement = measurement_summary(data)
    predictions, fold_audit = assemble(data)
    metrics, comparisons = score(predictions)
    strata = age_strata(predictions)
    paths = {
        "measurement": OUTPUT / "ces_psychological_measurement_summary.csv",
        "predictions": OUTPUT / "ces_psychological_measurement_predictions.parquet",
        "fold_audit": OUTPUT / "ces_psychological_measurement_fold_audit.csv",
        "metrics": OUTPUT / "ces_psychological_measurement_metrics.csv",
        "contrasts": OUTPUT / "ces_psychological_measurement_contrasts.csv",
        "age_strata": OUTPUT / "ces_psychological_measurement_age_strata.csv",
    }
    measurement.to_csv(paths["measurement"], index=False)
    predictions.to_parquet(paths["predictions"], index=False)
    fold_audit.to_csv(paths["fold_audit"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    comparisons.to_csv(paths["contrasts"], index=False)
    strata.to_csv(paths["age_strata"], index=False)
    inputs = (EXPLORATION / "ces_development_multiscale_features.parquet",
              EXPLORATION / "ces_exploration_participant_split.parquet", FOLDS)
    manifest = {"status": "post-inspection CES development outcome-only exploration",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "seed": SEED, "bootstrap_draws": BOOTSTRAPS,
                "protocol_sha256": sha256(PROTOCOL),
                "source_sha256": sha256(Path(__file__)),
                "inputs_sha256": {str(path.relative_to(ROOT)): sha256(path)
                                  for path in inputs},
                "outputs_sha256": {key: sha256(path) for key, path in paths.items()},
                "software": {"python": platform.python_version(), "pandas": pd.__version__,
                             "numpy": np.__version__, "sklearn": sklearn.__version__}}
    (OUTPUT / "ces_psychological_measurement_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("MEASUREMENT\n", measurement.to_string(index=False), flush=True)
    print("CONTRASTS\n", comparisons.to_string(index=False), flush=True)
    print("WEIGHTS\n", fold_audit.groupby("outcome")[["w_fixed", "w_online"]].agg(
        ["min", "median", "max"]
    ).to_string(), flush=True)


if __name__ == "__main__":
    main()
