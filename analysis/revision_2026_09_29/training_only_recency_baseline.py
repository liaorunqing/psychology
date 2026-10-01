"""Post-hoc four-cohort training-only self-report recency comparator."""

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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "analysis" / "outputs" / "revision_2026_09_29"
TABLES = ROOT / "analysis" / "outputs" / "tables"
OLD = ROOT / "analysis" / "outputs" / "exploration"
PROTOCOL = ROOT / "direction_reset" / "41_training_only_recency_baseline_protocol_2026-09-29.md"
FOLDS = OUTPUT / "fair_history_adapter_fold_map.parquet"
FAIR = OUTPUT / "fair_history_adapter_predictions.parquet"
SEED = 20260918
BOOTSTRAPS = 2000
COHORTS = ("CES", "PSYCHE-D", "MobileWell400+", "CrossCheck")
METHODS = ("personal_running_mean", "last_report", "recency_tuned", "h_online", "hs_online")
CONTRASTS = (
    ("personal_running_mean", "recency_tuned", "recency_vs_running_mean"),
    ("recency_tuned", "h_online", "history_model_vs_recency"),
    ("recency_tuned", "hs_online", "sensing_model_vs_recency"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source(filename: str):
    path = ROOT / "analysis" / "exploration" / filename
    spec = importlib.util.spec_from_file_location("recency_" + path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cohort_table(dataset: str) -> tuple[pd.DataFrame, str, str, list[str], str]:
    if dataset == "CES":
        module = load_source("17_construct_aware_benchmark.py")
        return module.build_wide_table(), "uid", "assessment_date", list(module.OUTCOMES), "distress_factor"
    if dataset == "PSYCHE-D":
        data = pd.read_parquet(TABLES / "psyche_analysis.parquet")
        return data[data.main_complete].copy(), "participant_id", "sample_month", ["phq9_end"], "phq9_end"
    if dataset == "MobileWell400+":
        module = load_source("26_mobilewell_sgpeba_confirmation.py")
        data = pd.read_parquet(OUTPUT / "mobilewell_sgpeba_analysis_table.parquet")
        return data, "participant", "timestamp", list(module.OUTCOMES), module.PRIMARY_OUTCOME
    if dataset == "CrossCheck":
        module = load_source("28_crosscheck_gate_confirmation.py")
        data = pd.read_parquet(OUTPUT / "crosscheck_confirmation_table.parquet")
        return data, "participant", "date", [module.PRIMARY_OUTCOME], module.PRIMARY_OUTCOME
    raise ValueError(dataset)


def scale_target(dataset: str, train: pd.DataFrame, test: pd.DataFrame,
                 outcomes: list[str], primary: str) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(train[outcomes])
    train_y = scaler.transform(train[outcomes])
    test_y = scaler.transform(test[outcomes])
    if dataset != "CES":
        index = outcomes.index(primary)
        return train_y[:, index], test_y[:, index]
    distress = [outcomes.index(name) for name in ("stress", "gad2", "phq2", "low_self_esteem")]
    pca = PCA(n_components=1, random_state=SEED).fit(train_y[:, distress])
    component = pca.components_[0].copy()
    if component.sum() < 0:
        component *= -1
    factor_sd = float(np.std(train_y[:, distress] @ component, ddof=1))
    return (train_y[:, distress] @ component / factor_sd,
            test_y[:, distress] @ component / factor_sd)


def history_rows(data: pd.DataFrame, id_column: str, time_column: str,
                 outcomes: np.ndarray, dataset: str, fold: int) -> pd.DataFrame:
    work = data[[id_column, time_column]].copy()
    work[id_column] = work[id_column].astype(str)
    work["actual"] = outcomes
    rows = []
    for participant, person in work.groupby(id_column, sort=False):
        person = person.sort_values(time_column, kind="stable")
        if len(person) <= 2:
            continue
        observed = list(person.actual.iloc[:2].to_numpy(dtype=float))
        for rank, row in enumerate(person.iloc[2:].itertuples(index=False), start=1):
            current = float(row.actual)
            prior_mean = float(np.mean(observed))
            last = observed[-1]
            rows.append({"dataset": dataset, "fold": fold,
                         "participant": str(participant),
                         "occasion": str(getattr(row, time_column)),
                         "future_rank": rank, "actual": current,
                         "personal_running_mean": prior_mean,
                         "last_report": last})
            # Only subsequent targets may use this observed outcome.
            observed.append(current)
    return pd.DataFrame(rows)


def tuning_weight(training_rows: pd.DataFrame) -> tuple[float, int, int]:
    if training_rows.empty:
        raise ValueError("No eligible training target")
    if training_rows.duplicated(["participant", "occasion"]).any():
        raise AssertionError("Duplicate training target")
    delta = training_rows.last_report - training_rows.personal_running_mean
    error = training_rows.personal_running_mean - training_rows.actual
    work = pd.DataFrame({"participant": training_rows.participant,
                         "numerator": error * delta, "denominator": delta * delta})
    by_person = work.groupby("participant")[["numerator", "denominator"]].mean()
    denominator = float(by_person.denominator.mean())
    weight = 0.0 if denominator <= 0 else float(np.clip(-by_person.numerator.mean() / denominator, 0, 1))
    return weight, len(by_person), len(training_rows)


def assemble() -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_map = pd.read_parquet(FOLDS)
    fair = pd.read_parquet(FAIR)
    pieces, audits = [], []
    for dataset in COHORTS:
        data, id_column, time_column, outcomes, primary = cohort_table(dataset)
        data[id_column] = data[id_column].astype(str)
        mapping = fold_map[fold_map.dataset.eq(dataset)].set_index("participant").fold
        if mapping.index.has_duplicates or set(data[id_column]) != set(mapping.index):
            raise AssertionError(f"Incomplete participant fold map for {dataset}")
        data["outer_fold"] = data[id_column].map(mapping).astype(int)
        for fold in range(1, 6):
            train = data[data.outer_fold.ne(fold)].copy()
            test = data[data.outer_fold.eq(fold)].copy()
            if set(train[id_column]) & set(test[id_column]):
                raise AssertionError("Participant crosses outer fold")
            y_train, y_test = scale_target(dataset, train, test, outcomes, primary)
            train_rows = history_rows(train, id_column, time_column, y_train, dataset, fold)
            test_rows = history_rows(test, id_column, time_column, y_test, dataset, fold)
            weight, train_people, train_records = tuning_weight(train_rows)
            test_rows["recency_tuned"] = (test_rows.personal_running_mean +
                                           weight * (test_rows.last_report - test_rows.personal_running_mean))
            pieces.append(test_rows)
            audits.append({"dataset": dataset, "fold": fold, "recency_weight": weight,
                           "train_people": train_people, "train_future_records": train_records,
                           "test_people": test_rows.participant.nunique(),
                           "test_future_records": len(test_rows)})
            print(f"{dataset} fold {fold}/5: w={weight:.4f}", flush=True)
    reconstructed = pd.concat(pieces, ignore_index=True)
    if reconstructed.duplicated(["dataset", "participant", "occasion"]).any():
        raise AssertionError("Duplicate reconstructed target")
    keys = ["dataset", "fold", "participant", "occasion", "future_rank"]
    comparable = fair[keys + ["actual", "personal_running_mean", "last_report", "h_online", "hs_online"]]
    joined = reconstructed.merge(comparable, on=keys, how="outer", validate="one_to_one",
                                 indicator=True, suffixes=("_new", "_fair"))
    if len(joined) != len(fair) or not joined._merge.eq("both").all():
        raise AssertionError("Saved fair targets do not match reconstructed targets")
    for name in ("actual", "personal_running_mean", "last_report"):
        if not np.allclose(joined[f"{name}_new"], joined[f"{name}_fair"], atol=1e-10, rtol=0):
            maximum = float(np.max(np.abs(joined[f"{name}_new"] - joined[f"{name}_fair"])))
            raise AssertionError(f"{dataset} {name} reconstruction differs: {maximum}")
    result = joined[keys + ["actual_fair", "personal_running_mean_fair", "last_report_fair",
                            "recency_tuned", "h_online", "hs_online"]].rename(columns={
                                "actual_fair": "actual", "personal_running_mean_fair": "personal_running_mean",
                                "last_report_fair": "last_report"})
    return result, pd.DataFrame(audits)


def summarize(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    metrics, contrasts, horizons = [], [], []
    for dataset in COHORTS:
        group = predictions[predictions.dataset.eq(dataset)]
        if group.empty:
            raise AssertionError(f"Missing {dataset} predictions")
        errors = group[["participant", "future_rank"]].copy()
        for method in METHODS:
            errors[method] = np.square(group.actual - group[method])
        by_person = errors.groupby("participant")[list(METHODS)].mean()
        for method in METHODS:
            metrics.append({"dataset": dataset, "method": method,
                            "participants": len(by_person), "records": len(group),
                            "rmse_pb": float(np.sqrt(by_person[method].mean()))})
        for reference, challenger, name in CONTRASTS:
            pair = by_person[[reference, challenger]].to_numpy()
            old, new = np.sqrt(pair.mean(axis=0))
            draws = np.empty(BOOTSTRAPS)
            for b in range(BOOTSTRAPS):
                sample = pair[rng.integers(0, len(pair), len(pair))]
                ref, test = np.sqrt(sample.mean(axis=0))
                draws[b] = 100 * (test / ref - 1)
            low, high = np.quantile(draws, [0.025, 0.975])
            contrasts.append({"dataset": dataset, "contrast": name,
                              "reference": reference, "challenger": challenger,
                              "participants": len(pair), "records": len(group),
                              "reference_rmse_pb": old, "challenger_rmse_pb": new,
                              "relative_change_percent": 100 * (new / old - 1),
                              "ci_low_percent": float(low), "ci_high_percent": float(high)})
        errors["horizon"] = np.where(errors.future_rank.eq(1), "first_future", "later_future")
        for horizon, subset in errors.groupby("horizon"):
            people = subset.groupby("participant")[list(METHODS)].mean()
            for method in METHODS:
                horizons.append({"dataset": dataset, "horizon": horizon, "method": method,
                                 "participants": len(people), "records": len(subset),
                                 "rmse_pb": float(np.sqrt(people[method].mean()))})
    return pd.DataFrame(metrics), pd.DataFrame(contrasts), pd.DataFrame(horizons)


def main() -> None:
    if not PROTOCOL.exists():
        raise RuntimeError("Frozen exploratory protocol missing")
    predictions, audits = assemble()
    metrics, contrasts, horizons = summarize(predictions)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(OUTPUT / "recency_training_only_predictions.parquet", index=False)
    audits.to_csv(OUTPUT / "recency_training_only_fold_audit.csv", index=False)
    metrics.to_csv(OUTPUT / "recency_training_only_metrics.csv", index=False)
    contrasts.to_csv(OUTPUT / "recency_training_only_contrasts.csv", index=False)
    horizons.to_csv(OUTPUT / "recency_training_only_horizons.csv", index=False)
    inputs = (FOLDS, FAIR, TABLES / "psyche_analysis.parquet",
              OLD / "ces_development_multiscale_features.parquet",
              OLD / "ces_exploration_participant_split.parquet",
              OUTPUT / "mobilewell_sgpeba_analysis_table.parquet",
              OUTPUT / "crosscheck_confirmation_table.parquet")
    manifest = {"status": "post-hoc exploratory; no untouched confirmation",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "seed": SEED, "bootstrap_resamples": BOOTSTRAPS,
                "protocol_sha256": sha256(PROTOCOL), "source_sha256": sha256(Path(__file__)),
                "inputs_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
                "software": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
                "participants": predictions.groupby("dataset").participant.nunique().to_dict(),
                "records": predictions.groupby("dataset").size().to_dict(),
                "target_reconstruction_check": "all saved actual, mean and last-report values match at 1e-10"}
    (OUTPUT / "recency_training_only_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nRECENCY CONTRASTS", flush=True)
    print(contrasts.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
