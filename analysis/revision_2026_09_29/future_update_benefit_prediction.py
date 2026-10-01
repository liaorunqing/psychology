"""Nested-CV prediction of late dynamic-baseline benefit from the first eight reports."""

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
import sklearn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "outputs" / "revision_2026_09_29"
PROTOCOL = ROOT / "direction_reset" / "82_future_update_benefit_prediction_protocol_2026-09-30.md"
BLOCK_METRICS = OUT / "individual_staleness_temporal_block_metrics.parquet"
CROSS_SCRIPT = Path(__file__).with_name("cross_dataset_equal_information_replication.py")
DATA_ROOT = Path(os.environ.get("DPT_DATA_ROOT", ROOT / "data")).expanduser()
DEJON_DATA = DATA_ROOT / "dejonckheere_openesm" / "0012_dejonckheere_ts.tsv"
SEED = 20260918
BOOTSTRAPS = 2000
PERMUTATIONS = 500
ALPHAS = np.asarray([.01, .1, 1., 10., 100.])
FEATURES = ("baseline_mean", "baseline_sd", "baseline_masd", "baseline_trend")
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


CROSS = load_module(CROSS_SCRIPT, "cross_for_benefit_prediction")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dejon_fold_map(ids: pd.Series) -> dict[str, int]:
    unique = np.asarray(sorted(ids.astype(str).unique()))
    rng = np.random.default_rng(SEED)
    shuffled = unique[rng.permutation(len(unique))]
    return {participant: int(index % 5 + 1)
            for index, participant in enumerate(shuffled)}


def load_specs() -> list[tuple[str, str, pd.DataFrame, str, str, dict[str, int]]]:
    dejon = pd.read_csv(DEJON_DATA, sep="\t")
    dejon["id"] = dejon.id.astype(str)
    mapping = dejon_fold_map(dejon.id)
    specs = [("Dejonckheere", outcome, dejon, "id", "counter", mapping)
             for outcome in ("sad", "stressed", "happy", "relaxed", "angry")]
    specs.extend(CROSS.load_inputs())
    return specs


def baseline_features(frame: pd.DataFrame, outcome: str, id_col: str,
                      order_col: str) -> pd.DataFrame:
    work = frame[[id_col, order_col, outcome]].dropna().copy()
    work[id_col] = work[id_col].astype(str)
    work = work.sort_values([id_col, order_col], kind="stable")
    rows = []
    index = np.arange(8, dtype=float)
    centered = index - index.mean()
    denominator = float(np.square(centered).sum())
    for participant, person in work.groupby(id_col, sort=False):
        values = person[outcome].to_numpy(dtype=float)
        if len(values) < 8:
            continue
        initial = values[:8]
        rows.append({"participant": participant,
                     "baseline_mean": float(initial.mean()),
                     "baseline_sd": float(initial.std(ddof=1)),
                     "baseline_masd": float(np.abs(np.diff(initial)).mean()),
                     "baseline_trend": float(np.dot(centered, initial) / denominator)})
    return pd.DataFrame(rows)


def build_model_tables() -> pd.DataFrame:
    targets = pd.read_parquet(BLOCK_METRICS)[
        ["dataset", "outcome", "participant", "mean_benefit_late"]].copy()
    targets["participant"] = targets.participant.astype(str)
    pieces = []
    for dataset, outcome, frame, id_col, order_col, fold_map in load_specs():
        features = baseline_features(frame, outcome, id_col, order_col)
        target = targets.loc[targets.dataset.eq(dataset) & targets.outcome.eq(outcome)]
        table = target.merge(features, on="participant", how="inner", validate="one_to_one")
        table["fold"] = table.participant.map(fold_map)
        if table.fold.isna().any() or table[list(FEATURES)].isna().any().any():
            raise AssertionError(f"Missing feature or fold: {(dataset, outcome)}")
        table["fold"] = table.fold.astype(int)
        pieces.append(table)
    result = pd.concat(pieces, ignore_index=True)
    if result.duplicated(["dataset", "outcome", "participant"]).any():
        raise AssertionError("Participant appears twice within task")
    return result


def inner_fold_map(participants: pd.Series, outer_fold: int,
                   task_index: int) -> dict[str, int]:
    ids = np.asarray(sorted(participants.astype(str).unique()))
    rng = np.random.default_rng(SEED + outer_fold + task_index * 100)
    shuffled = ids[rng.permutation(len(ids))]
    return {participant: int(index % 5 + 1)
            for index, participant in enumerate(shuffled)}


def choose_alpha(train: pd.DataFrame, task_index: int,
                 outer_fold: int) -> tuple[float, list[float]]:
    mapping = inner_fold_map(train.participant, outer_fold, task_index)
    squared_errors = {alpha: [] for alpha in ALPHAS}
    for inner_fold in range(1, 6):
        validation_mask = train.participant.map(mapping).eq(inner_fold)
        inner_train, validation = train.loc[~validation_mask], train.loc[validation_mask]
        scaler = StandardScaler().fit(inner_train[list(FEATURES)])
        x_train = scaler.transform(inner_train[list(FEATURES)])
        x_validation = scaler.transform(validation[list(FEATURES)])
        y_train = inner_train.mean_benefit_late.to_numpy(dtype=float)
        y_validation = validation.mean_benefit_late.to_numpy(dtype=float)
        for alpha in ALPHAS:
            model = Ridge(alpha=float(alpha)).fit(x_train, y_train)
            squared_errors[alpha].extend(np.square(y_validation - model.predict(x_validation)))
    losses = np.asarray([np.sqrt(np.mean(squared_errors[alpha])) for alpha in ALPHAS])
    minimum = losses.min()
    candidates = ALPHAS[np.isclose(losses, minimum, rtol=1e-12, atol=1e-12)]
    return float(candidates.max()), losses.tolist()


def nested_predict(table: pd.DataFrame, task_index: int,
                   target: np.ndarray | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = table.copy()
    if target is not None:
        work["mean_benefit_late"] = target
    outputs, audits = [], []
    for fold in range(1, 6):
        train, test = work.loc[work.fold.ne(fold)].copy(), work.loc[work.fold.eq(fold)].copy()
        if set(train.participant) & set(test.participant):
            raise AssertionError("Participant crosses outer fold")
        alpha, inner_losses = choose_alpha(train, task_index, fold)
        scaler = StandardScaler().fit(train[list(FEATURES)])
        x_train = scaler.transform(train[list(FEATURES)])
        x_test = scaler.transform(test[list(FEATURES)])
        model = Ridge(alpha=alpha).fit(x_train, train.mean_benefit_late)
        test["M0"] = float(train.mean_benefit_late.mean())
        test["M1"] = model.predict(x_test)
        outputs.append(test)
        audits.append({"dataset": table.dataset.iloc[0], "outcome": table.outcome.iloc[0],
                       "fold": fold, "train_participants": len(train),
                       "test_participants": len(test), "alpha": alpha,
                       **{f"coef_{feature}": float(value)
                          for feature, value in zip(FEATURES, model.coef_)},
                       **{f"inner_rmse_alpha_{alpha_value:g}": float(loss)
                          for alpha_value, loss in zip(ALPHAS, inner_losses)}})
    result = pd.concat(outputs, ignore_index=True)
    if result.participant.nunique() != len(table):
        raise AssertionError("Missing OOF prediction")
    return result, pd.DataFrame(audits)


def metrics(predictions: pd.DataFrame) -> dict:
    actual = predictions.mean_benefit_late.to_numpy(dtype=float)
    base = predictions.M0.to_numpy(dtype=float)
    model = predictions.M1.to_numpy(dtype=float)
    rmse0 = float(np.sqrt(np.mean(np.square(actual - base))))
    rmse1 = float(np.sqrt(np.mean(np.square(actual - model))))
    mae0 = float(np.mean(np.abs(actual - base)))
    mae1 = float(np.mean(np.abs(actual - model)))
    sst = float(np.square(actual - actual.mean()).sum())
    return {"participants": len(actual), "rmse_M0": rmse0, "rmse_M1": rmse1,
            "relative_rmse_percent": float(100 * (rmse1 / rmse0 - 1)),
            "improvement_percent": float(100 * (rmse0 - rmse1) / rmse0),
            "mae_M0": mae0, "mae_M1": mae1,
            "relative_mae_percent": float(100 * (mae1 / mae0 - 1)),
            "r2_oof": float(1 - np.square(actual - model).sum() / sst),
            "pearson_r_oof": float(np.corrcoef(actual, model)[0, 1])}


def bootstrap_rmse(predictions: pd.DataFrame,
                   rng: np.random.Generator) -> tuple[float, float]:
    actual = predictions.mean_benefit_late.to_numpy(dtype=float)
    base = predictions.M0.to_numpy(dtype=float)
    model = predictions.M1.to_numpy(dtype=float)
    draws = np.empty(BOOTSTRAPS)
    for iteration in range(BOOTSTRAPS):
        indices = rng.integers(0, len(actual), len(actual))
        rmse0 = np.sqrt(np.mean(np.square(actual[indices] - base[indices])))
        rmse1 = np.sqrt(np.mean(np.square(actual[indices] - model[indices])))
        draws[iteration] = 100 * (rmse1 / rmse0 - 1)
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def permutation_null(table: pd.DataFrame, task_index: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + task_index * 10000)
    original = table.mean_benefit_late.to_numpy(dtype=float)
    rows = []
    for iteration in range(PERMUTATIONS):
        permuted = original[rng.permutation(len(original))]
        predictions, _ = nested_predict(table, task_index, target=permuted)
        value = metrics(predictions)["improvement_percent"]
        rows.append({"dataset": table.dataset.iloc[0], "outcome": table.outcome.iloc[0],
                     "iteration": iteration + 1, "improvement_percent": value})
        if (iteration + 1) % 100 == 0:
            print(f"{table.dataset.iloc[0]}/{table.outcome.iloc[0]}: "
                  f"{iteration + 1}/{PERMUTATIONS}", flush=True)
    return pd.DataFrame(rows)


def classify(summary: pd.DataFrame) -> tuple[str, dict[str, bool]]:
    primary = summary.loc[summary.role.eq("primary")].copy()
    passed = (primary.improvement_percent.gt(10) &
              primary.rmse_ci_high.lt(0) &
              primary.pearson_r_oof.gt(0) &
              primary.monte_carlo_p_one_sided.le(.0125))
    mapping = {f"{row.dataset}|{row.outcome}": bool(value)
               for row, value in zip(primary.itertuples(index=False), passed)}
    if passed.all():
        result = "broad_baseline_personalization_support"
    elif passed.sum() >= 2:
        result = "construct_dependent_limited_support"
    else:
        result = "stop_individual_benefit_prediction"
    return result, mapping


def main() -> None:
    if not PROTOCOL.exists():
        raise RuntimeError("Frozen benefit-prediction protocol missing")
    tables = build_model_tables()
    rng = np.random.default_rng(SEED)
    prediction_parts, audit_parts, summary_rows, null_parts = [], [], [], []
    grouped = list(tables.groupby(["dataset", "outcome"], sort=False))
    for task_index, ((dataset, outcome), table) in enumerate(grouped):
        predictions, audit = nested_predict(table, task_index)
        result = metrics(predictions)
        low, high = bootstrap_rmse(predictions, rng)
        summary_rows.append({"dataset": dataset, "outcome": outcome,
                             "role": "primary" if (dataset, outcome) in PRIMARY else "secondary",
                             **result, "rmse_ci_low": low, "rmse_ci_high": high})
        prediction_parts.append(predictions); audit_parts.append(audit)
        if (dataset, outcome) in PRIMARY:
            null_parts.append(permutation_null(table, task_index))
    summary = pd.DataFrame(summary_rows)
    null = pd.concat(null_parts, ignore_index=True)
    p_values = []
    for row in summary.itertuples(index=False):
        if row.role == "primary":
            values = null.loc[(null.dataset.eq(row.dataset)) & null.outcome.eq(row.outcome),
                              "improvement_percent"].to_numpy(dtype=float)
            p_values.append(float((1 + np.sum(values >= row.improvement_percent)) /
                                  (PERMUTATIONS + 1)))
        else:
            p_values.append(np.nan)
    summary["monte_carlo_p_one_sided"] = p_values
    classification, primary_pass = classify(summary)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    audits = pd.concat(audit_parts, ignore_index=True)
    predictions.to_parquet(OUT / "future_update_benefit_predictions.parquet", index=False)
    audits.to_csv(OUT / "future_update_benefit_fold_audit.csv", index=False)
    summary.to_csv(OUT / "future_update_benefit_prediction_summary.csv", index=False)
    null.to_parquet(OUT / "future_update_benefit_permutation_null.parquet", index=False)
    manifest = {"status": "post-confirmation frozen algorithm feasibility test",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "classification": classification, "primary_pass": primary_pass,
                "seed": SEED, "bootstrap_resamples": BOOTSTRAPS,
                "permutations_primary_outcome": PERMUTATIONS,
                "features": list(FEATURES), "alphas": ALPHAS.tolist(),
                "protocol_sha256": sha256(PROTOCOL), "script_sha256": sha256(Path(__file__)),
                "input_sha256": {"block_metrics": sha256(BLOCK_METRICS),
                                 "dejonckheere": sha256(DEJON_DATA)},
                "software": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
                "checks": {"one_row_per_participant_task": True,
                           "features_first_eight_only": True,
                           "target_late_block_only": True,
                           "participant_disjoint_outer_folds": True,
                           "training_only_preprocessing": True,
                           "nested_training_only_alpha_selection": True,
                           "full_nested_refit_each_permutation": True}}
    (OUT / "future_update_benefit_prediction_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nCLASSIFICATION:", classification)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
