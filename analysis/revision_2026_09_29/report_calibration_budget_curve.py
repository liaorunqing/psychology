"""Fixed early-report calibration budgets in CES and Marian EMA.

The analysis protocol was frozen in direction_reset/61_* before execution.
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
import sklearn
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "outputs" / "revision_2026_09_29"
PROTOCOL = ROOT / "direction_reset" / "61_report_calibration_budget_curve_protocol_2026-09-30.md"
FOLDS = OUT / "fair_history_adapter_fold_map.parquet"
DATA_ROOT = Path(os.environ.get("DPT_DATA_ROOT", ROOT / "data")).expanduser()
MARIAN = DATA_ROOT / "marian_openesm" / "0052_marian_ts.tsv"
SEED = 20260918
BOOTSTRAPS = 2000
BUDGETS = (2, 3, 4, 6, 8)
WEIGHTS = np.linspace(0.0, 1.0, 41)
METHODS = tuple(f"B{budget}" for budget in BUDGETS) + ("R_online",)
CONTRASTS = (("B2", "B3"), ("B3", "B4"), ("B4", "B6"),
             ("B6", "B8"), ("B8", "R_online"))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PROFILE = load_module(Path(__file__).with_name("ces_psychological_measurement_profile.py"),
                      "budget_ces_profile")
MARIAN_MODULE = load_module(Path(__file__).with_name("marian_history_observation_experiment.py"),
                            "budget_marian_history")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def participant_weights(ids: pd.Series) -> np.ndarray:
    counts = ids.value_counts()
    weights = ids.map(1.0 / counts).to_numpy(dtype=float)
    return weights / weights.mean()


def build_rows(frame: pd.DataFrame, id_col: str, order_col: str,
               value_col: str, dataset: str, outcome: str,
               fold_map: dict[str, int]) -> pd.DataFrame:
    """Build common rank>=9 targets and fixed-budget/online history features."""
    rows: list[dict] = []
    work = frame[[id_col, order_col, value_col]].dropna().copy()
    work[id_col] = work[id_col].astype(str)
    work = work.sort_values([id_col, order_col], kind="stable")
    for participant, person in work.groupby(id_col, sort=False):
        values = person[value_col].to_numpy(dtype=float)
        orders = person[order_col].to_numpy()
        is_datetime = np.issubdtype(person[order_col].dtype, np.datetime64)
        if len(values) < 9:
            continue
        fixed = {}
        for budget in BUDGETS:
            initial = values[:budget]
            fixed[budget] = (float(initial.mean()), float(initial[-1]))
        for index in range(8, len(values)):
            prior = values[:index]
            record = {
                "dataset": dataset,
                "outcome": outcome,
                "fold": int(fold_map[participant]),
                "participant": participant,
                "occasion": str(orders[index]),
                "report_rank": int(index + 1),
                "actual": float(values[index]),
                "online_mean": float(prior.mean()),
                "online_last": float(prior[-1]),
            }
            for budget, (mean, last) in fixed.items():
                record[f"mean_{budget}"] = mean
                record[f"last_{budget}"] = last
                difference = orders[index] - orders[budget - 1]
                record[f"distance_{budget}"] = (float(difference / np.timedelta64(1, "D"))
                                                  if is_datetime else float(difference))
            rows.append(record)
    result = pd.DataFrame(rows)
    key = ["dataset", "outcome", "participant", "occasion"]
    if result.duplicated(key).any():
        raise AssertionError("Duplicate target key")
    if result.empty or result.report_rank.lt(9).any():
        raise AssertionError("Invalid common target set")
    return result


def tune_weight(frame: pd.DataFrame, mean_col: str, last_col: str) -> tuple[float, float]:
    base = frame[mean_col].to_numpy(dtype=float)
    delta = frame[last_col].to_numpy(dtype=float) - base
    actual = frame.actual.to_numpy(dtype=float)
    weights = participant_weights(frame.participant)
    losses = []
    for weight in WEIGHTS:
        loss = float(np.average(np.abs(actual - (base + weight * delta)), weights=weights))
        losses.append(loss)
    best = int(np.argmin(losses))
    return float(WEIGHTS[best]), float(losses[best])


def ces_rows() -> pd.DataFrame:
    data = PROFILE.BENCHMARK.build_wide_table().copy()
    data["uid"] = data.uid.astype(str)
    mapping = (pd.read_parquet(FOLDS).query("dataset == 'CES'")
               .set_index("participant").fold.astype(int).to_dict())
    if set(data.uid) != set(mapping):
        raise AssertionError("CES fold map mismatch")
    pieces = []
    for fold in range(1, 6):
        train_people = {pid for pid, assigned in mapping.items() if assigned != fold}
        scaler = StandardScaler().fit(data.loc[data.uid.isin(train_people), ["phq2"]])
        transformed = data.copy()
        transformed["value"] = scaler.transform(data[["phq2"]]).ravel()
        fold_rows = build_rows(transformed, "uid", "assessment_date", "value",
                               "CES", "phq2", mapping)
        pieces.append(fold_rows.loc[fold_rows.fold.eq(fold)])
    result = pd.concat(pieces, ignore_index=True)
    if result.participant.nunique() != 105 or len(result) != 4541:
        raise AssertionError("CES common-budget sample changed")
    return result


def marian_rows() -> pd.DataFrame:
    data = pd.read_csv(MARIAN, sep="\t")
    complete = data.loc[data.status.eq(1)].copy()
    complete["id"] = complete.id.astype(str)
    mapping = MARIAN_MODULE.fold_map(data.id.astype(str))
    pieces = []
    for outcome in ("anhedonia", "depressed"):
        piece = build_rows(complete, "id", "counter", outcome,
                           "Marian", outcome, mapping)
        if piece.participant.nunique() != 145 or len(piece) != 7071:
            raise AssertionError(f"Marian common-budget sample changed: {outcome}")
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def fit_predict(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    output, audits = [], []
    for (dataset, outcome), group in rows.groupby(["dataset", "outcome"], sort=False):
        for fold in range(1, 6):
            train = group.loc[group.fold.ne(fold)].copy()
            test = group.loc[group.fold.eq(fold)].copy()
            if set(train.participant) & set(test.participant):
                raise AssertionError("Participant crosses outer fold")
            audit = {
                "dataset": dataset, "outcome": outcome, "fold": fold,
                "train_participants": train.participant.nunique(),
                "train_targets": len(train), "test_participants": test.participant.nunique(),
                "test_targets": len(test), "participant_overlap": False,
            }
            for budget in BUDGETS:
                weight, inner_mae = tune_weight(train, f"mean_{budget}", f"last_{budget}")
                test[f"B{budget}"] = (test[f"mean_{budget}"] + weight *
                                       (test[f"last_{budget}"] - test[f"mean_{budget}"]))
                audit[f"weight_B{budget}"] = weight
                audit[f"train_mae_B{budget}"] = inner_mae
            weight, inner_mae = tune_weight(train, "online_mean", "online_last")
            test["R_online"] = (test.online_mean + weight *
                                (test.online_last - test.online_mean))
            audit["weight_R_online"] = weight
            audit["train_mae_R_online"] = inner_mae
            output.append(test)
            audits.append(audit)
    predictions = pd.concat(output, ignore_index=True)
    key = ["dataset", "outcome", "fold", "participant", "occasion"]
    if predictions.duplicated(key).any():
        raise AssertionError("Duplicate prediction key")
    if predictions[list(METHODS)].isna().any().any():
        raise AssertionError("Missing prediction")
    return predictions, pd.DataFrame(audits)


def score(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    metrics, contrasts = [], []
    for (dataset, outcome), group in predictions.groupby(["dataset", "outcome"], sort=False):
        ae = pd.DataFrame({"participant": group.participant})
        se = pd.DataFrame({"participant": group.participant})
        for method in METHODS:
            residual = group.actual - group[method]
            ae[method] = np.abs(residual)
            se[method] = np.square(residual)
        person_ae = ae.groupby("participant")[list(METHODS)].mean()
        person_se = se.groupby("participant")[list(METHODS)].mean()
        for method in METHODS:
            metrics.append({
                "dataset": dataset, "outcome": outcome, "method": method,
                "participants": len(person_ae), "targets": len(group),
                "mae_pb": float(person_ae[method].mean()),
                "rmse_pb": float(np.sqrt(person_se[method].mean())),
            })
        for reference, challenger in CONTRASTS:
            person = pd.DataFrame({
                "ref_ae": person_ae[reference], "new_ae": person_ae[challenger],
                "ref_se": person_se[reference], "new_se": person_se[challenger],
            })
            mae_ref, mae_new = person[["ref_ae", "new_ae"]].mean()
            rmse_ref, rmse_new = np.sqrt(person[["ref_se", "new_se"]].mean())
            mae_draws = np.empty(BOOTSTRAPS)
            rmse_draws = np.empty(BOOTSTRAPS)
            values = person.to_numpy()
            for iteration in range(BOOTSTRAPS):
                sample = values[rng.integers(0, len(values), len(values))]
                means = sample.mean(axis=0)
                mae_draws[iteration] = 100 * (means[1] / means[0] - 1)
                rmse_draws[iteration] = 100 * (np.sqrt(means[3]) / np.sqrt(means[2]) - 1)
            contrasts.append({
                "dataset": dataset, "outcome": outcome,
                "contrast": f"{challenger}_vs_{reference}",
                "reference": reference, "challenger": challenger,
                "participants": len(person), "targets": len(group),
                "relative_mae_percent": float(100 * (mae_new / mae_ref - 1)),
                "mae_ci_low": float(np.quantile(mae_draws, .025)),
                "mae_ci_high": float(np.quantile(mae_draws, .975)),
                "relative_rmse_percent": float(100 * (rmse_new / rmse_ref - 1)),
                "rmse_ci_low": float(np.quantile(rmse_draws, .025)),
                "rmse_ci_high": float(np.quantile(rmse_draws, .975)),
            })
    return pd.DataFrame(metrics), pd.DataFrame(contrasts)


def check_fixed_budget_invariance(rows: pd.DataFrame) -> None:
    """Saved fixed-budget features must be constant within participant."""
    feature_columns = [column for budget in BUDGETS
                       for column in (f"mean_{budget}", f"last_{budget}")]
    counts = rows.groupby(["dataset", "outcome", "participant"])[feature_columns].nunique()
    if not counts.eq(1).all().all():
        raise AssertionError("A fixed budget changes after its calibration reports")


def main() -> None:
    if not PROTOCOL.exists():
        raise RuntimeError("Frozen budget protocol is missing")
    source = pd.concat([ces_rows(), marian_rows()], ignore_index=True)
    check_fixed_budget_invariance(source)
    predictions, audit = fit_predict(source)
    metrics, contrasts = score(predictions)
    distances = []
    for (dataset, outcome), group in predictions.groupby(["dataset", "outcome"], sort=False):
        unit = "days" if dataset == "CES" else "scheduled_opportunities"
        for budget in BUDGETS:
            values = group[f"distance_{budget}"]
            distances.append({"dataset": dataset, "outcome": outcome,
                              "budget": budget, "unit": unit,
                              "participants": group.participant.nunique(),
                              "targets": len(group), "median": float(values.median()),
                              "p25": float(values.quantile(.25)),
                              "p75": float(values.quantile(.75)),
                              "maximum": float(values.max())})
    distances = pd.DataFrame(distances)
    OUT.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(OUT / "report_calibration_budget_predictions.parquet", index=False)
    audit.to_csv(OUT / "report_calibration_budget_fold_audit.csv", index=False)
    metrics.to_csv(OUT / "report_calibration_budget_metrics.csv", index=False)
    contrasts.to_csv(OUT / "report_calibration_budget_contrasts.csv", index=False)
    distances.to_csv(OUT / "report_calibration_budget_distances.csv", index=False)
    manifest = {
        "status": "post-inspection exploratory fixed early-report budget curve",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED, "bootstrap_resamples": BOOTSTRAPS,
        "budgets": list(BUDGETS), "weight_grid": WEIGHTS.tolist(),
        "data_sha256": {"marian": sha256(MARIAN),
                        "ces_fold_map": sha256(FOLDS)},
        "protocol_sha256": sha256(PROTOCOL),
        "script_sha256": sha256(Path(__file__)),
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "participants": (predictions.groupby(["dataset", "outcome"])
                         .participant.nunique().astype(int).to_dict()),
        "targets": predictions.groupby(["dataset", "outcome"]).size().astype(int).to_dict(),
        "checks": {"common_targets_across_budgets": True,
                   "participant_disjoint_outer_folds": True,
                   "fixed_budget_features_constant_after_calibration": True,
                   "training_only_scaling_and_weight_tuning": True},
    }
    # JSON cannot directly encode tuple keys.
    manifest["participants"] = {"|".join(key): value
                                for key, value in manifest["participants"].items()}
    manifest["targets"] = {"|".join(key): value
                           for key, value in manifest["targets"].items()}
    (OUT / "report_calibration_budget_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nBUDGET METRICS")
    print(metrics.to_string(index=False))
    print("\nBUDGET CONTRASTS")
    print(contrasts.to_string(index=False))


if __name__ == "__main__":
    main()
