"""External exploratory test of report history and prompt observation in Marian EMA.

The protocol was frozen in direction_reset/59_* before running this script.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("DPT_DATA_ROOT", ROOT / "data")).expanduser()
DATA = DATA_ROOT / "marian_openesm" / "0052_marian_ts.tsv"
PROTOCOL = ROOT / "direction_reset" / "59_marian_history_observation_experiment_protocol_2026-09-29.md"
OUTPUT = ROOT / "analysis" / "outputs" / "revision_2026_09_29"
SEED = 20260918
BOOTSTRAPS = 2000
OUTCOMES = ("anhedonia", "depressed")
METHODS = ("F", "M", "R", "Q")
Q_FEATURES = (
    "M", "L", "n_prior_complete", "prior_completion_rate",
    "missing_streak", "day", "beep", "counter",
)
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
LOGISTIC_CS = (0.001, 0.01, 0.1, 1.0, 10.0)


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


def fold_map(ids: pd.Series) -> dict[str, int]:
    unique = np.asarray(sorted(ids.astype(str).unique()))
    rng = np.random.default_rng(SEED)
    shuffled = unique[rng.permutation(len(unique))]
    return {participant: int(index % 5 + 1) for index, participant in enumerate(shuffled)}


def opportunity_rows(data: pd.DataFrame, outcome: str) -> pd.DataFrame:
    rows: list[dict] = []
    for participant, person in data.groupby("id", sort=False):
        person = person.sort_values("counter", kind="stable")
        history: list[float] = []
        prior_opportunities = 0
        missing_streak = 0
        first_two: list[float] | None = None
        for row in person.itertuples(index=False):
            completed = int(row.status == 1)
            if len(history) >= 2:
                if first_two is None:
                    first_two = history[:2]
                rows.append({
                    "participant": str(participant),
                    "day": int(row.day),
                    "beep": int(row.beep),
                    "counter": int(row.counter),
                    "completed": completed,
                    "actual": float(getattr(row, outcome)) if completed else np.nan,
                    "F_mean": float(np.mean(first_two)),
                    "F_last": float(first_two[-1]),
                    "M": float(np.mean(history)),
                    "L": float(history[-1]),
                    "n_prior_complete": int(len(history)),
                    "prior_completion_rate": float(len(history) / prior_opportunities),
                    "missing_streak": int(missing_streak),
                })
            prior_opportunities += 1
            if completed:
                history.append(float(getattr(row, outcome)))
                missing_streak = 0
            else:
                missing_streak += 1
    result = pd.DataFrame(rows)
    if result.duplicated(["participant", "counter"]).any():
        raise AssertionError("Duplicate opportunity key")
    if result[list(Q_FEATURES)].isna().any().any():
        raise AssertionError("Missing history feature")
    return result


def tune_recency_weight(frame: pd.DataFrame, mean_col: str, last_col: str) -> float:
    delta = frame[last_col] - frame[mean_col]
    error = frame[mean_col] - frame.actual
    work = pd.DataFrame({"participant": frame.participant,
                         "numerator": error * delta,
                         "denominator": delta * delta})
    by_person = work.groupby("participant")[["numerator", "denominator"]].mean()
    denominator = float(by_person.denominator.mean())
    if denominator <= 0:
        return 0.0
    return float(np.clip(-by_person.numerator.mean() / denominator, 0, 1))


def fit_ridge(train: pd.DataFrame, alpha: float) -> tuple[StandardScaler, Ridge]:
    scaler = StandardScaler().fit(train[list(Q_FEATURES)])
    model = Ridge(alpha=alpha).fit(
        scaler.transform(train[list(Q_FEATURES)]), train.actual,
        sample_weight=participant_weights(train.participant),
    )
    return scaler, model


def choose_ridge_alpha(train: pd.DataFrame) -> tuple[float, dict[str, float]]:
    groups = train.participant.to_numpy()
    splitter = GroupKFold(n_splits=5)
    errors = {alpha: [] for alpha in RIDGE_ALPHAS}
    for fit_idx, val_idx in splitter.split(train, groups=groups):
        fit = train.iloc[fit_idx]
        val = train.iloc[val_idx]
        for alpha in RIDGE_ALPHAS:
            scaler, model = fit_ridge(fit, alpha)
            predicted = model.predict(scaler.transform(val[list(Q_FEATURES)]))
            loss = pd.DataFrame({"participant": val.participant.to_numpy(),
                                 "ae": np.abs(val.actual.to_numpy() - predicted)})
            errors[alpha].append(float(loss.groupby("participant").ae.mean().mean()))
    mean_error = {str(alpha): float(np.mean(values)) for alpha, values in errors.items()}
    selected = min(RIDGE_ALPHAS, key=lambda a: (mean_error[str(a)], -a))
    return float(selected), mean_error


def fit_logistic(train: pd.DataFrame, c_value: float) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(train[list(Q_FEATURES)])
    model = LogisticRegression(C=c_value, max_iter=3000, solver="lbfgs").fit(
        scaler.transform(train[list(Q_FEATURES)]), train.completed,
        sample_weight=participant_weights(train.participant),
    )
    return scaler, model


def choose_logistic_c(train: pd.DataFrame) -> tuple[float, dict[str, float]]:
    groups = train.participant.to_numpy()
    splitter = GroupKFold(n_splits=5)
    losses = {c: [] for c in LOGISTIC_CS}
    for fit_idx, val_idx in splitter.split(train, groups=groups):
        fit = train.iloc[fit_idx]
        val = train.iloc[val_idx]
        for c_value in LOGISTIC_CS:
            scaler, model = fit_logistic(fit, c_value)
            probability = model.predict_proba(scaler.transform(val[list(Q_FEATURES)]))[:, 1]
            scored = pd.DataFrame({"participant": val.participant.to_numpy(),
                                   "loss": np.square(val.completed.to_numpy() - probability)})
            losses[c_value].append(float(scored.groupby("participant").loss.mean().mean()))
    mean_loss = {str(c): float(np.mean(values)) for c, values in losses.items()}
    selected = min(LOGISTIC_CS, key=lambda c: (mean_loss[str(c)], c))
    return float(selected), mean_loss


def check_current_target_invariance(data: pd.DataFrame, outcome: str) -> None:
    original = opportunity_rows(data, outcome)
    changed = data.copy()
    complete_positions = changed.index[changed.status.eq(1)]
    changed.loc[complete_positions, outcome] = changed.loc[complete_positions, outcome] + 100.0
    # Perturbing every observed value changes history. Instead compare the features at
    # the first eligible target after perturbing only that target per person.
    changed = data.copy()
    for _, group in changed[changed.status.eq(1)].groupby("id", sort=False):
        if len(group) >= 3:
            changed.loc[group.index[2], outcome] += 100.0
    altered = opportunity_rows(changed, outcome)
    key = ["participant", "counter"]
    first_target_keys = (original[original.completed.eq(1)]
                         .groupby("participant", as_index=False).first()[key])
    a = original.merge(first_target_keys, on=key, how="inner").sort_values(key)
    b = altered.merge(first_target_keys, on=key, how="inner").sort_values(key)
    pd.testing.assert_frame_equal(a[key + list(Q_FEATURES)].reset_index(drop=True),
                                  b[key + list(Q_FEATURES)].reset_index(drop=True),
                                  check_exact=True)


def assemble(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapping = fold_map(data.id)
    outcome_predictions, opportunity_predictions, audits = [], [], []
    for outcome in OUTCOMES:
        check_current_target_invariance(data, outcome)
        rows = opportunity_rows(data, outcome)
        rows["fold"] = rows.participant.map(mapping).astype(int)
        for fold in range(1, 6):
            train_all = rows[rows.fold.ne(fold)].copy()
            test_all = rows[rows.fold.eq(fold)].copy()
            if set(train_all.participant) & set(test_all.participant):
                raise AssertionError("Participant crosses outer fold")
            train = train_all[train_all.completed.eq(1)].copy()
            test = test_all[test_all.completed.eq(1)].copy()
            fixed_weight = tune_recency_weight(train, "F_mean", "F_last")
            recent_weight = tune_recency_weight(train, "M", "L")
            ridge_alpha, ridge_trace = choose_ridge_alpha(train)
            ridge_scaler, ridge = fit_ridge(train, ridge_alpha)
            logistic_c, logistic_trace = choose_logistic_c(train_all)
            logistic_scaler, logistic = fit_logistic(train_all, logistic_c)

            test["F"] = test.F_mean + fixed_weight * (test.F_last - test.F_mean)
            test["R"] = test.M + recent_weight * (test.L - test.M)
            test["Q"] = ridge.predict(ridge_scaler.transform(test[list(Q_FEATURES)]))
            test_all["response_probability"] = logistic.predict_proba(
                logistic_scaler.transform(test_all[list(Q_FEATURES)]))[:, 1]
            test = test.merge(
                test_all[["participant", "counter", "response_probability"]],
                on=["participant", "counter"], how="left", validate="one_to_one",
            )
            test["outcome"] = outcome
            test_all["outcome"] = outcome
            outcome_predictions.append(test[[
                "outcome", "fold", "participant", "day", "beep", "counter", "actual",
                "F", "M", "R", "Q", "response_probability", *[x for x in Q_FEATURES if x not in {"M", "L", "day", "beep", "counter"}],
            ]])
            opportunity_predictions.append(test_all[[
                "outcome", "fold", "participant", "day", "beep", "counter",
                "completed", "response_probability",
            ]])
            audits.append({
                "outcome": outcome, "fold": fold,
                "train_participants": int(train_all.participant.nunique()),
                "train_opportunities": len(train_all), "train_targets": len(train),
                "test_participants": int(test_all.participant.nunique()),
                "test_opportunities": len(test_all), "test_targets": len(test),
                "fixed_weight": fixed_weight, "recent_weight": recent_weight,
                "ridge_alpha": ridge_alpha, "logistic_c": logistic_c,
                "ridge_inner_mae": json.dumps(ridge_trace, sort_keys=True),
                "logistic_inner_brier": json.dumps(logistic_trace, sort_keys=True),
                "participant_overlap": False, "current_target_invariance": True,
            })
            print(f"{outcome} fold {fold}/5", flush=True)

    predictions = pd.concat(outcome_predictions, ignore_index=True)
    opportunities = pd.concat(opportunity_predictions, ignore_index=True)
    key = ["fold", "participant", "counter"]
    reference = predictions[predictions.outcome.eq(OUTCOMES[0])][key].sort_values(key).reset_index(drop=True)
    candidate = predictions[predictions.outcome.eq(OUTCOMES[1])][key].sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(reference, candidate, check_exact=True)
    if predictions.duplicated(["outcome", *key]).any():
        raise AssertionError("Duplicate scored target")
    if opportunities.duplicated(["outcome", *key]).any():
        raise AssertionError("Duplicate opportunity")
    return predictions, opportunities, pd.DataFrame(audits)


def participant_losses(group: pd.DataFrame, weighted: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    mae_rows, mse_rows = [], []
    for participant, person in group.groupby("participant", sort=False):
        if weighted:
            weights = 1.0 / person.response_probability.clip(0.10, 0.99).to_numpy()
            weights = weights / weights.sum()
        else:
            weights = np.repeat(1.0 / len(person), len(person))
        mae = {method: float(np.sum(weights * np.abs(person.actual - person[method]))) for method in METHODS}
        mse = {method: float(np.sum(weights * np.square(person.actual - person[method]))) for method in METHODS}
        mae_rows.append({"participant": participant, **mae})
        mse_rows.append({"participant": participant, **mse})
    return pd.DataFrame(mae_rows).set_index("participant"), pd.DataFrame(mse_rows).set_index("participant")


def score_outcomes(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    metrics, contrasts = [], []
    comparisons = (("F", "R", "R_vs_F"), ("M", "R", "R_vs_M"), ("R", "Q", "Q_vs_R"))
    for outcome, group in predictions.groupby("outcome", sort=False):
        for analysis in ("unweighted_primary", "ipw_sensitivity"):
            mae, mse = participant_losses(group, weighted=analysis.startswith("ipw"))
            for method in METHODS:
                metrics.append({
                    "outcome": outcome, "analysis": analysis, "method": method,
                    "participants": len(mae), "targets": len(group),
                    "mae_pb": float(mae[method].mean()),
                    "rmse_pb": float(np.sqrt(mse[method].mean())),
                })
            for reference, challenger, label in comparisons:
                mae_pair = mae[[reference, challenger]].to_numpy()
                mse_pair = mse[[reference, challenger]].to_numpy()
                old_mae, new_mae = mae_pair.mean(axis=0)
                old_rmse, new_rmse = np.sqrt(mse_pair.mean(axis=0))
                mae_draws, rmse_draws = np.empty(BOOTSTRAPS), np.empty(BOOTSTRAPS)
                for b in range(BOOTSTRAPS):
                    index = rng.integers(0, len(mae_pair), len(mae_pair))
                    bm = mae_pair[index].mean(axis=0)
                    bs = np.sqrt(mse_pair[index].mean(axis=0))
                    mae_draws[b] = 100 * (bm[1] / bm[0] - 1)
                    rmse_draws[b] = 100 * (bs[1] / bs[0] - 1)
                contrasts.append({
                    "outcome": outcome, "analysis": analysis, "contrast": label,
                    "reference": reference, "challenger": challenger,
                    "participants": len(mae), "targets": len(group),
                    "relative_mae_percent": float(100 * (new_mae / old_mae - 1)),
                    "mae_ci_low": float(np.quantile(mae_draws, 0.025)),
                    "mae_ci_high": float(np.quantile(mae_draws, 0.975)),
                    "relative_rmse_percent": float(100 * (new_rmse / old_rmse - 1)),
                    "rmse_ci_low": float(np.quantile(rmse_draws, 0.025)),
                    "rmse_ci_high": float(np.quantile(rmse_draws, 0.975)),
                })
    return pd.DataFrame(metrics), pd.DataFrame(contrasts)


def response_metrics(opportunities: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome, group in opportunities.groupby("outcome", sort=False):
        weights = participant_weights(group.participant)
        actual = group.completed.to_numpy()
        probability = group.response_probability.to_numpy()
        logit = np.log(np.clip(probability, 1e-6, 1 - 1e-6) / np.clip(1 - probability, 1e-6, 1))
        calibrator = LogisticRegression(C=1e6, max_iter=3000).fit(
            logit.reshape(-1, 1), actual, sample_weight=weights,
        )
        rows.append({
            "outcome_history": outcome,
            "participants": int(group.participant.nunique()),
            "opportunities": len(group), "complete": int(actual.sum()),
            "completion_fraction": float(np.average(actual, weights=weights)),
            "roc_auc": float(roc_auc_score(actual, probability, sample_weight=weights)),
            "pr_auc": float(average_precision_score(actual, probability, sample_weight=weights)),
            "brier": float(np.average(np.square(actual - probability), weights=weights)),
            "calibration_intercept": float(calibrator.intercept_[0]),
            "calibration_slope": float(calibrator.coef_[0, 0]),
            "probability_min": float(probability.min()),
            "probability_p01": float(np.quantile(probability, 0.01)),
            "probability_median": float(np.median(probability)),
            "probability_p99": float(np.quantile(probability, 0.99)),
            "probability_max": float(probability.max()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    if not PROTOCOL.exists():
        raise RuntimeError("Frozen protocol is missing")
    data = pd.read_csv(DATA, sep="\t")
    if data.duplicated(["id", "day", "beep"]).any() or not data.groupby("id").size().eq(63).all():
        raise AssertionError("Prompt opportunity grid invalid")
    predictions, opportunities, fold_audit = assemble(data)
    metrics, contrasts = score_outcomes(predictions)
    response = response_metrics(opportunities)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(OUTPUT / "marian_history_observation_predictions.parquet", index=False)
    opportunities.to_parquet(OUTPUT / "marian_response_predictions.parquet", index=False)
    metrics.to_csv(OUTPUT / "marian_history_observation_metrics.csv", index=False)
    contrasts.to_csv(OUTPUT / "marian_history_observation_contrasts.csv", index=False)
    response.to_csv(OUTPUT / "marian_response_metrics.csv", index=False)
    fold_audit.to_csv(OUTPUT / "marian_history_observation_fold_audit.csv", index=False)
    manifest = {
        "status": "externally sourced exploratory design-level validation; not OSF-confirmatory",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED, "bootstraps": BOOTSTRAPS,
        "data_sha256": sha256(DATA), "protocol_sha256": sha256(PROTOCOL),
        "script_sha256": sha256(Path(__file__)),
        "python": platform.python_version(), "pandas": pd.__version__,
        "numpy": np.__version__, "sklearn": sklearn.__version__,
        "participants": int(data.id.nunique()), "opportunities": len(data),
        "scored_targets": int(len(predictions) / len(OUTCOMES)),
        "outcomes": list(OUTCOMES), "methods": list(METHODS),
        "assumptions": [
            "status=1 defines completed primary targets",
            "EMA items are 4-hour adaptations and are not standard PHQ-2",
            "IPW sensitivity depends on observed-history MAR and clipped probabilities",
        ],
    }
    (OUTPUT / "marian_history_observation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(contrasts.to_string(index=False))
    print(response.to_string(index=False))


if __name__ == "__main__":
    main()
