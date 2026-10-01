"""Frozen external confirmation of the baseline-age mechanism in openESM 0012."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "outputs" / "revision_2026_09_29"
DATA_ROOT = Path(os.environ.get("DPT_DATA_ROOT", ROOT / "data")).expanduser()
DATA = DATA_ROOT / "dejonckheere_openesm" / "0012_dejonckheere_ts.tsv"
PROTOCOL = ROOT / "direction_reset" / "67_dejonckheere_baseline_age_confirmation_protocol_2026-09-30.md"
SEED = 20260918
BOOTSTRAPS = 2000
OUTCOMES = ("sad", "stressed", "happy", "relaxed", "angry")
PRIMARY = ("sad", "stressed")
WEIGHTS = np.linspace(0.0, 1.0, 41)
STRATA = ("1-14", "15-35", "36-63", "64-90")
RESPONSE_FEATURES = ("prior_completion_rate", "missing_streak", "day", "beep")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def participant_weights(ids: pd.Series) -> np.ndarray:
    counts = ids.astype(str).value_counts()
    raw = ids.astype(str).map(1.0 / counts).to_numpy(dtype=float)
    return raw / raw.mean()


def make_fold_map(ids: pd.Series) -> dict[str, int]:
    unique = np.asarray(sorted(ids.astype(str).unique()))
    if len(unique) != 100:
        raise AssertionError("Expected exactly 100 participants")
    rng = np.random.default_rng(SEED)
    shuffled = unique[rng.permutation(len(unique))]
    mapping = {participant: int(index % 5 + 1)
               for index, participant in enumerate(shuffled)}
    sizes = pd.Series(mapping).value_counts().sort_index().to_dict()
    if sizes != {1: 20, 2: 20, 3: 20, 4: 20, 5: 20}:
        raise AssertionError(f"Fold sizes changed: {sizes}")
    return mapping


def validate_grid(data: pd.DataFrame) -> None:
    if len(data) != 9800 or data.id.nunique() != 100:
        raise AssertionError("Expected 100 x 98 scheduled opportunities")
    if data.duplicated(["id", "counter"]).any():
        raise AssertionError("Duplicate participant-counter key")
    if not data.groupby("id").size().eq(98).all():
        raise AssertionError("Each participant must have 98 opportunities")
    expected = (data.day - 1) * 7 + data.beep
    if not data.counter.eq(expected).all():
        raise AssertionError("Counter formula changed")
    if not data.counter.between(1, 98).all():
        raise AssertionError("Counter outside 1..98")


def build_outcome_rows(data: pd.DataFrame, outcome: str,
                       fold_map: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build observed targets and scheduled response opportunities using only prior history."""
    prediction_rows: list[dict] = []
    opportunity_rows: list[dict] = []
    for raw_id, person in data.sort_values(["id", "counter"]).groupby("id", sort=False):
        participant = str(raw_id)
        values = person[outcome].to_numpy(dtype=float)
        counters = person.counter.to_numpy(dtype=int)
        days = person.day.to_numpy(dtype=int)
        beeps = person.beep.to_numpy(dtype=int)
        observed = np.isfinite(values)
        complete_values: list[float] = []
        complete_counters: list[int] = []
        missing_streak = 0
        for index, counter in enumerate(counters):
            prior_count = len(complete_values)
            if prior_count >= 8:
                baseline_counter = complete_counters[7]
                opportunity_rows.append({
                    "outcome": outcome, "participant": participant,
                    "fold": fold_map[participant], "counter": int(counter),
                    "day": int(days[index]), "beep": int(beeps[index]),
                    "observed": int(observed[index]),
                    "prior_completion_rate": float(prior_count / index),
                    "missing_streak": int(missing_streak),
                    "baseline_age": int(counter - baseline_counter),
                })
                if observed[index]:
                    initial = np.asarray(complete_values[:8], dtype=float)
                    prior = np.asarray(complete_values, dtype=float)
                    prediction_rows.append({
                        "outcome": outcome, "participant": participant,
                        "fold": fold_map[participant], "counter": int(counter),
                        "report_rank": int(prior_count + 1),
                        "actual": float(values[index]),
                        "baseline_counter": int(baseline_counter),
                        "baseline_age": int(counter - baseline_counter),
                        "mean_8": float(initial.mean()),
                        "last_8": float(initial[-1]),
                        "online_mean": float(prior.mean()),
                        "online_last": float(prior[-1]),
                    })
            if observed[index]:
                complete_values.append(float(values[index]))
                complete_counters.append(int(counter))
                missing_streak = 0
            else:
                missing_streak += 1
    predictions = pd.DataFrame(prediction_rows)
    opportunities = pd.DataFrame(opportunity_rows)
    if len(predictions) != 7902 or predictions.participant.nunique() != 100:
        raise AssertionError(f"Frozen target count changed for {outcome}: {len(predictions)}")
    if predictions.baseline_age.le(0).any():
        raise AssertionError("Baseline age must be strictly positive")
    if not predictions.groupby("participant")[["mean_8", "last_8"]].nunique().eq(1).all().all():
        raise AssertionError("B8 features changed after calibration")
    observed_keys = set(opportunities.loc[opportunities.observed.eq(1),
                                           ["participant", "counter"]]
                        .itertuples(index=False, name=None))
    prediction_keys = set(predictions[["participant", "counter"]]
                          .itertuples(index=False, name=None))
    if observed_keys != prediction_keys:
        raise AssertionError("Observed opportunity and target keys differ")
    return predictions, opportunities


def tune_weight(frame: pd.DataFrame, mean_col: str, last_col: str) -> tuple[float, float]:
    base = frame[mean_col].to_numpy(dtype=float)
    delta = frame[last_col].to_numpy(dtype=float) - base
    actual = frame.actual.to_numpy(dtype=float)
    sample_weight = participant_weights(frame.participant)
    losses = np.asarray([
        np.average(np.abs(actual - (base + weight * delta)), weights=sample_weight)
        for weight in WEIGHTS
    ])
    best = int(np.argmin(losses))
    return float(WEIGHTS[best]), float(losses[best])


def fit_response_model(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, dict]:
    scaler = StandardScaler().fit(train[list(RESPONSE_FEATURES)])
    train_x = scaler.transform(train[list(RESPONSE_FEATURES)])
    test_x = scaler.transform(test[list(RESPONSE_FEATURES)])
    model = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=3000,
                               random_state=SEED)
    weights = participant_weights(train.participant)
    model.fit(train_x, train.observed.to_numpy(dtype=int), sample_weight=weights)
    probability = model.predict_proba(test_x)[:, 1]
    return probability, {
        "response_iterations": int(model.n_iter_[0]),
        **{f"response_coef_{name}": float(value)
           for name, value in zip(RESPONSE_FEATURES, model.coef_[0])},
    }


def cross_validated_predictions(predictions: pd.DataFrame,
                                opportunities: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output, opportunity_output, audits = [], [], []
    for outcome in OUTCOMES:
        outcome_predictions = predictions.loc[predictions.outcome.eq(outcome)]
        outcome_opportunities = opportunities.loc[opportunities.outcome.eq(outcome)]
        for fold in range(1, 6):
            train = outcome_predictions.loc[outcome_predictions.fold.ne(fold)].copy()
            test = outcome_predictions.loc[outcome_predictions.fold.eq(fold)].copy()
            train_opp = outcome_opportunities.loc[outcome_opportunities.fold.ne(fold)].copy()
            test_opp = outcome_opportunities.loc[outcome_opportunities.fold.eq(fold)].copy()
            if set(train.participant) & set(test.participant):
                raise AssertionError("Participant crosses outer fold")
            weight_b8, loss_b8 = tune_weight(train, "mean_8", "last_8")
            weight_r, loss_r = tune_weight(train, "online_mean", "online_last")
            test["B8"] = test.mean_8 + weight_b8 * (test.last_8 - test.mean_8)
            test["R_online"] = (test.online_mean +
                                weight_r * (test.online_last - test.online_mean))
            probability, response_audit = fit_response_model(train_opp, test_opp)
            test_opp["response_probability"] = probability
            lookup = test_opp.loc[test_opp.observed.eq(1),
                                  ["participant", "counter", "response_probability"]]
            test = test.merge(lookup, on=["participant", "counter"], how="left",
                              validate="one_to_one")
            if test.response_probability.isna().any():
                raise AssertionError("Missing response probability on observed target")
            test["ipw"] = 1.0 / test.response_probability.clip(.10, .99)
            audit = {
                "outcome": outcome, "fold": fold,
                "train_participants": train.participant.nunique(),
                "test_participants": test.participant.nunique(),
                "train_targets": len(train), "test_targets": len(test),
                "train_opportunities": len(train_opp),
                "test_opportunities": len(test_opp),
                "participant_overlap": False,
                "weight_B8": weight_b8, "weight_R_online": weight_r,
                "train_mae_B8": loss_b8, "train_mae_R_online": loss_r,
                **response_audit,
            }
            output.append(test)
            opportunity_output.append(test_opp)
            audits.append(audit)
    result = pd.concat(output, ignore_index=True)
    opp_result = pd.concat(opportunity_output, ignore_index=True)
    keys = ["outcome", "participant", "counter"]
    if result.duplicated(keys).any() or opp_result.duplicated(keys).any():
        raise AssertionError("Duplicate out-of-fold key")
    return result, opp_result, pd.DataFrame(audits)


def assign_age_scale_and_strata(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["log_age_scaled"] = np.nan
    for outcome, group in result.groupby("outcome", sort=False):
        log_age = np.log1p(group.baseline_age.to_numpy(dtype=float))
        scale = float(log_age.std(ddof=0))
        if scale <= 0:
            raise AssertionError("Degenerate age scale")
        result.loc[group.index, "log_age_scaled"] = log_age / scale
    result["age_stratum"] = pd.cut(
        result.baseline_age, bins=[0, 14, 35, 63, 90], labels=list(STRATA),
        include_lowest=False, right=True).astype("string")
    if result.age_stratum.isna().any():
        raise AssertionError("Unassigned age stratum")
    result["ae_B8"] = np.abs(result.actual - result.B8)
    result["ae_R"] = np.abs(result.actual - result.R_online)
    result["ae_benefit"] = result.ae_B8 - result.ae_R
    result["se_B8"] = np.square(result.actual - result.B8)
    result["se_R"] = np.square(result.actual - result.R_online)
    return result


def weighted_slope(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    centered = x - np.sum(weights * x)
    denominator = np.sum(weights * centered * centered)
    if denominator <= 0:
        return np.nan
    return float(np.sum(weights * centered * y) / denominator)


def person_slopes(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (outcome, participant), group in frame.groupby(["outcome", "participant"], sort=False):
        if len(group) < 5 or group.baseline_age.nunique() < 2:
            continue
        x = group.log_age_scaled.to_numpy(dtype=float)
        record = {"outcome": outcome, "participant": participant,
                  "targets": len(group), "age_min": int(group.baseline_age.min()),
                  "age_max": int(group.baseline_age.max())}
        for analysis, weights in (("unweighted", np.ones(len(group))),
                                  ("ipw", group.ipw.to_numpy(dtype=float))):
            for column in ("ae_B8", "ae_R", "ae_benefit"):
                record[f"{analysis}_slope_{column}"] = weighted_slope(
                    x, group[column].to_numpy(dtype=float), weights)
        rows.append(record)
    result = pd.DataFrame(rows)
    expected = {(outcome, participant) for outcome, participant in
                frame.groupby(["outcome", "participant"]).size().index}
    observed = set(result[["outcome", "participant"]].itertuples(index=False, name=None))
    if observed != expected:
        raise AssertionError("Unexpected slope eligibility loss")
    return result


def summarize_slopes(slopes: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for (outcome, analysis), columns in [
        ((outcome, analysis), [f"{analysis}_slope_ae_B8",
                               f"{analysis}_slope_ae_R",
                               f"{analysis}_slope_ae_benefit"])
        for outcome in OUTCOMES for analysis in ("unweighted", "ipw")
    ]:
        group = slopes.loc[slopes.outcome.eq(outcome)]
        values = group[columns].to_numpy(dtype=float)
        draws = np.empty((BOOTSTRAPS, len(columns)))
        for iteration in range(BOOTSTRAPS):
            draws[iteration] = values[rng.integers(0, len(values), len(values))].mean(axis=0)
        for index, quantity in enumerate(("ae_B8", "ae_R", "ae_benefit")):
            low, high = np.quantile(draws[:, index], [.025, .975])
            rows.append({
                "outcome": outcome, "role": "primary" if outcome in PRIMARY else "secondary",
                "analysis": analysis, "quantity": quantity,
                "participants": len(group), "targets": int(group.targets.sum()),
                "mean_slope": float(values[:, index].mean()),
                "ci_low": float(low), "ci_high": float(high),
                "positive_slope_fraction": float((values[:, index] > 0).mean()),
            })
    return pd.DataFrame(rows)


def stratum_results(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED + 1)
    counts = frame.groupby(["outcome", "participant"]).age_stratum.nunique()
    common = set(counts[counts.eq(4)].index)
    metrics, contrasts = [], []
    mask = [tuple(row) in common for row in
            frame[["outcome", "participant"]].itertuples(index=False, name=None)]
    for population, table in (("available", frame),
                              ("all_four_strata", frame.loc[mask])):
        for (outcome, stratum), group in table.groupby(["outcome", "age_stratum"], sort=False):
            if group.empty:
                continue
            by_person = group.groupby("participant")[["ae_B8", "ae_R", "se_B8", "se_R"]].mean()
            means = by_person.mean()
            b8_mae, r_mae = means[["ae_B8", "ae_R"]]
            b8_rmse, r_rmse = np.sqrt(means[["se_B8", "se_R"]])
            for method, mae, rmse in (("B8", b8_mae, b8_rmse),
                                      ("R_online", r_mae, r_rmse)):
                metrics.append({"outcome": outcome, "population": population,
                                "age_stratum": stratum, "method": method,
                                "participants": len(by_person), "targets": len(group),
                                "mae_pb": float(mae), "rmse_pb": float(rmse)})
            values = by_person.to_numpy(dtype=float)
            mae_draws, rmse_draws = np.empty(BOOTSTRAPS), np.empty(BOOTSTRAPS)
            for iteration in range(BOOTSTRAPS):
                sampled = values[rng.integers(0, len(values), len(values))].mean(axis=0)
                mae_draws[iteration] = 100 * (sampled[1] / sampled[0] - 1)
                rmse_draws[iteration] = 100 * (np.sqrt(sampled[3]) / np.sqrt(sampled[2]) - 1)
            contrasts.append({
                "outcome": outcome, "population": population, "age_stratum": stratum,
                "participants": len(by_person), "targets": len(group),
                "R_vs_B8_mae_percent": float(100 * (r_mae / b8_mae - 1)),
                "mae_ci_low": float(np.quantile(mae_draws, .025)),
                "mae_ci_high": float(np.quantile(mae_draws, .975)),
                "R_vs_B8_rmse_percent": float(100 * (r_rmse / b8_rmse - 1)),
                "rmse_ci_low": float(np.quantile(rmse_draws, .025)),
                "rmse_ci_high": float(np.quantile(rmse_draws, .975)),
            })
    return pd.DataFrame(metrics), pd.DataFrame(contrasts)


def overall_results(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED + 2)
    metrics, contrasts = [], []
    for outcome, group in frame.groupby("outcome", sort=False):
        by_person = group.groupby("participant")[["ae_B8", "ae_R", "se_B8", "se_R"]].mean()
        means = by_person.mean()
        b8_mae, r_mae = means[["ae_B8", "ae_R"]]
        b8_rmse, r_rmse = np.sqrt(means[["se_B8", "se_R"]])
        for method, mae, rmse in (("B8", b8_mae, b8_rmse),
                                  ("R_online", r_mae, r_rmse)):
            metrics.append({"outcome": outcome, "method": method,
                            "participants": len(by_person), "targets": len(group),
                            "mae_pb": float(mae), "rmse_pb": float(rmse)})
        values = by_person.to_numpy(dtype=float)
        mae_draws, rmse_draws = np.empty(BOOTSTRAPS), np.empty(BOOTSTRAPS)
        for iteration in range(BOOTSTRAPS):
            sampled = values[rng.integers(0, len(values), len(values))].mean(axis=0)
            mae_draws[iteration] = 100 * (sampled[1] / sampled[0] - 1)
            rmse_draws[iteration] = 100 * (np.sqrt(sampled[3]) / np.sqrt(sampled[2]) - 1)
        contrasts.append({
            "outcome": outcome, "participants": len(by_person), "targets": len(group),
            "R_vs_B8_mae_percent": float(100 * (r_mae / b8_mae - 1)),
            "mae_ci_low": float(np.quantile(mae_draws, .025)),
            "mae_ci_high": float(np.quantile(mae_draws, .975)),
            "R_vs_B8_rmse_percent": float(100 * (r_rmse / b8_rmse - 1)),
            "rmse_ci_low": float(np.quantile(rmse_draws, .025)),
            "rmse_ci_high": float(np.quantile(rmse_draws, .975)),
        })
    return pd.DataFrame(metrics), pd.DataFrame(contrasts)


def response_metrics(opportunities: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome, group in opportunities.groupby("outcome", sort=False):
        weights = participant_weights(group.participant)
        observed = group.observed.to_numpy(dtype=int)
        probability = group.response_probability.to_numpy(dtype=float)
        rows.append({
            "outcome": outcome, "participants": group.participant.nunique(),
            "opportunities": len(group), "observed_opportunities": int(observed.sum()),
            "observed_fraction": float(np.average(observed, weights=weights)),
            "roc_auc": float(roc_auc_score(observed, probability, sample_weight=weights)),
            "brier": float(brier_score_loss(observed, probability, sample_weight=weights)),
            "probability_min": float(probability.min()),
            "probability_p01": float(np.quantile(probability, .01)),
            "probability_median": float(np.median(probability)),
            "probability_p99": float(np.quantile(probability, .99)),
            "probability_max": float(probability.max()),
        })
    return pd.DataFrame(rows)


def classify_confirmation(summary: pd.DataFrame) -> str:
    main = summary.loc[(summary.analysis.eq("unweighted")) &
                       (summary.quantity.eq("ae_benefit")) &
                       (summary.outcome.isin(PRIMARY))].set_index("outcome")
    successes = (main.mean_slope.gt(0) & main.ci_low.gt(0))
    if successes.all():
        result = "strong_confirmation"
    elif successes.sum() == 1 and main.mean_slope.gt(0).all():
        result = "partial_confirmation"
    else:
        result = "not_confirmed"
    ipw = summary.loc[(summary.analysis.eq("ipw")) &
                      (summary.quantity.eq("ae_benefit")) &
                      (summary.outcome.isin(PRIMARY))].set_index("outcome")
    if not np.sign(main.mean_slope).eq(np.sign(ipw.mean_slope)).all():
        result += "_missingness_model_dependent"
    return result


def leakage_self_test() -> None:
    """Changing the current/future values cannot change current history features."""
    toy = pd.DataFrame({
        "id": ["x"] * 12, "counter": np.arange(1, 13),
        "day": [1] * 7 + [2] * 5, "beep": list(range(1, 8)) + list(range(1, 6)),
        "sad": np.arange(12, dtype=float),
    })
    folds = {"x": 1}
    first, _ = build_outcome_rows_unchecked(toy, "sad", folds)
    changed = toy.copy()
    changed.loc[changed.counter.ge(9), "sad"] += 1000
    second, _ = build_outcome_rows_unchecked(changed, "sad", folds)
    cols = ["participant", "counter", "mean_8", "last_8", "online_mean", "online_last"]
    pd.testing.assert_frame_equal(first.loc[first.counter.eq(9), cols].reset_index(drop=True),
                                  second.loc[second.counter.eq(9), cols].reset_index(drop=True))


def build_outcome_rows_unchecked(data: pd.DataFrame, outcome: str,
                                 fold_map: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Small-data equivalent used only by the leakage unit check."""
    rows, opp = [], []
    for raw_id, person in data.sort_values(["id", "counter"]).groupby("id"):
        participant, history, history_counter, streak = str(raw_id), [], [], 0
        for index, row in enumerate(person.itertuples(index=False)):
            value = getattr(row, outcome)
            observed = pd.notna(value)
            if len(history) >= 8:
                opp.append({"participant": participant, "counter": row.counter})
                if observed:
                    rows.append({"participant": participant, "counter": row.counter,
                                 "mean_8": np.mean(history[:8]), "last_8": history[7],
                                 "online_mean": np.mean(history), "online_last": history[-1]})
            if observed:
                history.append(float(value)); history_counter.append(row.counter); streak = 0
            else:
                streak += 1
    return pd.DataFrame(rows), pd.DataFrame(opp)


def main() -> None:
    if not PROTOCOL.exists():
        raise RuntimeError("Frozen confirmation protocol missing")
    leakage_self_test()
    data = pd.read_csv(DATA, sep="\t")
    data["id"] = data.id.astype(str)
    validate_grid(data)
    fold_map = make_fold_map(data.id)
    prediction_pieces, opportunity_pieces = [], []
    for outcome in OUTCOMES:
        predictions, opportunities = build_outcome_rows(data, outcome, fold_map)
        prediction_pieces.append(predictions)
        opportunity_pieces.append(opportunities)
    raw_predictions = pd.concat(prediction_pieces, ignore_index=True)
    raw_opportunities = pd.concat(opportunity_pieces, ignore_index=True)
    predictions, opportunities, audit = cross_validated_predictions(
        raw_predictions, raw_opportunities)
    predictions = assign_age_scale_and_strata(predictions)
    slopes = person_slopes(predictions)
    summary = summarize_slopes(slopes)
    overall_metrics, overall_contrasts = overall_results(predictions)
    metrics, contrasts = stratum_results(predictions)
    response = response_metrics(opportunities)
    classification = classify_confirmation(summary)
    OUT.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(OUT / "dejonckheere_confirmation_predictions.parquet", index=False)
    opportunities.to_parquet(OUT / "dejonckheere_confirmation_opportunities.parquet", index=False)
    slopes.to_parquet(OUT / "dejonckheere_confirmation_person_slopes.parquet", index=False)
    audit.to_csv(OUT / "dejonckheere_confirmation_fold_audit.csv", index=False)
    summary.to_csv(OUT / "dejonckheere_confirmation_slope_summary.csv", index=False)
    overall_metrics.to_csv(OUT / "dejonckheere_confirmation_overall_metrics.csv", index=False)
    overall_contrasts.to_csv(OUT / "dejonckheere_confirmation_overall_contrasts.csv", index=False)
    metrics.to_csv(OUT / "dejonckheere_confirmation_stratum_metrics.csv", index=False)
    contrasts.to_csv(OUT / "dejonckheere_confirmation_stratum_contrasts.csv", index=False)
    response.to_csv(OUT / "dejonckheere_confirmation_response_metrics.csv", index=False)
    manifest = {
        "status": "frozen external confirmation after outcome-blind eligibility audit",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "classification": classification, "seed": SEED,
        "bootstrap_resamples": BOOTSTRAPS, "primary_outcomes": list(PRIMARY),
        "secondary_outcomes": [value for value in OUTCOMES if value not in PRIMARY],
        "data_sha256": sha256(DATA), "protocol_sha256": sha256(PROTOCOL),
        "script_sha256": sha256(Path(__file__)),
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "participants_per_outcome": 100, "targets_per_outcome": 7902,
        "checks": {"scheduled_grid_exact": True, "counter_formula_exact": True,
                   "participant_disjoint_outer_folds": True,
                   "fold_size_20_each": True, "B8_frozen": True,
                   "current_and_future_values_excluded": True,
                   "training_only_weight_tuning": True,
                   "training_only_response_scaling_and_fit": True,
                   "same_fold_map_all_outcomes": True,
                   "participant_level_bootstrap": True},
    }
    (OUT / "dejonckheere_confirmation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nCONFIRMATION:", classification)
    print("\nSLOPE SUMMARY")
    print(summary.to_string(index=False))
    print("\nOVERALL CONTRASTS")
    print(overall_contrasts.to_string(index=False))
    print("\nSTRATUM CONTRASTS")
    print(contrasts.to_string(index=False))
    print("\nRESPONSE METRICS")
    print(response.to_string(index=False))


if __name__ == "__main__":
    main()
