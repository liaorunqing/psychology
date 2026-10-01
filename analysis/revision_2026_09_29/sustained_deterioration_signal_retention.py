from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "outputs" / "revision_2026_09_29"
INPUT = OUT / "equal_weight_mean_baseline_predictions.parquet"
PROTOCOL = ROOT / "direction_reset" / "85_sustained_deterioration_signal_retention_protocol_2026-09-30.md"
SEED = 20260918
BOOTSTRAPS = 2000
WINDOW = 8
PLATEAU = 8
METHODS = ("B8", "L8", "A50")
PRIMARY = {
    ("Dejonckheere", "sad"),
    ("Dejonckheere", "stressed"),
    ("CES", "phq2"),
    ("Marian", "depressed"),
}
DIRECTION = {
    "sad": 1.0,
    "stressed": 1.0,
    "angry": 1.0,
    "phq2": 1.0,
    "depressed": 1.0,
    "anhedonia": 1.0,
    "happy": -1.0,
    "relaxed": -1.0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="stable")
    values = np.asarray(values, dtype=float)[order]
    weights = np.asarray(weights, dtype=float)[order]
    if len(values) == 0 or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("Weighted quantile requires finite values and positive weights")
    cumulative = np.cumsum(weights) / weights.sum()
    return float(values[np.searchsorted(cumulative, q, side="left")])


def outcome_within_sd(frame: pd.DataFrame) -> float:
    variances = frame.groupby("participant", sort=False).actual.var(ddof=1).dropna()
    value = float(np.sqrt(variances.mean()))
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Outcome within-person SD is not positive")
    return value


def injection_offsets(n: int, onset_fraction: float, ramp: int, magnitude_raw: float) -> tuple[int, np.ndarray] | None:
    onset = max(16, int(np.floor(n * onset_fraction)))
    if onset + ramp + PLATEAU > n:
        return None
    offsets = np.zeros(n, dtype=float)
    offsets[onset : onset + ramp] = magnitude_raw * np.arange(1, ramp + 1) / ramp
    offsets[onset + ramp :] = magnitude_raw
    return onset, offsets


def prior_window_adaptation(offsets: np.ndarray, window: int = WINDOW) -> np.ndarray:
    adaptation = np.zeros_like(offsets, dtype=float)
    for index in range(len(offsets)):
        adaptation[index] = offsets[max(0, index - window) : index].sum() / window
    return adaptation


def prepare_participant(
    group: pd.DataFrame,
    within_sd: float,
    direction: float,
    magnitude_sd: float,
    onset_fraction: float,
    ramp: int,
) -> pd.DataFrame | None:
    group = group.sort_values("baseline_age", kind="stable").reset_index(drop=True).copy()
    specification = injection_offsets(len(group), onset_fraction, ramp, magnitude_sd * within_sd)
    if specification is None:
        return None
    onset, raw_offsets = specification
    signed_offsets = direction * raw_offsets
    rolling_adaptation = prior_window_adaptation(signed_offsets)
    original = {
        "B8": group.B8_mean.to_numpy(float),
        "L8": group.L8_mean.to_numpy(float),
    }
    original["A50"] = 0.5 * original["B8"] + 0.5 * original["L8"]
    injected = {
        "B8": original["B8"],
        "L8": original["L8"] + rolling_adaptation,
    }
    injected["A50"] = 0.5 * injected["B8"] + 0.5 * injected["L8"]
    actual = group.actual.to_numpy(float)
    injected_actual = actual + signed_offsets
    rows: list[pd.DataFrame] = []
    for method in METHODS:
        original_score = direction * (actual - original[method]) / within_sd
        injected_score = direction * (injected_actual - injected[method]) / within_sd
        current = pd.DataFrame(
            {
                "participant": str(group.participant.iloc[0]),
                "index": np.arange(len(group)),
                "method": method,
                "onset": onset,
                "phase": np.where(
                    np.arange(len(group)) < onset,
                    "pre",
                    np.where(np.arange(len(group)) < onset + ramp, "ramp", "plateau"),
                ),
                "original_score": original_score,
                "injected_score": injected_score,
                "score_increment": injected_score - original_score,
                "shift_sd": raw_offsets / within_sd,
            }
        )
        rows.append(current)
    return pd.concat(rows, ignore_index=True)


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    observed = float(np.mean(values))
    draws = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    samples = values[draws].mean(axis=1)
    return observed, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def analyse_scenario(
    frame: pd.DataFrame,
    dataset: str,
    outcome: str,
    magnitude_sd: float,
    onset_fraction: float,
    ramp: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    within_sd = outcome_within_sd(frame)
    direction = DIRECTION[outcome]
    pieces = []
    for _, group in frame.groupby("participant", sort=False):
        prepared = prepare_participant(
            group, within_sd, direction, magnitude_sd, onset_fraction, ramp
        )
        if prepared is not None:
            pieces.append(prepared)
    if not pieces:
        raise ValueError(f"No eligible participants for {dataset}/{outcome}")
    long = pd.concat(pieces, ignore_index=True)
    long.insert(0, "outcome", outcome)
    long.insert(0, "dataset", dataset)
    long["magnitude_sd"] = magnitude_sd
    long["onset_fraction"] = onset_fraction
    long["ramp"] = ramp

    thresholds: dict[str, float] = {}
    for method, method_frame in long[long.phase == "pre"].groupby("method"):
        counts = method_frame.groupby("participant").size()
        weights = method_frame.participant.map((1.0 / counts).to_dict()).to_numpy(float)
        thresholds[method] = weighted_quantile(
            method_frame.original_score.to_numpy(float), weights, 0.95
        )
    long["threshold"] = long.method.map(thresholds)
    long["original_alarm"] = long.original_score > long.threshold
    long["injected_alarm"] = long.injected_score > long.threshold

    participant_rows = []
    for (participant, method), group in long.groupby(["participant", "method"], sort=False):
        onset = int(group.onset.iloc[0])
        ramp_end = onset + ramp
        window = group[(group["index"] >= onset) & (group["index"] < ramp_end + PLATEAU)]
        plateau = group[(group["index"] >= ramp_end) & (group["index"] < ramp_end + PLATEAU)]
        alarms = np.flatnonzero(window.injected_alarm.to_numpy(bool))
        participant_rows.append(
            {
                "dataset": dataset,
                "outcome": outcome,
                "participant": participant,
                "method": method,
                "magnitude_sd": magnitude_sd,
                "onset_fraction": onset_fraction,
                "ramp": ramp,
                "targets_evaluated": len(window),
                "retention": float(plateau.score_increment.mean() / magnitude_sd),
                "false_alarm_rate": float(window.original_alarm.mean()),
                "detection_rate": float(window.injected_alarm.mean()),
                "excess_detection": float(window.injected_alarm.mean() - window.original_alarm.mean()),
                "detected": float(len(alarms) > 0),
                "delay_censored": float(alarms[0] if len(alarms) else len(window)),
            }
        )
    participant = pd.DataFrame(participant_rows)

    summary_rows = []
    for method, group in participant.groupby("method", sort=False):
        for metric in [
            "retention",
            "false_alarm_rate",
            "detection_rate",
            "excess_detection",
            "detected",
            "delay_censored",
        ]:
            estimate, low, high = bootstrap_mean(group[metric].to_numpy(float), rng)
            summary_rows.append(
                {
                    "dataset": dataset,
                    "outcome": outcome,
                    "method": method,
                    "magnitude_sd": magnitude_sd,
                    "onset_fraction": onset_fraction,
                    "ramp": ramp,
                    "metric": metric,
                    "participants": group.participant.nunique(),
                    "targets": int(group.targets_evaluated.sum()),
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return long, participant, pd.DataFrame(summary_rows)


def predictive_tradeoff(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    work = frame.copy()
    work["A50"] = 0.5 * work.B8_mean + 0.5 * work.L8_mean
    rows = []
    for (dataset, outcome), group in work.groupby(["dataset", "outcome"], sort=True):
        for method, column in {"B8": "B8_mean", "L8": "L8_mean", "A50": "A50"}.items():
            person = group.assign(ae=np.abs(group.actual - group[column])).groupby("participant").ae.mean()
            estimate, low, high = bootstrap_mean(person.to_numpy(float), rng)
            rows.append({"dataset": dataset, "outcome": outcome, "method": method,
                         "participants": len(person), "targets": len(group),
                         "mae": estimate, "mae_ci_low": low, "mae_ci_high": high})
    result = pd.DataFrame(rows)
    b8 = result[result.method == "B8"][["dataset", "outcome", "mae"]].rename(columns={"mae": "b8_mae"})
    result = result.merge(b8, on=["dataset", "outcome"], validate="many_to_one")
    result["relative_mae_vs_b8"] = result.mae / result.b8_mae - 1.0
    return result


def main() -> None:
    if not PROTOCOL.exists():
        raise FileNotFoundError("Frozen protocol is missing")
    data = pd.read_parquet(INPUT)
    required = {"dataset", "outcome", "participant", "actual", "baseline_age", "B8_mean", "L8_mean"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing columns: {required - set(data.columns)}")
    if data[list(required)].isna().any().any():
        raise ValueError("Required input contains missing values")
    rng = np.random.default_rng(SEED)
    long_parts, participant_parts, summary_parts, audit_rows = [], [], [], []
    scenarios = [(m, o, r) for m in (0.5, 1.0) for o in (0.4, 0.5, 0.6) for r in (4, 8, 16)]
    for (dataset, outcome), group in data.groupby(["dataset", "outcome"], sort=True):
        for magnitude, onset, ramp in scenarios:
            try:
                long, participant, summary = analyse_scenario(
                    group, dataset, outcome, magnitude, onset, ramp, rng
                )
            except ValueError as error:
                if not str(error).startswith("No eligible participants"):
                    raise
                audit_rows.append({"dataset": dataset, "outcome": outcome,
                                   "magnitude_sd": magnitude, "onset_fraction": onset,
                                   "ramp": ramp, "participants": 0, "status": "ineligible"})
                continue
            long_parts.append(long)
            participant_parts.append(participant)
            summary_parts.append(summary)
            audit_rows.append({"dataset": dataset, "outcome": outcome,
                               "magnitude_sd": magnitude, "onset_fraction": onset,
                               "ramp": ramp,
                               "participants": participant.participant.nunique(),
                               "status": "analysed"})
    long = pd.concat(long_parts, ignore_index=True)
    participant = pd.concat(participant_parts, ignore_index=True)
    summary = pd.concat(summary_parts, ignore_index=True)
    tradeoff = predictive_tradeoff(data, rng)

    primary = summary[
        summary.apply(lambda row: (row.dataset, row.outcome) in PRIMARY, axis=1)
        & (summary.magnitude_sd == 1.0)
        & (summary.onset_fraction == 0.5)
        & (summary.ramp == 8)
    ].copy()
    if not np.allclose(
        primary[(primary.method == "B8") & (primary.metric == "retention")].estimate,
        1.0,
        atol=1e-12,
    ):
        raise AssertionError("Frozen B8 must retain exactly all injected signal")
    OUT.mkdir(parents=True, exist_ok=True)
    long.to_parquet(OUT / "sustained_deterioration_target_scores.parquet", index=False)
    participant.to_parquet(OUT / "sustained_deterioration_participant_metrics.parquet", index=False)
    summary.to_csv(OUT / "sustained_deterioration_sensitivity_summary.csv", index=False)
    primary.to_csv(OUT / "sustained_deterioration_primary_summary.csv", index=False)
    tradeoff.to_csv(OUT / "sustained_deterioration_predictive_tradeoff.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(OUT / "sustained_deterioration_scenario_audit.csv", index=False)
    manifest = {
        "analysis": "semi-synthetic sustained deterioration signal-retention stress test",
        "status": "post-hoc methodological extension; not clinical safety validation",
        "seed": SEED,
        "bootstrap_iterations": BOOTSTRAPS,
        "window": WINDOW,
        "plateau_reports": PLATEAU,
        "primary_scenario": {"magnitude_sd": 1.0, "onset_fraction": 0.5, "ramp": 8},
        "primary_outcomes": sorted([list(item) for item in PRIMARY]),
        "methods": list(METHODS),
        "participant_is_independent_unit": True,
        "input_sha256": sha256(INPUT),
        "protocol_sha256": sha256(PROTOCOL),
        "script_sha256": sha256(Path(__file__)),
    }
    (OUT / "sustained_deterioration_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(primary.to_string(index=False))
    print("\nPredictive trade-off (primary outcomes)")
    print(tradeoff[tradeoff.apply(lambda row: (row.dataset, row.outcome) in PRIMARY, axis=1)].to_string(index=False))


if __name__ == "__main__":
    main()
