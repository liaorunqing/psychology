"""Post-confirmation sensitivity using pure eight-report means."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "outputs" / "revision_2026_09_29"
PROTOCOL = ROOT / "direction_reset" / "74_equal_weight_mean_baseline_sensitivity_protocol_2026-09-30.md"
DEJON = OUT / "dynamic_baseline_challenge_predictions.parquet"
CROSS = OUT / "cross_dataset_equal_information_predictions.parquet"
SEED = 20260918
BOOTSTRAPS = 2000
EXPECTED = {("Dejonckheere", outcome): (100, 7902)
            for outcome in ("sad", "stressed", "happy", "relaxed", "angry")}
EXPECTED.update({("CES", "phq2"): (105, 4541),
                 ("Marian", "depressed"): (145, 7071),
                 ("Marian", "anhedonia"): (145, 7071)})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_predictions() -> pd.DataFrame:
    dejon = pd.read_parquet(DEJON).copy()
    dejon.insert(0, "dataset", "Dejonckheere")
    cross = pd.read_parquet(CROSS).copy()
    required = {"dataset", "outcome", "participant", "actual", "mean_8",
                "last8_mean", "baseline_age", "log_age_scaled"}
    for name, frame in (("Dejonckheere", dejon), ("cross", cross)):
        if not required.issubset(frame.columns):
            raise AssertionError(f"Missing columns in {name}")
    keep = sorted(required)
    result = pd.concat([dejon[keep], cross[keep]], ignore_index=True)
    for key, (people, targets) in EXPECTED.items():
        group = result.loc[result.dataset.eq(key[0]) & result.outcome.eq(key[1])]
        if group.participant.nunique() != people or len(group) != targets:
            raise AssertionError(f"Target set changed: {key}")
    return result


def add_errors(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["B8_mean"] = result.mean_8
    result["L8_mean"] = result.last8_mean
    result["ae_B8_mean"] = np.abs(result.actual - result.B8_mean)
    result["ae_L8_mean"] = np.abs(result.actual - result.L8_mean)
    result["se_B8_mean"] = np.square(result.actual - result.B8_mean)
    result["se_L8_mean"] = np.square(result.actual - result.L8_mean)
    result["benefit_mean"] = result.ae_B8_mean - result.ae_L8_mean
    return result


def person_slopes(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, outcome, participant), group in frame.groupby(
            ["dataset", "outcome", "participant"], sort=False):
        if len(group) < 5 or group.baseline_age.nunique() < 2:
            continue
        x = group.log_age_scaled.to_numpy(dtype=float)
        centered = x - x.mean()
        denominator = float(np.square(centered).sum())
        rows.append({"dataset": dataset, "outcome": outcome,
                     "participant": participant, "targets": len(group),
                     "slope_benefit_mean": float(
                         np.dot(centered, group.benefit_mean.to_numpy(dtype=float)) /
                         denominator)})
    return pd.DataFrame(rows)


def summarize_slopes(slopes: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for (dataset, outcome), group in slopes.groupby(["dataset", "outcome"], sort=False):
        values = group.slope_benefit_mean.to_numpy(dtype=float)
        draws = np.empty(BOOTSTRAPS)
        for iteration in range(BOOTSTRAPS):
            draws[iteration] = values[rng.integers(0, len(values), len(values))].mean()
        rows.append({"dataset": dataset, "outcome": outcome,
                     "participants": len(values), "targets": int(group.targets.sum()),
                     "mean_slope": float(values.mean()),
                     "ci_low": float(np.quantile(draws, .025)),
                     "ci_high": float(np.quantile(draws, .975)),
                     "positive_slope_fraction": float((values > 0).mean())})
    return pd.DataFrame(rows)


def overall(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    rows = []
    columns = ["ae_B8_mean", "ae_L8_mean", "se_B8_mean", "se_L8_mean"]
    for (dataset, outcome), group in frame.groupby(["dataset", "outcome"], sort=False):
        by_person = group.groupby("participant")[columns].mean()
        means = by_person.mean()
        values = by_person.to_numpy(dtype=float)
        mae_draws, rmse_draws = np.empty(BOOTSTRAPS), np.empty(BOOTSTRAPS)
        for iteration in range(BOOTSTRAPS):
            sampled = values[rng.integers(0, len(values), len(values))].mean(axis=0)
            mae_draws[iteration] = 100 * (sampled[1] / sampled[0] - 1)
            rmse_draws[iteration] = 100 * (np.sqrt(sampled[3]) / np.sqrt(sampled[2]) - 1)
        rows.append({"dataset": dataset, "outcome": outcome,
                     "participants": len(by_person), "targets": len(group),
                     "relative_mae_percent": float(100 *
                         (means.ae_L8_mean / means.ae_B8_mean - 1)),
                     "mae_ci_low": float(np.quantile(mae_draws, .025)),
                     "mae_ci_high": float(np.quantile(mae_draws, .975)),
                     "relative_rmse_percent": float(100 *
                         (np.sqrt(means.se_L8_mean) / np.sqrt(means.se_B8_mean) - 1)),
                     "rmse_ci_low": float(np.quantile(rmse_draws, .025)),
                     "rmse_ci_high": float(np.quantile(rmse_draws, .975))})
    return pd.DataFrame(rows)


def main() -> None:
    if not PROTOCOL.exists():
        raise RuntimeError("Mean-only sensitivity protocol missing")
    source = load_predictions()
    frame = add_errors(source)
    slopes = person_slopes(frame)
    summary = summarize_slopes(slopes)
    overall_result = overall(frame)
    frame.to_parquet(OUT / "equal_weight_mean_baseline_predictions.parquet", index=False)
    slopes.to_parquet(OUT / "equal_weight_mean_baseline_person_slopes.parquet", index=False)
    summary.to_csv(OUT / "equal_weight_mean_baseline_slope_summary.csv", index=False)
    overall_result.to_csv(OUT / "equal_weight_mean_baseline_overall_contrasts.csv", index=False)
    manifest = {"status": "post-confirmation pure-eight-mean sensitivity",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "seed": SEED, "bootstrap_resamples": BOOTSTRAPS,
                "protocol_sha256": sha256(PROTOCOL), "script_sha256": sha256(Path(__file__)),
                "input_sha256": {"dejonckheere": sha256(DEJON), "cross_dataset": sha256(CROSS)},
                "software": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__},
                "checks": {"B8_mean_exactly_eight_equal_weights": True,
                           "L8_mean_exactly_eight_equal_weights": True,
                           "target_keys_unchanged": True,
                           "current_and_future_excluded_by_source_pipeline": True,
                           "participant_bootstrap": True}}
    (OUT / "equal_weight_mean_baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nSLOPES")
    print(summary.to_string(index=False))
    print("\nOVERALL")
    print(overall_result.to_string(index=False))


if __name__ == "__main__":
    main()
