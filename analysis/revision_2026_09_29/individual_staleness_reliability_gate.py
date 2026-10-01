"""Split-half reliability gate for individual baseline-staleness differences."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "outputs" / "revision_2026_09_29"
INPUT = OUT / "equal_weight_mean_baseline_predictions.parquet"
PROTOCOL = ROOT / "direction_reset" / "78_individual_staleness_reliability_protocol_2026-09-30.md"
SEED = 20260918
BOOTSTRAPS = 2000
PRIMARY = {("Dejonckheere", "sad"), ("Dejonckheere", "stressed"),
           ("CES", "phq2"), ("Marian", "depressed")}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def slope(x: np.ndarray, y: np.ndarray) -> float:
    centered = x - x.mean()
    denominator = float(np.square(centered).sum())
    if denominator <= 0:
        return np.nan
    return float(np.dot(centered, y) / denominator)


def split_person_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, outcome, participant), group in frame.groupby(
            ["dataset", "outcome", "participant"], sort=False):
        group = group.sort_values("baseline_age", kind="stable").reset_index(drop=True)
        if len(group) < 10 or group.baseline_age.nunique() < 2:
            continue
        halves = {"A": group.iloc[::2], "B": group.iloc[1::2]}
        if any(len(half) < 5 or half.baseline_age.nunique() < 2 for half in halves.values()):
            continue
        if set(halves["A"].index) & set(halves["B"].index):
            raise AssertionError("Split halves overlap")
        record = {"dataset": dataset, "outcome": outcome,
                  "participant": participant, "targets": len(group),
                  "targets_A": len(halves["A"]), "targets_B": len(halves["B"])}
        for name, half in halves.items():
            record[f"mean_benefit_{name}"] = float(half.benefit_mean.mean())
            record[f"slope_benefit_{name}"] = slope(
                half.log_age_scaled.to_numpy(dtype=float),
                half.benefit_mean.to_numpy(dtype=float))
        rows.append(record)
    result = pd.DataFrame(rows)
    if result.empty or result.filter(like="slope_").isna().any().any():
        raise AssertionError("Invalid split-half slope")
    if not (result.targets_A + result.targets_B == result.targets).all():
        raise AssertionError("Split target counts do not reconcile")
    return result


def spearman_brown(correlation: float) -> float:
    if correlation <= -0.999999999:
        return np.nan
    return float(2 * correlation / (1 + correlation))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x, ddof=0) <= 0 or np.std(y, ddof=0) <= 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_sb(x: np.ndarray, y: np.ndarray,
                 rng: np.random.Generator) -> tuple[float, float, int]:
    draws = []
    for _ in range(BOOTSTRAPS):
        indices = rng.integers(0, len(x), len(x))
        correlation = pearson(x[indices], y[indices])
        if np.isfinite(correlation) and correlation > -0.999999999:
            draws.append(spearman_brown(correlation))
    if len(draws) < 1900:
        raise AssertionError(f"Too few valid bootstrap draws: {len(draws)}")
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975)), len(draws)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for (dataset, outcome), group in metrics.groupby(["dataset", "outcome"], sort=False):
        for quantity in ("slope_benefit", "mean_benefit"):
            x = group[f"{quantity}_A"].to_numpy(dtype=float)
            y = group[f"{quantity}_B"].to_numpy(dtype=float)
            pearson_r = pearson(x, y)
            pearson_sb = spearman_brown(pearson_r)
            low, high, valid = bootstrap_sb(x, y, rng)
            spearman_r = float(spearmanr(x, y).statistic)
            rows.append({
                "dataset": dataset, "outcome": outcome,
                "role": "primary" if (dataset, outcome) in PRIMARY else "secondary",
                "quantity": quantity, "participants": len(group),
                "targets": int(group.targets.sum()),
                "pearson_half_r": pearson_r, "pearson_sb": pearson_sb,
                "pearson_sb_ci_low": low, "pearson_sb_ci_high": high,
                "valid_bootstrap_draws": valid,
                "spearman_half_r": spearman_r,
                "spearman_sb": spearman_brown(spearman_r),
                "same_direction_fraction": float((np.sign(x) == np.sign(y)).mean()),
            })
    return pd.DataFrame(rows)


def classify(summary: pd.DataFrame) -> tuple[str, dict[str, bool]]:
    primary = summary.loc[(summary.role.eq("primary")) &
                          (summary.quantity.eq("slope_benefit"))].copy()
    passed = primary.pearson_sb.ge(.50) & primary.pearson_sb_ci_low.gt(.20)
    mapping = {f"{row.dataset}|{row.outcome}": bool(value)
               for row, value in zip(primary.itertuples(index=False), passed)}
    if passed.all():
        result = "confirmatory_personalization_allowed"
    elif passed.sum() >= 2 and primary.pearson_sb.gt(0).all():
        result = "exploratory_partial_pooling_only"
    else:
        result = "stop_individual_slope_prediction"
    return result, mapping


def self_test() -> None:
    x = np.arange(20, dtype=float)
    assert np.isclose(slope(x, 2 * x + 3), 2.0)
    assert np.isclose(spearman_brown(.5), 2 / 3)


def main() -> None:
    if not PROTOCOL.exists():
        raise RuntimeError("Frozen reliability protocol missing")
    self_test()
    frame = pd.read_parquet(INPUT)
    metrics = split_person_metrics(frame)
    summary = summarize(metrics)
    classification, primary_pass = classify(summary)
    metrics.to_parquet(OUT / "individual_staleness_split_half_metrics.parquet", index=False)
    summary.to_csv(OUT / "individual_staleness_reliability_summary.csv", index=False)
    manifest = {"status": "individual staleness split-half reliability gate",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "classification": classification, "primary_pass": primary_pass,
                "seed": SEED, "bootstrap_resamples": BOOTSTRAPS,
                "protocol_sha256": sha256(PROTOCOL), "script_sha256": sha256(Path(__file__)),
                "input_sha256": sha256(INPUT),
                "software": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "scipy": scipy.__version__},
                "checks": {"minimum_ten_targets": True,
                           "alternating_age_order_split": True,
                           "halves_disjoint": True, "half_target_counts_reconcile": True,
                           "participant_bootstrap": True}}
    (OUT / "individual_staleness_reliability_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nCLASSIFICATION:", classification)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
