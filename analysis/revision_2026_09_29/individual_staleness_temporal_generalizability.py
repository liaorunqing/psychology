"""Early-block to late-block generalizability of individual staleness slopes."""

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
PROTOCOL = ROOT / "direction_reset" / "80_individual_staleness_temporal_generalizability_protocol_2026-09-30.md"
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


def block_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, outcome, participant), group in frame.groupby(
            ["dataset", "outcome", "participant"], sort=False):
        group = group.sort_values("baseline_age", kind="stable").reset_index(drop=True)
        if len(group) < 10 or group.baseline_age.nunique() < 2:
            continue
        cut = len(group) // 2
        blocks = {"early": group.iloc[:cut], "late": group.iloc[cut:]}
        if any(len(block) < 5 or block.baseline_age.nunique() < 2 for block in blocks.values()):
            continue
        if not blocks["early"].baseline_age.max() < blocks["late"].baseline_age.min():
            raise AssertionError("Temporal blocks are not separated")
        record = {"dataset": dataset, "outcome": outcome,
                  "participant": participant, "targets": len(group),
                  "targets_early": len(blocks["early"]),
                  "targets_late": len(blocks["late"]),
                  "early_age_max": float(blocks["early"].baseline_age.max()),
                  "late_age_min": float(blocks["late"].baseline_age.min())}
        for name, block in blocks.items():
            record[f"mean_benefit_{name}"] = float(block.benefit_mean.mean())
            record[f"slope_benefit_{name}"] = slope(
                block.log_age_scaled.to_numpy(dtype=float),
                block.benefit_mean.to_numpy(dtype=float))
        rows.append(record)
    result = pd.DataFrame(rows)
    if result.empty or result.filter(like="slope_").isna().any().any():
        raise AssertionError("Invalid block slope")
    if not (result.targets_early + result.targets_late == result.targets).all():
        raise AssertionError("Block targets do not reconcile")
    return result


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x, ddof=0) <= 0 or np.std(y, ddof=0) <= 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_correlation(x: np.ndarray, y: np.ndarray,
                          rng: np.random.Generator) -> tuple[float, float, int]:
    draws = []
    for _ in range(BOOTSTRAPS):
        indices = rng.integers(0, len(x), len(x))
        value = pearson(x[indices], y[indices])
        if np.isfinite(value):
            draws.append(value)
    if len(draws) < 1900:
        raise AssertionError("Too few valid bootstrap correlations")
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975)), len(draws)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for (dataset, outcome), group in metrics.groupby(["dataset", "outcome"], sort=False):
        for quantity in ("slope_benefit", "mean_benefit"):
            x = group[f"{quantity}_early"].to_numpy(dtype=float)
            y = group[f"{quantity}_late"].to_numpy(dtype=float)
            correlation = pearson(x, y)
            low, high, valid = bootstrap_correlation(x, y, rng)
            rows.append({"dataset": dataset, "outcome": outcome,
                         "role": "primary" if (dataset, outcome) in PRIMARY else "secondary",
                         "quantity": quantity, "participants": len(group),
                         "targets": int(group.targets.sum()),
                         "pearson_r": correlation, "ci_low": low, "ci_high": high,
                         "valid_bootstrap_draws": valid,
                         "spearman_r": float(spearmanr(x, y).statistic),
                         "same_direction_fraction": float((np.sign(x) == np.sign(y)).mean())})
    return pd.DataFrame(rows)


def classify(summary: pd.DataFrame) -> tuple[str, dict[str, bool]]:
    primary = summary.loc[(summary.role.eq("primary")) &
                          (summary.quantity.eq("slope_benefit"))].copy()
    passed = primary.pearson_r.gt(0) & primary.ci_low.gt(0)
    mapping = {f"{row.dataset}|{row.outcome}": bool(value)
               for row, value in zip(primary.itertuples(index=False), passed)}
    if passed.all():
        result = "confirmatory_personalization_allowed"
    elif passed.sum() >= 2 and primary.pearson_r.gt(0).all():
        result = "exploratory_partial_pooling_only"
    else:
        result = "stop_individual_slope_prediction"
    return result, mapping


def main() -> None:
    if not PROTOCOL.exists():
        raise RuntimeError("Frozen temporal generalizability protocol missing")
    frame = pd.read_parquet(INPUT)
    metrics = block_metrics(frame)
    summary = summarize(metrics)
    classification, primary_pass = classify(summary)
    metrics.to_parquet(OUT / "individual_staleness_temporal_block_metrics.parquet", index=False)
    summary.to_csv(OUT / "individual_staleness_temporal_generalizability_summary.csv", index=False)
    manifest = {"status": "individual staleness temporal block generalizability gate",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "classification": classification, "primary_pass": primary_pass,
                "seed": SEED, "bootstrap_resamples": BOOTSTRAPS,
                "protocol_sha256": sha256(PROTOCOL), "script_sha256": sha256(Path(__file__)),
                "input_sha256": sha256(INPUT),
                "software": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "scipy": scipy.__version__},
                "checks": {"minimum_ten_targets": True,
                           "temporally_separated_blocks": True,
                           "blocks_disjoint": True, "block_target_counts_reconcile": True,
                           "participant_bootstrap": True}}
    (OUT / "individual_staleness_temporal_generalizability_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nCLASSIFICATION:", classification)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
