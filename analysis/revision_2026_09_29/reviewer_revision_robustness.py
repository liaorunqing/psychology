"""Post-review absolute-error, comparator, scale, and reporting audit."""

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


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "outputs" / "revision_2026_09_29"
PROTOCOL = ROOT / "direction_reset" / "87_reviewer_revision_robustness_protocol_2026-10-01.md"
DEJON_PRED = OUT / "dynamic_baseline_challenge_predictions.parquet"
CROSS_PRED = OUT / "cross_dataset_equal_information_predictions.parquet"
DATA_ROOT = Path(os.environ.get("DPT_DATA_ROOT", ROOT / "data")).expanduser()
DEJON_RAW = DATA_ROOT / "dejonckheere_openesm" / "0012_dejonckheere_ts.tsv"
MARIAN_RAW = DATA_ROOT / "marian_openesm" / "0052_marian_ts.tsv"
CES_DEMOGRAPHICS = DATA_ROOT / "Demographics" / "demographics.csv"
BUDGET_SCRIPT = Path(__file__).with_name("report_calibration_budget_curve.py")
SEED = 20260918
BOOTSTRAPS = 2000
ALPHA = 2.0 / 9.0
METHODS = ("B8", "L8", "Last", "Cumulative", "EWM8")
EXPECTED = {("Dejonckheere", outcome): (100, 7902)
            for outcome in ("sad", "stressed", "happy", "relaxed", "angry")}
EXPECTED.update({("CES", "phq2"): (105, 4541),
                 ("Marian", "depressed"): (145, 7071),
                 ("Marian", "anhedonia"): (145, 7071)})
BOUNDS = {"Dejonckheere": (0.0, 100.0), "CES": (0.0, 6.0), "Marian": (1.0, 4.0)}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_common_predictions() -> pd.DataFrame:
    dejon = pd.read_parquet(DEJON_PRED).copy()
    dejon.insert(0, "dataset", "Dejonckheere")
    dejon["order"] = dejon.counter.astype(float)
    cross = pd.read_parquet(CROSS_PRED).copy()
    cross["order"] = cross.baseline_age.astype(float)
    keep = ["dataset", "outcome", "participant", "actual", "baseline_age",
            "mean_8", "last8_mean", "last8_last", "order"]
    frame = pd.concat([dejon[keep], cross[keep]], ignore_index=True)
    outputs: list[pd.DataFrame] = []
    for (dataset, outcome, participant), group in frame.groupby(
            ["dataset", "outcome", "participant"], sort=False):
        group = group.sort_values("order", kind="stable").copy()
        group["B8"] = group.mean_8
        group["L8"] = group.last8_mean
        group["Last"] = group.last8_last
        cumulative_total = 8.0 * float(group.mean_8.iloc[0])
        ewm = float(group.mean_8.iloc[0])
        cumulative_predictions, ewm_predictions = [], []
        for index, actual in enumerate(group.actual.to_numpy(dtype=float)):
            cumulative_predictions.append(cumulative_total / (8.0 + index))
            ewm_predictions.append(ewm)
            cumulative_total += actual
            ewm = ALPHA * actual + (1.0 - ALPHA) * ewm
        group["Cumulative"] = cumulative_predictions
        group["EWM8"] = ewm_predictions
        outputs.append(group)
    result = pd.concat(outputs, ignore_index=True)
    for key, (people, targets) in EXPECTED.items():
        group = result.loc[result.dataset.eq(key[0]) & result.outcome.eq(key[1])]
        if group.participant.nunique() != people or len(group) != targets:
            raise AssertionError(f"Target sample changed: {key}")
        first = group.sort_values(["participant", "order"]).groupby("participant").head(1)
        if not np.allclose(first.Cumulative, first.B8) or not np.allclose(first.EWM8, first.B8):
            raise AssertionError(f"Sequential comparator initialization failed: {key}")
    if result[list(METHODS)].isna().any().any():
        raise AssertionError("Missing comparator prediction")
    return result


def participant_metric_table(group: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scale_range = BOUNDS[str(group.dataset.iloc[0])][1] - BOUNDS[str(group.dataset.iloc[0])][0]
    for method in METHODS:
        errors = group.actual.to_numpy(dtype=float) - group[method].to_numpy(dtype=float)
        temp = pd.DataFrame({"participant": group.participant.astype(str),
                             "ae": np.abs(errors), "se": np.square(errors)})
        by_person = temp.groupby("participant", sort=False).mean()
        by_person["method"] = method
        by_person["nmae"] = by_person.ae / scale_range
        rows.append(by_person.reset_index())
    return pd.concat(rows, ignore_index=True)


def summarize_comparators(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    metric_rows, contrast_rows = [], []
    for (dataset, outcome), group in frame.groupby(["dataset", "outcome"], sort=False):
        per = participant_metric_table(group)
        wide_ae = per.pivot(index="participant", columns="method", values="ae")
        for method in METHODS:
            values = per.loc[per.method.eq(method)].set_index("participant")
            array = values[["ae", "se", "nmae"]].to_numpy(dtype=float)
            draws = np.empty((BOOTSTRAPS, 3), dtype=float)
            for iteration in range(BOOTSTRAPS):
                sampled = array[rng.integers(0, len(array), len(array))].mean(axis=0)
                draws[iteration] = [sampled[0], np.sqrt(sampled[1]), sampled[2]]
            observed = array.mean(axis=0)
            metric_rows.append({
                "dataset": dataset, "outcome": outcome, "method": method,
                "participants": len(array), "targets": len(group),
                "mae": observed[0], "mae_ci_low": np.quantile(draws[:, 0], .025),
                "mae_ci_high": np.quantile(draws[:, 0], .975),
                "rmse": np.sqrt(observed[1]), "rmse_ci_low": np.quantile(draws[:, 1], .025),
                "rmse_ci_high": np.quantile(draws[:, 1], .975),
                "normalized_mae": observed[2],
                "normalized_mae_ci_low": np.quantile(draws[:, 2], .025),
                "normalized_mae_ci_high": np.quantile(draws[:, 2], .975),
            })
        for comparator in ("B8", "Last", "Cumulative", "EWM8"):
            values = wide_ae[["L8", comparator]].to_numpy(dtype=float)
            relative = np.empty(BOOTSTRAPS, dtype=float)
            for iteration in range(BOOTSTRAPS):
                sampled = values[rng.integers(0, len(values), len(values))].mean(axis=0)
                relative[iteration] = 100.0 * (sampled[0] / sampled[1] - 1.0)
            means = values.mean(axis=0)
            contrast_rows.append({
                "dataset": dataset, "outcome": outcome, "comparator": comparator,
                "participants": len(values), "targets": len(group),
                "l8_relative_mae_percent": 100.0 * (means[0] / means[1] - 1.0),
                "ci_low": np.quantile(relative, .025), "ci_high": np.quantile(relative, .975),
            })
    return pd.DataFrame(metric_rows), pd.DataFrame(contrast_rows)


def ordinal_sensitivity(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED + 1)
    summaries, contrasts = [], []
    bounded = frame.loc[frame.dataset.isin(["CES", "Marian"])].copy()
    for (dataset, outcome), group in bounded.groupby(["dataset", "outcome"], sort=False):
        lower, upper = BOUNDS[dataset]
        person_method: dict[str, pd.DataFrame] = {}
        for method in METHODS:
            rounded = np.rint(np.clip(group[method].to_numpy(dtype=float), lower, upper))
            actual = group.actual.to_numpy(dtype=float)
            temp = pd.DataFrame({
                "participant": group.participant.astype(str),
                "exact": (rounded == actual).astype(float),
                "within_one": (np.abs(rounded - actual) <= 1.0).astype(float),
                "ordinal_mae": np.abs(rounded - actual),
            }).groupby("participant", sort=False).mean()
            person_method[method] = temp
            array = temp[["exact", "within_one", "ordinal_mae"]].to_numpy(dtype=float)
            draws = np.empty((BOOTSTRAPS, 3), dtype=float)
            for iteration in range(BOOTSTRAPS):
                draws[iteration] = array[rng.integers(0, len(array), len(array))].mean(axis=0)
            observed = array.mean(axis=0)
            summaries.append({
                "dataset": dataset, "outcome": outcome, "method": method,
                "participants": len(array), "targets": len(group),
                "exact_accuracy": observed[0], "exact_ci_low": np.quantile(draws[:, 0], .025),
                "exact_ci_high": np.quantile(draws[:, 0], .975),
                "within_one_accuracy": observed[1],
                "within_one_ci_low": np.quantile(draws[:, 1], .025),
                "within_one_ci_high": np.quantile(draws[:, 1], .975),
                "rounded_ordinal_mae": observed[2],
                "ordinal_mae_ci_low": np.quantile(draws[:, 2], .025),
                "ordinal_mae_ci_high": np.quantile(draws[:, 2], .975),
            })
        joined = person_method["L8"].join(person_method["B8"], lsuffix="_L8", rsuffix="_B8")
        array = joined.to_numpy(dtype=float)
        draws = np.empty((BOOTSTRAPS, 3), dtype=float)
        for iteration in range(BOOTSTRAPS):
            sampled = array[rng.integers(0, len(array), len(array))].mean(axis=0)
            draws[iteration] = [sampled[0] - sampled[3], sampled[1] - sampled[4],
                                sampled[2] - sampled[5]]
        observed = array.mean(axis=0)
        differences = np.array([observed[0] - observed[3], observed[1] - observed[4],
                                observed[2] - observed[5]])
        contrasts.append({
            "dataset": dataset, "outcome": outcome, "participants": len(array),
            "exact_accuracy_difference_L8_minus_B8": differences[0],
            "exact_ci_low": np.quantile(draws[:, 0], .025),
            "exact_ci_high": np.quantile(draws[:, 0], .975),
            "within_one_difference_L8_minus_B8": differences[1],
            "within_one_ci_low": np.quantile(draws[:, 1], .025),
            "within_one_ci_high": np.quantile(draws[:, 1], .975),
            "ordinal_mae_difference_L8_minus_B8": differences[2],
            "ordinal_mae_ci_low": np.quantile(draws[:, 2], .025),
            "ordinal_mae_ci_high": np.quantile(draws[:, 2], .975),
        })
    return pd.DataFrame(summaries), pd.DataFrame(contrasts)


def score_distributions(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, outcome), group in frame.groupby(["dataset", "outcome"], sort=False):
        lower, upper = BOUNDS[dataset]
        values = group.actual.to_numpy(dtype=float)
        rows.append({
            "dataset": dataset, "outcome": outcome,
            "participants": group.participant.nunique(), "targets": len(group),
            "mean": values.mean(), "sd": values.std(ddof=1), "minimum": values.min(),
            "maximum": values.max(), "floor_fraction": np.mean(values == lower),
            "ceiling_fraction": np.mean(values == upper),
        })
    return pd.DataFrame(rows)


def reporting_audit(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    budget = load_module(BUDGET_SCRIPT, "budget_for_reviewer_revision")
    ces = budget.PROFILE.BENCHMARK.build_wide_table()[["uid", "assessment_date", "phq2"]].copy()
    ces.uid = ces.uid.astype(str)
    ces_complete = ces.dropna(subset=["phq2"])
    ces_counts = ces_complete.groupby("uid").size()
    dejon = pd.read_csv(DEJON_RAW, sep="\t")
    marian = pd.read_csv(MARIAN_RAW, sep="\t")
    rows = [{
        "dataset": "Dejonckheere", "source_participants": dejon.id.nunique(),
        "source_opportunities_or_rows": len(dejon), "complete_analysis_records": int(dejon.sad.notna().sum()),
        "eligible_participants": 100, "evaluated_targets": 7902,
        "eligibility_rule": "at least 9 complete outcome reports",
    }, {
        "dataset": "CES", "source_participants": ces.uid.nunique(),
        "source_opportunities_or_rows": len(ces), "complete_analysis_records": len(ces_complete),
        "eligible_participants": int((ces_counts >= 9).sum()), "evaluated_targets": 4541,
        "eligibility_rule": "at least 9 complete PHQ-2 assessments in constructed panel",
    }, {
        "dataset": "Marian", "source_participants": marian.id.nunique(),
        "source_opportunities_or_rows": len(marian),
        "complete_analysis_records": int((marian.status == 1).sum()),
        "eligible_participants": 145, "evaluated_targets": 7071,
        "eligibility_rule": "at least 9 complete outcome reports",
    }]
    eligible_ids = set(frame.loc[frame.dataset.eq("CES"), "participant"].astype(str))
    demographics = pd.read_csv(CES_DEMOGRAPHICS, dtype={"uid": str})
    demographics = (pd.DataFrame({"uid": sorted(eligible_ids)})
                    .merge(demographics, on="uid", how="left", validate="one_to_one"))
    if demographics.uid.nunique() != len(eligible_ids):
        raise AssertionError("CES demographic merge changed the analytic participant set")
    demo_rows = []
    for variable in ("gender", "race"):
        counts = demographics[variable].fillna("Missing").value_counts(dropna=False)
        for level, count in counts.items():
            demo_rows.append({"dataset": "CES", "variable": variable, "level": str(level),
                              "count": int(count), "percent": 100.0 * count / len(demographics)})
    return pd.DataFrame(rows), pd.DataFrame(demo_rows)


def main() -> None:
    if not PROTOCOL.exists():
        raise FileNotFoundError("Reviewer-revision protocol is missing")
    OUT.mkdir(parents=True, exist_ok=True)
    frame = build_common_predictions()
    metrics, contrasts = summarize_comparators(frame)
    ordinal, ordinal_contrasts = ordinal_sensitivity(frame)
    distributions = score_distributions(frame)
    flow, demographics = reporting_audit(frame)
    paths = {
        "reviewer_comparator_predictions.parquet": frame,
        "reviewer_absolute_comparator_metrics.csv": metrics,
        "reviewer_l8_comparator_contrasts.csv": contrasts,
        "reviewer_ordinal_sensitivity.csv": ordinal,
        "reviewer_ordinal_contrasts.csv": ordinal_contrasts,
        "reviewer_score_distributions.csv": distributions,
        "reviewer_cohort_flow.csv": flow,
        "reviewer_ces_demographics.csv": demographics,
    }
    for name, table in paths.items():
        path = OUT / name
        if path.suffix == ".parquet":
            table.to_parquet(path, index=False)
        else:
            table.to_csv(path, index=False)
    manifest = {
        "status": "post-review descriptive robustness analysis",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED, "bootstrap_resamples": BOOTSTRAPS,
        "ewm_alpha": ALPHA,
        "protocol_sha256": sha256(PROTOCOL), "script_sha256": sha256(Path(__file__)),
        "inputs": {str(path.name): sha256(path) for path in (DEJON_PRED, CROSS_PRED)},
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__},
        "checks": {"strictly_prior_comparators": True, "participant_bootstrap": True,
                   "target_samples_unchanged": True, "fixed_ewm_alpha": True,
                   "ordinal_analysis_post_review": True},
    }
    (OUT / "reviewer_revision_robustness_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nABSOLUTE METRICS\n", metrics.to_string(index=False))
    print("\nL8 CONTRASTS\n", contrasts.to_string(index=False))
    print("\nORDINAL CONTRASTS\n", ordinal_contrasts.to_string(index=False))
    print("\nFLOW\n", flow.to_string(index=False))


if __name__ == "__main__":
    main()

