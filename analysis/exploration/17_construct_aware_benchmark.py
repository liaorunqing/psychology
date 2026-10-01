from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT / "analysis" / "outputs" / "exploration"
SEED = 20260918
OUTCOMES = ["stress", "gad2", "phq2", "low_self_esteem", "perceived_isolation"]
DISTRESS_OUTCOMES = ["stress", "gad2", "phq2", "low_self_esteem"]
SENSOR_FEATURES = [
    "w7__location_diversity_level",
    "w7__study_place_exposure",
    "w7__social_place_exposure",
    "w7__home_exposure",
    "w7__mobility_level",
    "w7__phone_use_level",
    "w7__stillness_level",
    "w7__night_phone_share",
    "w7__location_diversity_variability",
    "w7__mobility_variability",
    "w7__phone_use_variability",
    "w7__stillness_variability",
    "w7__sleep_duration_variability",
]
HISTORY_FEATURES = [f"previous__{name}" for name in OUTCOMES] + ["gap_days"]


@dataclass(frozen=True)
class Fold:
    validation: str
    fold: int
    train_index: np.ndarray
    test_index: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_wide_table() -> pd.DataFrame:
    source = OUTPUT / "ces_development_multiscale_features.parquet"
    long = pd.read_parquet(source)
    split = pd.read_parquet(OUTPUT / "ces_exploration_participant_split.parquet")
    development_ids = set(split.loc[split["role"].eq("development"), "uid"].astype(str))
    held_out_ids = set(split.loc[split["role"].eq("held_out"), "uid"].astype(str))
    long["uid"] = long["uid"].astype(str)
    if not set(long["uid"]).issubset(development_ids) or long["uid"].isin(held_out_ids).any():
        raise RuntimeError("Held-out participants entered the construct benchmark.")

    outcomes = long.pivot(
        index=["uid", "assessment_date"], columns="outcome", values="current_outcome"
    ).reset_index()
    outcomes["low_self_esteem"] = 6.0 - outcomes.pop("self_esteem")
    features = long.loc[long["outcome"].eq("stress"), [
        "uid", "assessment_date", "w7__is_ios", *SENSOR_FEATURES,
    ]].copy()
    data = outcomes.merge(features, on=["uid", "assessment_date"], how="inner")
    data = data.sort_values(["uid", "assessment_date"]).reset_index(drop=True)
    for outcome in OUTCOMES:
        data[f"previous__{outcome}"] = data.groupby("uid")[outcome].shift(1)
    data["previous_date"] = data.groupby("uid")["assessment_date"].shift(1)
    data["gap_days"] = (data["assessment_date"] - data["previous_date"]).dt.days
    data["calendar_year"] = data["assessment_date"].dt.year.astype(str)
    data["is_ios"] = data["w7__is_ios"].round().astype("Int64").astype(str)
    data = data[
        data[OUTCOMES].notna().all(axis=1)
        & data[HISTORY_FEATURES].notna().all(axis=1)
        & data["gap_days"].between(14, 30)
    ].copy()
    if data["uid"].isin(held_out_ids).any():
        raise RuntimeError("Held-out participant leakage detected after filtering.")
    return data.reset_index(drop=True)


def make_folds(data: pd.DataFrame) -> list[Fold]:
    folds: list[Fold] = []
    random_cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for number, (train, test) in enumerate(random_cv.split(data), start=1):
        folds.append(Fold("random_record", number, train, test))

    grouped_cv = GroupKFold(n_splits=5)
    for number, (train, test) in enumerate(grouped_cv.split(data, groups=data["uid"]), start=1):
        folds.append(Fold("participant_heldout", number, train, test))

    train_indices: list[int] = []
    test_indices: list[int] = []
    for _, group in data.groupby("uid"):
        ordered = group.sort_values("assessment_date")
        cut = max(1, int(np.floor(len(ordered) * 0.70)))
        if len(ordered) - cut < 2:
            continue
        first_test_date = ordered.iloc[cut]["assessment_date"]
        train_indices.extend(
            ordered.loc[ordered["assessment_date"] <= first_test_date - pd.Timedelta(14, "D")].index
        )
        test_indices.extend(ordered.iloc[cut:].index)
    folds.append(Fold(
        "blocked_time_known_person", 1, np.asarray(train_indices, dtype=int), np.asarray(test_indices, dtype=int)
    ))
    return folds


def validate_fold(data: pd.DataFrame, fold: Fold) -> dict[str, int | str]:
    train = data.iloc[fold.train_index]
    test = data.iloc[fold.test_index]
    overlap = set(train["uid"]) & set(test["uid"])
    if fold.validation == "participant_heldout" and overlap:
        raise AssertionError("Participant-held-out fold contains overlapping IDs.")
    if fold.validation == "blocked_time_known_person":
        if not overlap:
            raise AssertionError("Blocked-time fold unexpectedly has no recurring participants.")
        for uid in overlap:
            latest_train = train.loc[train["uid"].eq(uid), "assessment_date"].max()
            earliest_test = test.loc[test["uid"].eq(uid), "assessment_date"].min()
            if (earliest_test - latest_train).days < 14:
                raise AssertionError(f"Blocked-time purge gap failed for {uid}.")
    return {
        "validation": fold.validation,
        "fold": fold.fold,
        "train_records": len(train),
        "test_records": len(test),
        "train_participants": train["uid"].nunique(),
        "test_participants": test["uid"].nunique(),
        "participant_overlap": len(overlap),
    }


def x_pipeline(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric:
        transformers.append((
            "numeric",
            Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "categorical", OneHotEncoder(handle_unknown="ignore"), categorical
        ))
    return ColumnTransformer(transformers)


def safe_correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    if len(actual) < 3 or np.std(actual) == 0 or np.std(predicted) == 0:
        return float("nan")
    return float(np.corrcoef(actual, predicted)[0, 1])


def descriptive_loading_stability(data: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler().fit(data[DISTRESS_OUTCOMES])
    overall_values = scaler.transform(data[DISTRESS_OUTCOMES])
    overall_pca = PCA(n_components=1, random_state=SEED).fit(overall_values)
    reference = overall_pca.components_[0].copy()
    if reference.sum() < 0:
        reference *= -1
    rows: list[dict[str, float | int | str]] = []
    groupings = [("overall", pd.Series("all", index=data.index))]
    groupings.extend([
        ("calendar_year", data["calendar_year"]),
        ("platform", data["is_ios"]),
    ])
    for grouping, labels in groupings:
        for label in sorted(labels.dropna().unique()):
            mask = labels.eq(label)
            if mask.sum() < 200:
                continue
            values = StandardScaler().fit_transform(data.loc[mask, DISTRESS_OUTCOMES])
            pca = PCA(n_components=1, random_state=SEED).fit(values)
            component = pca.components_[0].copy()
            if np.dot(component, reference) < 0:
                component *= -1
            congruence = float(np.dot(component, reference) / (
                np.linalg.norm(component) * np.linalg.norm(reference)
            ))
            for outcome, loading in zip(DISTRESS_OUTCOMES, component):
                rows.append({
                    "grouping": grouping,
                    "group": str(label),
                    "records": int(mask.sum()),
                    "participants": int(data.loc[mask, "uid"].nunique()),
                    "outcome": outcome,
                    "loading": float(loading),
                    "congruence_with_overall": congruence,
                    "explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
                })
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = build_wide_table()
    folds = make_folds(data)
    split_rows = [validate_fold(data, fold) for fold in folds]

    model_specs = {
        "intercept": ([], []),
        "participant_identity": ([], ["uid"]),
        "history": (HISTORY_FEATURES, ["calendar_year", "is_ios"]),
        "sensing": (SENSOR_FEATURES, ["calendar_year", "is_ios"]),
        "history_plus_sensing": (HISTORY_FEATURES + SENSOR_FEATURES, ["calendar_year", "is_ios"]),
        "history_sensing_identity": (
            HISTORY_FEATURES + SENSOR_FEATURES, ["calendar_year", "is_ios", "uid"]
        ),
    }

    predictions: list[pd.DataFrame] = []
    loading_rows: list[dict[str, float | int | str]] = []
    for fold in folds:
        train = data.iloc[fold.train_index]
        test = data.iloc[fold.test_index]
        y_scaler = StandardScaler().fit(train[OUTCOMES])
        train_y_z = y_scaler.transform(train[OUTCOMES])
        test_y_z = y_scaler.transform(test[OUTCOMES])
        distress_indices = [OUTCOMES.index(name) for name in DISTRESS_OUTCOMES]
        pca = PCA(n_components=1, random_state=SEED).fit(train_y_z[:, distress_indices])
        component = pca.components_[0].copy()
        if component.sum() < 0:
            component *= -1
        train_factor = train_y_z[:, distress_indices] @ component
        factor_sd = float(np.std(train_factor, ddof=1))
        actual_factor = (test_y_z[:, distress_indices] @ component) / factor_sd
        for outcome, loading in zip(DISTRESS_OUTCOMES, component):
            loading_rows.append({
                "validation": fold.validation,
                "fold": fold.fold,
                "outcome": outcome,
                "loading": float(loading),
                "explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
            })

        for model_name, (numeric, categorical) in model_specs.items():
            if model_name == "intercept":
                model = DummyRegressor(strategy="mean")
                model.fit(np.zeros((len(train), 1)), train_y_z)
                predicted_y_z = model.predict(np.zeros((len(test), 1)))
            else:
                processor = x_pipeline(numeric, categorical)
                model = Pipeline([("processor", processor), ("model", Ridge(alpha=1.0))])
                columns = numeric + categorical
                model.fit(train[columns], train_y_z)
                predicted_y_z = model.predict(test[columns])
            predicted_factor = (predicted_y_z[:, distress_indices] @ component) / factor_sd
            block = test[["uid", "assessment_date"]].copy()
            block["validation"] = fold.validation
            block["fold"] = fold.fold
            block["model"] = model_name
            block["actual__common_distress"] = actual_factor
            block["predicted__common_distress"] = predicted_factor
            for index, outcome in enumerate(OUTCOMES):
                block[f"actual__{outcome}"] = test_y_z[:, index]
                block[f"predicted__{outcome}"] = predicted_y_z[:, index]
            predictions.append(block)

    prediction_table = pd.concat(predictions, ignore_index=True)
    metric_rows: list[dict[str, float | int | str]] = []
    targets = ["common_distress", *OUTCOMES]
    for (validation, model), group in prediction_table.groupby(["validation", "model"]):
        for target in targets:
            actual = group[f"actual__{target}"].to_numpy()
            predicted = group[f"predicted__{target}"].to_numpy()
            metric_rows.append({
                "validation": validation,
                "model": model,
                "target": target,
                "records": len(group),
                "participants": group["uid"].nunique(),
                "mae_sd": mean_absolute_error(actual, predicted),
                "rmse_sd": float(np.sqrt(mean_squared_error(actual, predicted))),
                "r2": r2_score(actual, predicted),
                "correlation": safe_correlation(actual, predicted),
            })
    metrics = pd.DataFrame(metric_rows)

    bootstrap_rows: list[dict[str, float | int | str]] = []
    rng = np.random.default_rng(SEED)
    comparisons = ["sensing", "history_plus_sensing", "history_sensing_identity"]
    for validation in prediction_table["validation"].unique():
        subset = prediction_table[prediction_table["validation"].eq(validation)]
        for target in targets:
            actual = subset[subset["model"].eq("history")][
                ["uid", "assessment_date", "fold", f"actual__{target}", f"predicted__{target}"]
            ].rename(columns={f"predicted__{target}": "prediction_history"})
            for challenger in comparisons:
                other = subset[subset["model"].eq(challenger)][
                    ["uid", "assessment_date", "fold", f"predicted__{target}"]
                ].rename(columns={f"predicted__{target}": "prediction_challenger"})
                paired = actual.merge(other, on=["uid", "assessment_date", "fold"], how="inner")
                truth = f"actual__{target}"
                paired["sq_history"] = (paired[truth] - paired["prediction_history"]) ** 2
                paired["sq_challenger"] = (paired[truth] - paired["prediction_challenger"]) ** 2
                by_person = paired.groupby("uid")[["sq_history", "sq_challenger"]].mean()
                observed = float(
                    np.sqrt(by_person["sq_challenger"].mean())
                    - np.sqrt(by_person["sq_history"].mean())
                )
                draws = np.empty(2000)
                for draw in range(2000):
                    sample = by_person.iloc[rng.integers(0, len(by_person), len(by_person))]
                    draws[draw] = np.sqrt(sample["sq_challenger"].mean()) - np.sqrt(sample["sq_history"].mean())
                bootstrap_rows.append({
                    "validation": validation,
                    "target": target,
                    "reference": "history",
                    "challenger": challenger,
                    "participants": len(by_person),
                    "delta_rmse_sd": observed,
                    "ci_lower": float(np.quantile(draws, 0.025)),
                    "ci_upper": float(np.quantile(draws, 0.975)),
                })

    bootstrap = pd.DataFrame(bootstrap_rows)
    loadings = pd.DataFrame(loading_rows)
    split_audit = pd.DataFrame(split_rows)
    loading_stability = descriptive_loading_stability(data)
    prediction_table.to_parquet(OUTPUT / "ces_construct_benchmark_predictions.parquet", index=False)
    metrics.to_csv(OUTPUT / "ces_construct_benchmark_metrics.csv", index=False)
    bootstrap.to_csv(OUTPUT / "ces_construct_benchmark_bootstrap.csv", index=False)
    loadings.to_csv(OUTPUT / "ces_common_distress_loadings.csv", index=False)
    loading_stability.to_csv(OUTPUT / "ces_common_distress_loading_stability.csv", index=False)
    split_audit.to_csv(OUTPUT / "ces_construct_benchmark_split_audit.csv", index=False)

    source = OUTPUT / "ces_development_multiscale_features.parquet"
    manifest = {
        "stage": "development_only_construct_aware_benchmark",
        "seed": SEED,
        "source_sha256": sha256(source),
        "records": len(data),
        "participants": data["uid"].nunique(),
        "outcomes": OUTCOMES,
        "common_distress_definition": "Fold-local first principal component of standardized stress, GAD-2, PHQ-2, and reversed self-esteem; provisional composite, not a validated latent diagnosis. Perceived isolation remains a separate outcome.",
        "window": "preceding days -7 through -1",
        "eligible_gap_days": [14, 30],
        "models": list(model_specs),
        "validation": ["random record 5-fold", "participant-held-out 5-fold", "70/30 blocked time with >=14-day purge"],
        "preprocessing": "All imputation, feature scaling, outcome scaling, and PCA loadings were fit within each training fold.",
        "bootstrap": "2,000 participant-level resamples; paired RMSE difference versus history model.",
        "guardrail": "Only development participants were analyzed; the held-out half was not accessed.",
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (OUTPUT / "construct_aware_benchmark_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("SPLIT AUDIT")
    print(split_audit.to_string(index=False))
    print("\nCOMMON DISTRESS METRICS")
    print(metrics[metrics["target"].eq("common_distress")].to_string(index=False))
    print("\nSENSOR INCREMENT VS HISTORY")
    print(bootstrap[
        bootstrap["challenger"].eq("history_plus_sensing")
    ].to_string(index=False))


if __name__ == "__main__":
    main()
