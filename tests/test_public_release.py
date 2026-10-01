from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    path = ROOT / "analysis" / "revision_2026_09_29" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"release_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fold_assignment_is_reproducible_and_participant_disjoint() -> None:
    module = load("dejonckheere_baseline_age_confirmation")
    ids = pd.Series([str(value) for value in range(100)])
    first = module.make_fold_map(ids)
    second = module.make_fold_map(ids.sample(frac=1, random_state=4))
    assert first == second
    assert pd.Series(first).value_counts().sort_index().to_dict() == {
        1: 20, 2: 20, 3: 20, 4: 20, 5: 20}


def test_current_and_future_values_cannot_enter_history_features() -> None:
    module = load("dejonckheere_baseline_age_confirmation")
    module.leakage_self_test()


def test_equal_participant_weighting() -> None:
    module = load("dejonckheere_baseline_age_confirmation")
    ids = pd.Series(["a", "a", "a", "b", "b", "c"])
    weights = module.participant_weights(ids)
    totals = pd.Series(weights).groupby(ids).sum()
    assert np.allclose(totals, totals.iloc[0])


def test_sustained_deterioration_uses_only_prior_values() -> None:
    module = load("sustained_deterioration_signal_retention")
    offsets = np.arange(24, dtype=float)
    observed = module.prior_window_adaptation(offsets, window=8)
    changed = offsets.copy()
    changed[10:] += 1000
    counterfactual = module.prior_window_adaptation(changed, window=8)
    assert np.allclose(observed[:11], counterfactual[:11])


def test_baseline_age_strata_match_frozen_boundaries() -> None:
    module = load("baseline_age_mechanism")
    frame = pd.DataFrame({
        "dataset": ["CES"] * 5 + ["Marian"] * 5,
        "distance_8": [1, 90, 91, 730, 731, 1, 7, 8, 42, 43],
    })
    result = module.assign_strata(frame)
    assert result.age_stratum.tolist() == [
        "(0,90]", "(0,90]", "(90,365]", "(365,730]", ">730",
        "1-7", "1-7", "8-21", "22-42", ">=43",
    ]
