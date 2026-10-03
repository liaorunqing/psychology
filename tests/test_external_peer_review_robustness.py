from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = (Path(__file__).resolve().parents[1] / "analysis" / "revision_2026_09_29" /
          "external_peer_review_robustness.py")
SPEC = importlib.util.spec_from_file_location("external_peer_review_robustness_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_rolling_mean_is_strictly_prior() -> None:
    values = np.arange(12, dtype=float)
    prediction = MODULE.rolling_mean(values)
    assert np.isclose(prediction[0], np.mean(np.arange(8)))
    changed = values.copy()
    changed[8:] = 1000
    assert np.isclose(MODULE.rolling_mean(changed)[0], prediction[0])


def test_recent_mode_ties_resolve_to_most_recent() -> None:
    values = np.asarray([1, 2, 1, 2, 3, 3, 4, 4], dtype=float)
    assert MODULE.mode_recent_first(values) == 4.0


def test_person_statistics_recovers_recent_constant_shift() -> None:
    values = np.concatenate([np.zeros(8), np.ones(16)])
    x = np.log1p(np.arange(1, len(values) - 7, dtype=float))
    delta, slope = MODULE.person_statistics(values, x)
    assert delta > 0
    assert np.isfinite(slope)


def test_stationary_simulation_is_seed_reproducible_and_bounded() -> None:
    values = np.asarray([1, 2, 1, 2, 1, 2, 1, 2, 1, 2], dtype=float)
    first = MODULE.simulate_person(values, 10, .2, (0.0, 3.0), np.random.default_rng(7))
    second = MODULE.simulate_person(values, 10, .2, (0.0, 3.0), np.random.default_rng(7))
    np.testing.assert_allclose(first, second)
    assert first.min() >= 0
    assert first.max() <= 3
