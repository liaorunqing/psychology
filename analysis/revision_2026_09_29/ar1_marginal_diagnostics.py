"""Participant-balanced observed-versus-simulated AR(1) model checks."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "outputs" / "revision_2026_09_29"
PROTOCOL = ROOT / "direction_reset" / "92_ar1_marginal_diagnostics_protocol_2026-10-07.md"
SEED = 20260918
DRAWS = 200
DEVELOPMENT = Path(__file__).with_name("external_peer_review_robustness.py")
CORONA = Path(__file__).with_name("corona_health_external_extension.py")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def metrics(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Return mean, SD, floor, ceiling, lag1, outside, and integer-grid fractions."""
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=-1)
    sd = values.std(axis=-1, ddof=0)
    floor = np.mean(values == lower, axis=-1)
    ceiling = np.mean(values == upper, axis=-1)
    x = values[..., :-1]
    y = values[..., 1:]
    xc = x - x.mean(axis=-1, keepdims=True)
    yc = y - y.mean(axis=-1, keepdims=True)
    denominator = np.sqrt(np.sum(xc * xc, axis=-1) * np.sum(yc * yc, axis=-1))
    lag1 = np.divide(np.sum(xc * yc, axis=-1), denominator,
                     out=np.full_like(denominator, np.nan, dtype=float),
                     where=denominator > 0)
    outside = np.mean((values < lower) | (values > upper), axis=-1)
    integer = np.mean(np.isclose(values, np.rint(values), atol=1e-10), axis=-1)
    return np.stack([mean, sd, floor, ceiling, lag1, outside, integer], axis=-1)


def summarize(label: str, outcome: str, people: list[np.ndarray], simulated: list[np.ndarray],
              lower: float, upper: float, generator: str) -> dict:
    observed = np.vstack([metrics(value, lower, upper) for value in people])
    simulated_metrics = np.vstack([
        np.nanmean(metrics(value, lower, upper), axis=0) for value in simulated
    ])
    names = ("mean", "within_sd", "floor_fraction", "ceiling_fraction",
             "lag1", "outside_scale_fraction", "integer_grid_fraction")
    row = {"dataset": label, "outcome": outcome, "participants": len(people),
           "simulation_draws_per_participant": DRAWS, "generator": generator}
    for index, name in enumerate(names):
        row[f"observed_{name}"] = float(np.nanmean(observed[:, index]))
        row[f"simulated_{name}"] = float(np.nanmean(simulated_metrics[:, index]))
    return row


def development_rows(module) -> list[dict]:
    series = module.load_module(module.BASE_SCRIPT, "ar1_diag_base").load_series()
    rows = []
    for key_index, (dataset, outcome) in enumerate(sorted({key[:2] for key in series})):
        people, _ = module.prepared_people(series, dataset, outcome)
        pooled = module.pooled_phi(people)
        rng = np.random.default_rng(SEED + 1000 * key_index)
        observed, simulated = [], []
        for person in people:
            values = person["values"]
            observed.append(values)
            simulated.append(module.simulate_person(
                values, DRAWS, pooled, module.BOUNDS[dataset], rng))
        rows.append(summarize(dataset, outcome, observed, simulated,
                              *module.BOUNDS[dataset], "Gaussian AR(1), clipped to scale bounds"))
    return rows


def corona_row(module) -> dict:
    raw = module.load_raw()
    records = module.scale_records(raw, "PHQ-9")
    specifications, _ = module.fit_ar1(records)
    observed_lookup = {
        str(participant): group.sort_values("date").score.to_numpy(dtype=float)
        for participant, group in records.groupby("participant", sort=False)
        if len(group) >= 9
    }
    rng = np.random.default_rng(SEED + 9000)
    observed, simulated = [], []
    for spec in specifications:
        values = observed_lookup[str(spec["participant"])]
        n, phi, mean, sd = spec["n"], spec["phi"], spec["mean"], spec["sd"]
        innovation_sd = sd * np.sqrt(max(1 - phi * phi, 1e-8))
        draws = np.empty((DRAWS, n), dtype=float)
        draws[:, 0] = rng.normal(mean, sd, DRAWS)
        for index in range(1, n):
            draws[:, index] = mean + phi * (draws[:, index - 1] - mean) + \
                rng.normal(0, innovation_sd, DRAWS)
        observed.append(values)
        simulated.append(draws)
    return summarize("Corona Health", "PHQ-9", observed, simulated, 0.0, 27.0,
                     "Gaussian AR(1), no scale-bound clipping")


def main() -> None:
    if not PROTOCOL.exists():
        raise FileNotFoundError(PROTOCOL)
    development = load_module(DEVELOPMENT, "ar1_diagnostic_development")
    corona = load_module(CORONA, "ar1_diagnostic_corona")
    table = pd.DataFrame(development_rows(development) + [corona_row(corona)])
    output = OUT / "ar1_marginal_diagnostics.csv"
    table.to_csv(output, index=False)
    manifest = {
        "status": "post-hoc AR(1) marginal model check",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "draws_per_participant": DRAWS,
        "protocol_sha256": sha256(PROTOCOL),
        "script_sha256": sha256(Path(__file__)),
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__},
        "participant_level_output_exported": False,
    }
    (OUT / "ar1_marginal_diagnostics_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
