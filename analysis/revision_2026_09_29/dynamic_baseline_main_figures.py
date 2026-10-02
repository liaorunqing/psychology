from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "outputs" / "revision_2026_09_29"
FIGURES = ROOT / "figures" / "rendered"
SOURCE = ROOT / "figures" / "source_data"
SEED = 20260918
BOOTSTRAPS = 2000

COLORS = {
    "Dejonckheere": "#0072B2",
    "CES": "#D55E00",
    "Marian": "#009E73",
    "B8": "#4D4D4D",
    "L8": "#0072B2",
    "A50": "#CC79A7",
    "secondary": "#9A9A9A",
}
LABELS = {
    ("Dejonckheere", "sad"): "Sadness",
    ("Dejonckheere", "stressed"): "Stress",
    ("Dejonckheere", "happy"): "Happiness",
    ("Dejonckheere", "relaxed"): "Relaxation",
    ("Dejonckheere", "angry"): "Anger",
    ("CES", "phq2"): "PHQ-2",
    ("Marian", "depressed"): "Depressed mood",
    ("Marian", "anhedonia"): "Anhedonia",
}
PRIMARY = {
    ("Dejonckheere", "sad"),
    ("Dejonckheere", "stressed"),
    ("CES", "phq2"),
    ("Marian", "depressed"),
}


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.2,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {
        ".svg": {}, ".pdf": {}, ".tiff": {"dpi": 600}, ".png": {"dpi": 300}
    }.items():
        fig.savefig(FIGURES / f"{stem}{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.04, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom", ha="left")


def participant_bootstrap(frame: pd.DataFrame, value: str, rng: np.random.Generator) -> tuple[float, float, float]:
    values = frame.groupby("participant", sort=False)[value].mean().to_numpy(float)
    draw = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    samples = values[draw].mean(axis=1)
    return float(values.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def figure_1() -> None:
    fig = plt.figure(figsize=(7.2, 4.35), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.85], width_ratios=[1.35, 1.0])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    ax_a.set_xlim(0, 22); ax_a.set_ylim(-0.1, 3.0); ax_a.axis("off")
    y = np.array([1.9, 1.5, 1.75, 1.35, 1.2, 1.45, 1.65, 1.3,
                  1.1, 1.25, 1.55, 1.4, 1.75, 1.6, 1.35, 1.5, 1.85, 1.7, 1.45, 1.8])
    x = np.arange(1, 21)
    ax_a.plot(x, y, color="#B9B9B9", lw=1.0, zorder=1)
    ax_a.scatter(x[:8], y[:8], s=32, color=COLORS["B8"], zorder=3, label="First eight reports (B8)")
    ax_a.scatter(x[11:19], y[11:19], s=32, color=COLORS["L8"], zorder=3, label="Most recent eight reports (L8)")
    ax_a.scatter(x[19], y[19], s=54, facecolor="white", edgecolor="#D55E00", linewidth=1.5, zorder=4)
    ax_a.annotate("Target report", (x[19], y[19]), xytext=(19.2, 2.45), ha="center",
                  arrowprops=dict(arrowstyle="-|>", color="#D55E00", lw=0.9))
    ax_a.annotate("same report budget\nsame arithmetic mean", xy=(10, 0.45), ha="center", color="#333333")
    ax_a.plot([1, 8], [0.8, 0.8], color=COLORS["B8"], lw=2)
    ax_a.plot([12, 19], [0.8, 0.8], color=COLORS["L8"], lw=2)
    ax_a.text(4.5, 0.58, "frozen early history", ha="center", color=COLORS["B8"])
    ax_a.text(15.5, 0.58, "time-local history", ha="center", color=COLORS["L8"])
    ax_a.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["B8"], markeredgecolor=COLORS["B8"], label="B8"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["L8"], markeredgecolor=COLORS["L8"], label="L8"),
    ], loc="upper left", ncol=2, frameon=False)
    panel_label(ax_a, "a")

    ax_b.axis("off"); ax_b.set_xlim(0, 3); ax_b.set_ylim(0, 1)
    cards = [
        (0.02, COLORS["Dejonckheere"], "Dejonckheere", "n = 100 | 14 days\n7 prompts/day | five states"),
        (1.02, COLORS["CES"], "CES", "n = 105 | up to 4 years\nrepeated PHQ-2"),
        (2.02, COLORS["Marian"], "Marian", "n = 145 | 21 days\n3 prompts/day | two states"),
    ]
    for x0, color, title, body in cards:
        box = FancyBboxPatch((x0, 0.14), 0.9, 0.7, boxstyle="round,pad=0.02,rounding_size=0.03",
                             facecolor="white", edgecolor=color, linewidth=1.2)
        ax_b.add_patch(box)
        ax_b.text(x0 + 0.45, 0.68, title, ha="center", va="center", fontweight="bold", color=color)
        ax_b.text(x0 + 0.45, 0.42, body, ha="center", va="center", linespacing=1.45)
    ax_b.text(1.5, 0.02, "Contrasting measurement timescales; no pooling of raw outcome units", ha="center", color="#555555")
    panel_label(ax_b, "b")

    ax_c.axis("off"); ax_c.set_xlim(0, 1); ax_c.set_ylim(0, 1)
    stages = [
        (0.82, "Mechanism development", "CES + Marian", "#D9EAF5"),
        (0.52, "Association-blind confirmation", "Dejonckheere", "#CDEBDD"),
        (0.22, "Post-hoc challenges", "equal weights, shifts, stability, safety", "#F2D8E8"),
    ]
    for idx, (yc, title, body, fill) in enumerate(stages):
        box = FancyBboxPatch((0.08, yc - 0.09), 0.84, 0.18, boxstyle="round,pad=0.02,rounding_size=0.025",
                             facecolor=fill, edgecolor="#666666", linewidth=0.7)
        ax_c.add_patch(box)
        ax_c.text(0.5, yc + 0.025, title, ha="center", fontweight="bold")
        ax_c.text(0.5, yc - 0.045, body, ha="center", color="#444444", fontsize=6.3)
        if idx < 2:
            ax_c.add_patch(FancyArrowPatch((0.5, yc - 0.11), (0.5, yc - 0.19), arrowstyle="-|>",
                                           mutation_scale=8, color="#666666", lw=0.8))
    ax_c.text(0.5, 0.02, "Staged secondary analysis; not fully preregistered", ha="center", color="#7A3E00")
    panel_label(ax_c, "c")
    save_figure(fig, "Figure_1_study_design")


def figure_2() -> None:
    overall = pd.read_csv(OUT / "equal_weight_mean_baseline_overall_contrasts.csv")
    slopes = pd.read_csv(OUT / "equal_weight_mean_baseline_slope_summary.csv")
    predictions = pd.read_parquet(OUT / "equal_weight_mean_baseline_predictions.parquet")
    within = predictions.groupby(["dataset", "outcome", "participant"], sort=False).actual.var().groupby(level=[0, 1]).mean().pow(0.5)
    slope_plot = slopes.copy()
    slope_plot["within_sd"] = [within.loc[(r.dataset, r.outcome)] for r in slope_plot.itertuples()]
    for col in ["mean_slope", "ci_low", "ci_high"]:
        slope_plot[f"std_{col}"] = slope_plot[col] / slope_plot.within_sd
    order = list(overall.sort_values(["dataset", "outcome"])[["dataset", "outcome"]].itertuples(index=False, name=None))
    order = [x for x in order if x in PRIMARY] + [x for x in order if x not in PRIMARY]
    overall["order"] = overall.apply(lambda r: order.index((r.dataset, r.outcome)), axis=1)
    slope_plot["order"] = slope_plot.apply(lambda r: order.index((r.dataset, r.outcome)), axis=1)
    overall = overall.sort_values("order", ascending=False)
    slope_plot = slope_plot.sort_values("order", ascending=False)
    overall.to_csv(SOURCE / "Figure_2a_relative_mae.csv", index=False)
    slope_plot.to_csv(SOURCE / "Figure_2b_standardized_age_slope.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.3), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, data, estimate, low, high, xlabel in [
        (axes[0], overall, "relative_mae_percent", "mae_ci_low", "mae_ci_high", "L8 relative to B8: MAE change (%)"),
        (axes[1], slope_plot, "std_mean_slope", "std_ci_low", "std_ci_high", "Age slope (outcome SD per age SD)"),
    ]:
        ypos = np.arange(len(data))
        for pos, row in zip(ypos, data.itertuples()):
            color = COLORS[row.dataset]
            marker = "o" if (row.dataset, row.outcome) in PRIMARY else "s"
            ax.errorbar(getattr(row, estimate), pos,
                        xerr=[[getattr(row, estimate) - getattr(row, low)], [getattr(row, high) - getattr(row, estimate)]],
                        fmt=marker, color=color, ecolor=color, markersize=4.2, capsize=2, lw=1.0)
        ax.axvline(0, color="#555555", lw=0.8, ls="--")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color="#E5E5E5", lw=0.6)
    labels = [f"{r.dataset}: {LABELS[(r.dataset, r.outcome)]}" for r in overall.itertuples()]
    axes[0].set_yticks(np.arange(len(labels)), labels)
    axes[0].set_title("Prediction error")
    axes[1].set_title("Growth of update benefit with baseline age")
    axes[0].text(-0.03, -0.13, "← lower error with recent history", transform=axes[0].transAxes, color=COLORS["L8"])
    axes[1].text(0.45, -0.13, "larger recent-history advantage →", transform=axes[1].transAxes, color=COLORS["L8"])
    legend = [Line2D([0], [0], marker="o", color=COLORS[d], lw=0, label=d) for d in ["Dejonckheere", "CES", "Marian"]]
    legend += [Line2D([0], [0], marker="o", color="#555555", lw=0, label="Primary"),
               Line2D([0], [0], marker="s", color="#555555", lw=0, label="Secondary")]
    axes[1].legend(handles=legend, loc="lower right")
    panel_label(axes[0], "a"); panel_label(axes[1], "b")
    save_figure(fig, "Figure_2_cross_dataset_effects")


def relative_time_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    work = predictions.copy()
    within = work.groupby(["dataset", "outcome", "participant"], sort=False).actual.var().groupby(level=[0, 1]).mean().pow(0.5)
    for (dataset, outcome, participant), group in work.groupby(["dataset", "outcome", "participant"], sort=True):
        group = group.sort_values("baseline_age", kind="stable").copy()
        group["quartile"] = np.minimum(4, (np.arange(len(group)) * 4 // len(group)) + 1)
        group["benefit_std"] = group.benefit_mean / within.loc[(dataset, outcome)]
        for quartile, values in group.groupby("quartile"):
            rows.append({"dataset": dataset, "outcome": outcome, "participant": participant,
                         "quartile": int(quartile), "benefit_std": values.benefit_std.mean()})
    person = pd.DataFrame(rows)
    summary = []
    for (dataset, outcome, quartile), group in person.groupby(["dataset", "outcome", "quartile"]):
        estimate, low, high = participant_bootstrap(group, "benefit_std", rng)
        summary.append({"dataset": dataset, "outcome": outcome, "quartile": quartile,
                        "participants": group.participant.nunique(), "estimate": estimate,
                        "ci_low": low, "ci_high": high})
    return pd.DataFrame(summary)


def figure_3() -> None:
    predictions = pd.read_parquet(OUT / "equal_weight_mean_baseline_predictions.parquet")
    temporal = pd.read_csv(OUT / "individual_staleness_temporal_generalizability_summary.csv")
    age = relative_time_summary(predictions)
    age.to_csv(SOURCE / "Figure_3a_relative_time_benefit.csv", index=False)
    temporal.to_csv(SOURCE / "Figure_3b_temporal_generalizability.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.25), gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.34})
    ax = axes[0]
    primary_markers = {
        ("Dejonckheere", "sad"): "o",
        ("Dejonckheere", "stressed"): "^",
        ("CES", "phq2"): "D",
        ("Marian", "depressed"): "s",
    }
    for (dataset, outcome), group in age.groupby(["dataset", "outcome"], sort=False):
        primary = (dataset, outcome) in PRIMARY
        color = COLORS[dataset] if primary else COLORS["secondary"]
        alpha = 1.0 if primary else 0.6
        label = f"{dataset}: {LABELS[(dataset, outcome)]}"
        ax.plot(group.quartile, group.estimate, marker=primary_markers.get((dataset, outcome), "s"), ms=3.5,
                color=color, alpha=alpha, ls="-" if primary else "--", label=label)
        ax.fill_between(group.quartile, group.ci_low, group.ci_high, color=color, alpha=0.10 if primary else 0.05)
    ax.axhline(0, color="#555555", ls="--", lw=0.8)
    ax.set_xticks([1, 2, 3, 4], ["Q1\nearliest", "Q2", "Q3", "Q4\nlatest"])
    ax.set_xlabel("Participant-relative post-calibration time")
    ax.set_ylabel("Standardized update benefit (outcome SD)")
    ax.grid(axis="y", color="#E5E5E5", lw=0.6)
    primary_handles = [
        Line2D([0], [0], marker=primary_markers[key], color=COLORS[key[0]],
               label=f"{key[0]}: {LABELS[key]}")
        for key in [("Dejonckheere", "sad"), ("Dejonckheere", "stressed"),
                    ("CES", "phq2"), ("Marian", "depressed")]
    ]
    ax.legend(handles=primary_handles, loc="upper left", title="Primary outcomes")
    ax.text(0.03, 0.02, "Secondary outcomes shown in grey", transform=ax.transAxes, color="#666666", fontsize=6.2)
    panel_label(ax, "a")

    ax = axes[1]
    primary_temporal = temporal[(temporal.role == "primary")].copy()
    y_positions = np.arange(4)
    ordered = [("Dejonckheere", "sad"), ("Dejonckheere", "stressed"), ("CES", "phq2"), ("Marian", "depressed")]
    for offset, (quantity, marker, color, label) in enumerate([
        ("slope_benefit", "o", "#D55E00", "Age slope"),
        ("mean_benefit", "s", "#0072B2", "Mean update benefit"),
    ]):
        current = primary_temporal[primary_temporal.quantity == quantity].set_index(["dataset", "outcome"]).loc[ordered].reset_index()
        y = y_positions + (-0.10 if offset == 0 else 0.10)
        ax.errorbar(current.pearson_r, y,
                    xerr=[current.pearson_r - current.ci_low, current.ci_high - current.pearson_r],
                    fmt=marker, color=color, ecolor=color, capsize=2, markersize=4, label=label)
    ax.axvline(0, color="#555555", ls="--", lw=0.8)
    ax.set_yticks(y_positions, [LABELS[x] for x in ordered])
    ax.invert_yaxis()
    ax.set_xlim(-0.55, 1.0)
    ax.set_xlabel("Early--late Pearson correlation")
    ax.grid(axis="x", color="#E5E5E5", lw=0.6)
    ax.legend(loc="lower right")
    panel_label(ax, "b")
    save_figure(fig, "Figure_3_group_and_individual_patterns")


def figure_4() -> None:
    primary = pd.read_csv(OUT / "sustained_deterioration_primary_summary.csv")
    tradeoff = pd.read_csv(OUT / "sustained_deterioration_predictive_tradeoff.csv")
    primary = primary[primary.apply(lambda r: (r.dataset, r.outcome) in PRIMARY, axis=1)].copy()
    tradeoff = tradeoff[tradeoff.apply(lambda r: (r.dataset, r.outcome) in PRIMARY, axis=1)].copy()
    retention = primary[primary.metric == "retention"]
    detection = primary[primary.metric == "excess_detection"]
    joined = tradeoff.merge(retention[["dataset", "outcome", "method", "estimate"]],
                            on=["dataset", "outcome", "method"], validate="one_to_one")
    joined["mae_improvement_percent"] = -100 * joined.relative_mae_vs_b8
    joined = joined.rename(columns={"estimate": "signal_retention"})
    detection.to_csv(SOURCE / "Figure_4b_excess_alarm.csv", index=False)
    joined.to_csv(SOURCE / "Figure_4c_accuracy_retention_tradeoff.csv", index=False)

    fig = plt.figure(figsize=(7.2, 4.25), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[0.92, 1.2])
    ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[1, 0]); ax_c = fig.add_subplot(gs[:, 1])
    t = np.arange(24)
    shift = np.r_[np.zeros(8), np.arange(1, 9) / 8, np.ones(8)]
    adapt = np.zeros_like(shift, dtype=float)
    for idx in range(len(shift)):
        adapt[idx] = shift[max(0, idx - 8):idx].sum() / 8
    ax_a.plot(t, shift, color="#D55E00", label="Injected worsening", lw=1.6)
    ax_a.plot(t, np.zeros_like(t), color=COLORS["B8"], label="B8 adaptation")
    ax_a.plot(t, adapt, color=COLORS["L8"], label="L8 adaptation")
    ax_a.plot(t, 0.5 * adapt, color=COLORS["A50"], label="A50 adaptation")
    ax_a.axvspan(8, 15, color="#D55E00", alpha=0.07); ax_a.axvspan(16, 23, color="#0072B2", alpha=0.05)
    ax_a.set_ylabel("Change (SD)"); ax_a.set_xlabel("Report index")
    ax_a.set_title("Adaptation to imposed worsening", loc="left")
    ax_a.set_xticks([0, 8, 16, 23], ["pre", "onset", "plateau", "end"])
    ax_a.legend(ncol=2, loc="upper left", fontsize=5.8)
    panel_label(ax_a, "a")

    ax_b.axhline(0, color="#555555", lw=0.7, ls="--")
    for (dataset, outcome), group in detection.groupby(["dataset", "outcome"]):
        current = group.set_index("method").loc[["B8", "A50", "L8"]].reset_index()
        x = np.arange(3)
        ax_b.plot(x, 100 * current.estimate, color=COLORS[dataset], alpha=0.55, lw=0.9)
        for xpos, row in zip(x, current.itertuples()):
            ax_b.errorbar(xpos, 100 * row.estimate,
                          yerr=[[100 * (row.estimate - row.ci_low)], [100 * (row.ci_high - row.estimate)]],
                          fmt="o", color=COLORS[row.method], ecolor=COLORS[dataset], capsize=2, ms=3.5)
    ax_b.set_xticks([0, 1, 2], ["B8", "A50", "L8"])
    ax_b.set_ylabel("Excess alarm rate (percentage points)")
    ax_b.set_xlabel("Baseline method")
    ax_b.set_title("Alarm response", loc="left")
    ax_b.grid(axis="y", color="#E5E5E5", lw=0.6)
    panel_label(ax_b, "b")

    ax_c.axhline(1.0, color="#BBBBBB", lw=0.6); ax_c.axvline(0, color="#BBBBBB", lw=0.6)
    for (dataset, outcome), group in joined.groupby(["dataset", "outcome"]):
        current = group.set_index("method").loc[["B8", "A50", "L8"]].reset_index()
        for row in current.itertuples():
            ax_c.scatter(row.mae_improvement_percent, row.signal_retention, s=30,
                         color=COLORS[row.method], edgecolor=COLORS[dataset], linewidth=0.8, zorder=3)
        end = current[current.method == "L8"].iloc[0]
        ax_c.text(end.mae_improvement_percent + 0.5, end.signal_retention,
                  LABELS[(dataset, outcome)], color=COLORS[dataset], va="center", fontsize=6)
    ax_c.set_xlabel("Original-data MAE improvement over B8 (%)")
    ax_c.set_ylabel("Sustained worsening signal retained")
    ax_c.set_title("Accuracy--retention trade-off", loc="left")
    ax_c.set_ylim(0, 1.08); ax_c.set_xlim(-1, 32)
    ax_c.grid(color="#E5E5E5", lw=0.6)
    method_legend = [Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[m],
                            markeredgecolor="#555555", label=m) for m in ["B8", "A50", "L8"]]
    ax_c.legend(handles=method_legend, title="Baseline", loc="upper right")
    panel_label(ax_c, "c")
    save_figure(fig, "Figure_4_accuracy_signal_tradeoff")


def main() -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    print(f"Saved figures to {FIGURES}")


if __name__ == "__main__":
    main()
