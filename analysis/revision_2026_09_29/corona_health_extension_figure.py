"""Figure 4: protocol-locked Corona Health external extension.

Figure contract
---------------
Core conclusion: In an independently sourced smartphone-app cohort, recent
histories improve next-report error for PHQ-9 and GAD-7; the PHQ-9 improvement
survives a long-gap sensitivity and exceeds the fitted stationary AR(1) null.
Archetype: asymmetric quantitative grid.
Hero evidence: relative participant-balanced MAE change and 95% bootstrap CI.
Validation evidence: PHQ-9 observed effect against 5,000 stationary draws.
Robustness: participant-balanced ordinal accuracy for mean and median rules.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "outputs" / "revision_2026_09_29" / "corona_health_extension"
FIG = ROOT / "submission" / "bmc_dynamic_baseline_manuscript_2026-09-30" / "figures"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": .7,
    "xtick.major.width": .7,
    "ytick.major.width": .7,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

BLUE = "#2A6F97"
TEAL = "#2A9D8F"
ORANGE = "#D98E32"
DARK = "#30343B"
GREY = "#A7ADB4"
LIGHT = "#E7ECF0"


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-.12, 1.06, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", ha="left")


def main() -> None:
    summary = pd.read_csv(OUT / "corona_health_extension_summary.csv")
    ordinal = pd.read_csv(OUT / "corona_health_ordinal_sensitivity.csv")
    null = pd.read_parquet(OUT / "corona_health_phq9_stationary_null.parquet")
    if set(summary.outcome) != {"PHQ-9", "GAD-7"} or len(null) != 5000:
        raise AssertionError("Unexpected figure source data")

    fig = plt.figure(figsize=(7.2, 5.2), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], width_ratios=[1.08, .92],
                            left=.09, right=.98, bottom=.11, top=.95,
                            wspace=.34, hspace=.48)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])

    order = ["PHQ-9", "GAD-7"]
    y = np.arange(2)[::-1]
    for position, outcome, color in zip(y, order, [BLUE, TEAL]):
        row = summary.loc[summary.outcome.eq(outcome)].iloc[0]
        effect = -row.relative_mae_percent
        low = -row.relative_mae_ci_high
        high = -row.relative_mae_ci_low
        ax_a.errorbar(effect, position, xerr=[[effect - low], [high - effect]],
                      fmt="o", ms=6, color=color, ecolor=color, capsize=3, lw=1.3)
        ax_a.text(high + .7, position, f"{effect:.1f}%", color=color, va="center")
    ax_a.axvline(0, color=DARK, lw=.8, ls=(0, (3, 2)))
    ax_a.set_yticks(y, ["PHQ-9 (primary)\n$n=324$", "GAD-7 (secondary)\n$n=327$"])
    ax_a.set_xlim(0, 25)
    ax_a.set_xlabel("Reduction in participant-balanced MAE (%)")
    ax_a.set_title("External extension reproduces the direction", loc="left", fontweight="bold")
    panel_label(ax_a, "a")

    values = null.mae_difference_B8_minus_L8.to_numpy()
    observed = float(summary.loc[summary.outcome.eq("PHQ-9"),
                                 "mae_difference_B8_minus_L8"].iloc[0])
    ax_b.hist(values, bins=35, density=True, color=LIGHT, edgecolor="white", linewidth=.4)
    ax_b.axvline(values.mean(), color=GREY, lw=1.2, label=f"Null mean = {values.mean():.3f}")
    ax_b.axvline(observed, color=ORANGE, lw=2, label=f"Observed = {observed:.3f}")
    ax_b.text(observed, ax_b.get_ylim()[1] * .92, "$p_{MC}=0.0002$", color=ORANGE,
              ha="right", va="top")
    ax_b.set_xlabel("PHQ-9 B8 minus L8 MAE")
    ax_b.set_ylabel("Density")
    ax_b.set_title("Observed gain exceeds fitted stationary persistence", loc="left",
                   fontweight="bold")
    ax_b.legend(loc="upper left", bbox_to_anchor=(0, .86))
    panel_label(ax_b, "b")

    metrics = [("exact_accuracy", "Exact score"),
               ("within_two_accuracy", "Within two points")]
    methods = [("B8", DARK, "o"), ("L8", BLUE, "s"), ("Median8", TEAL, "D")]
    x_positions, labels = [], []
    cursor = 0
    for outcome in order:
        row = ordinal.loc[ordinal.outcome.eq(outcome)].iloc[0]
        for suffix, metric_label in metrics:
            for offset, (method, color, marker) in zip([-.18, 0, .18], methods):
                value = 100 * row[f"{method}_{suffix}"]
                ax_c.plot(cursor + offset, value, marker=marker, ms=5.5, color=color,
                          markeredgecolor="white", markeredgewidth=.4, linestyle="none")
            x_positions.append(cursor); labels.append(f"{outcome}\n{metric_label}")
            cursor += 1
        cursor += .45
    ax_c.set_xticks(x_positions, labels)
    ax_c.set_ylabel("Participant-balanced accuracy (%)")
    ax_c.set_ylim(15, 85)
    ax_c.grid(axis="y", color=LIGHT, lw=.7)
    ax_c.set_axisbelow(True)
    handles = [plt.Line2D([], [], marker=marker, color=color, linestyle="none", ms=5.5,
                          label=method) for method, color, marker in methods]
    ax_c.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.16))
    ax_c.set_title("Ordinal-score checks favour recent summaries", loc="left", fontweight="bold")
    panel_label(ax_c, "c")

    FIG.mkdir(parents=True, exist_ok=True)
    base = FIG / "Figure_4_corona_health_extension"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    source = []
    for row in summary.itertuples(index=False):
        source.append({"panel": "a", "outcome": row.outcome,
                       "estimate": -row.relative_mae_percent,
                       "ci_low": -row.relative_mae_ci_high,
                       "ci_high": -row.relative_mae_ci_low,
                       "participants": row.participants})
    for value in values:
        source.append({"panel": "b", "outcome": "PHQ-9 stationary null",
                       "estimate": value, "ci_low": np.nan, "ci_high": np.nan,
                       "participants": 324})
    pd.DataFrame(source).to_csv(OUT / "Figure_4_source_data.csv", index=False)
    print(base)


if __name__ == "__main__":
    main()
