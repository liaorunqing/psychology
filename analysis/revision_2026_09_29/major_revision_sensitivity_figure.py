"""Create a submission-grade robustness figure from frozen CSV outputs."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "aggregate"
FIG = ROOT / "figures" / "rendered"
COLORS = {"Sadness": "#0072B2", "Stress": "#D55E00", "PHQ-2": "#009E73",
          "Depressed mood": "#CC79A7"}
LABELS = {("Dejonckheere", "sad"): "Sadness", ("Dejonckheere", "stressed"): "Stress",
          ("CES", "phq2"): "PHQ-2", ("Marian", "depressed"): "Depressed mood"}


def style():
    mpl.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.titlesize": 9,
                         "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.linewidth": .7, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "legend.frameon": False, "figure.dpi": 180})


def forest(ax, frame, y_col, label_col, title):
    y = np.arange(len(frame))[::-1]
    for pos, row in zip(y, frame.itertuples(index=False)):
        color = COLORS[getattr(row, label_col)]
        estimate = row.relative_mae_percent
        ax.errorbar(estimate, pos,
                    xerr=[[estimate-row.ci_low], [row.ci_high-estimate]],
                    fmt="o", ms=4.2, color=color, ecolor=color, capsize=2, lw=1)
    ax.axvline(0, color="#555555", lw=.8, ls="--")
    ax.set_yticks(y, frame[y_col], fontsize=6.3)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("L8 vs B8 MAE change (%)")
    ax.grid(axis="x", color="#D9D9D9", lw=.45)
    ax.spines[["top", "right", "left"]].set_visible(False)


def main():
    style(); FIG.mkdir(parents=True, exist_ok=True)
    windows = pd.read_csv(OUT / "major_revision_window_length.csv")
    origins = pd.read_csv(OUT / "major_revision_pseudo_origin.csv")
    ipw = pd.read_csv(OUT / "major_revision_marian_missingness.csv")
    gap = pd.read_csv(OUT / "major_revision_ces_gap.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.75),
                             gridspec_kw={"width_ratios": [1.0, 1.28, 1.0]})

    ax = axes[0]
    for key, label in LABELS.items():
        g = windows[(windows.dataset == key[0]) & (windows.outcome == key[1])]
        ax.plot(g.window, g.relative_mae_percent, marker="o", ms=3.8, lw=1.2,
                color=COLORS[label], label=label)
    ax.axhline(0, color="#555555", lw=.8, ls="--")
    ax.set_xticks([4, 6, 8, 12]); ax.set_xlabel("History window K (reports)")
    ax.set_ylabel("L$K$ vs B$K$ MAE change (%)")
    ax.set_title("a  Window length", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#D9D9D9", lw=.45)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6.1, ncol=1, loc="upper right", handlelength=1.4)

    primary = origins[[tuple(x) in LABELS for x in origins[["dataset", "outcome"]].itertuples(index=False, name=None)]].copy()
    primary["label"] = [LABELS[x] for x in primary[["dataset", "outcome"]].itertuples(index=False, name=None)]
    primary["display"] = primary.label + " · " + primary.origin
    order = []
    for label in LABELS.values():
        order.extend([f"{label} · early", f"{label} · middle", f"{label} · late"])
    primary["rank"] = primary.display.map({x: i for i, x in enumerate(order)})
    primary = primary.sort_values("rank")
    forest(axes[1], primary, "display", "label", "b  Re-anchored origins")

    rows = []
    for row in ipw.itertuples(index=False):
        if row.outcome == "depressed":
            rows.append({"display": f"Marian · {row.analysis}", "label": "Depressed mood",
                         "relative_mae_percent": row.relative_mae_percent,
                         "ci_low": row.ci_low, "ci_high": row.ci_high})
    row = gap.iloc[0]
    rows.append({"display": "CES · ≥14-day gap", "label": "PHQ-2",
                 "relative_mae_percent": row.relative_mae_percent,
                 "ci_low": row.ci_low, "ci_high": row.ci_high})
    selection = pd.DataFrame(rows)
    forest(axes[2], selection, "display", "label", "c  Observation checks")

    fig.subplots_adjust(left=.09, right=.99, bottom=.17, top=.92, wspace=.72)
    fig.savefig(FIG / "Figure_5_major_revision_robustness.pdf", bbox_inches="tight")
    fig.savefig(FIG / "Figure_5_major_revision_robustness.png", bbox_inches="tight", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()

