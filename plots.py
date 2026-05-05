"""Production-grade plot helpers for the credit-card fraud notebook.

The colour system is deliberate:
  PRIMARY (deep teal)  -> chosen / headline series
  ACCENT  (warm clay)  -> contrast series
  WARN    (rust)       -> fraud / errors / costs
  GOOD    (muted sage) -> calibrated / corrected / baseline-beating
  MUTED   (slate)      -> baselines, naive lines, axis chrome
"""

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, confusion_matrix, roc_curve,
)

PRIMARY = "#0E4F5F"
ACCENT  = "#D88C4A"
GOOD    = "#5C9D7E"
WARN    = "#B0413E"
MUTED   = "#7A8C99"
INK     = "#1A1A1A"
PAPER   = "#FAFAF7"

LEGIT = PRIMARY
FRAUD = WARN
CONTEXT = MUTED
MODEL_PALETTE = {
    "logistic (scratch)":      PRIMARY,
    "logistic (sklearn)":      ACCENT,
    "Keras NN":                "#5E548E",
    "hist gradient boosting":  WARN,
    "anomaly":                 GOOD,
    # short-name aliases
    "scratch LR":  PRIMARY,
    "sklearn LR":  ACCENT,
    "HistGB":      WARN,
}


def apply_style():
    sns.set_theme(style="white", context="notebook")
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 140,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "savefig.edgecolor": PAPER,
        "axes.titleweight": "semibold",
        "axes.titlesize": 12.5,
        "axes.titlepad": 12,
        "axes.titlelocation": "left",
        "axes.labelsize": 10.5,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#BFC4CA",
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "font.family": "sans-serif",
        "font.size": 10.5,
    })
    try:
        mpl.rcParams["text.parse_math"] = False
    except KeyError:
        pass


# ------------------------------------------------------------------
# Generic helpers (ported from tas1-rainfall-to-price)
# ------------------------------------------------------------------
def caption(fig, text):
    fig.text(0.5, -0.04, text, ha="center", va="top",
             fontsize=9, color="#555", style="italic", wrap=True)


def annotate_point(ax, x, y, text, dx=20, dy=20, color=INK):
    ax.annotate(
        text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
        fontsize=9, color=color,
        arrowprops=dict(arrowstyle="-", color=color, lw=0.7, alpha=0.7),
        bbox=dict(boxstyle="round,pad=0.25", fc=PAPER, ec=color, lw=0.6, alpha=0.95),
    )


def kpi_card(ax, value, label, sub=None, color=PRIMARY):
    ax.axis("off")
    ax.text(0.5, 0.62, value, ha="center", va="center",
            fontsize=22, color=color, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.30, label, ha="center", va="center",
            fontsize=10, color=INK, transform=ax.transAxes)
    if sub:
        ax.text(0.5, 0.12, sub, ha="center", va="center",
                fontsize=8.5, color=MUTED, transform=ax.transAxes, style="italic")
    ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                    fill=False, ec="#D8DCE2", lw=1.0))


def kpi_banner(values):
    fig, axes = plt.subplots(1, len(values), figsize=(2.8 * len(values), 1.8))
    if len(values) == 1:
        axes = [axes]
    palette = [PRIMARY, ACCENT, WARN, GOOD, "#5E548E"]
    for ax, v, color in zip(axes, values, palette):
        kpi_card(ax, *v, color=color)
    plt.tight_layout()
    return fig


def business_summary(rows, ax=None):
    """Two-column verdict table: left = label, right = takeaway."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 0.55 * len(rows) + 1))
    ax.axis("off")
    n = len(rows)
    for i, (lhs, rhs) in enumerate(rows):
        y = 1 - (i + 0.5) / n
        ax.text(0.02, y, lhs, transform=ax.transAxes,
                fontsize=10.5, color=INK, fontweight="bold", va="center")
        ax.text(0.30, y, rhs, transform=ax.transAxes,
                fontsize=10, color=INK, va="center")
        ax.plot([0.01, 0.99], [1 - i / n, 1 - i / n],
                color="#E0E4EA", lw=0.6, transform=ax.transAxes)
    ax.set_xlim(0, 1)
    return ax


# ------------------------------------------------------------------
# Class balance + feature distributions
# ------------------------------------------------------------------
def class_balance(y, ax=None):
    counts = np.bincount(y)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.8))
    bars = ax.bar(["Legitimate", "Fraud"], counts, color=[LEGIT, FRAUD],
                  edgecolor=INK, lw=0.7, alpha=0.9, width=0.6)
    ax.set_yscale("log")
    for b, c in zip(bars, counts):
        rate = c / counts.sum() * 100
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c:,}\n({rate:.3f}%)",
                ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Fraud is 0.17% of all transactions: every TP move shifts measured recall by ~1pp",
                 color=INK)
    ax.set_ylim(top=counts.max() * 4)
    # plain-text log ticks (parse_math is off globally, so mathtext labels break)
    log_ticks = [10 ** k for k in range(int(np.log10(counts.max())) + 2)]
    ax.set_yticks(log_ticks)
    ax.set_yticklabels([f"{t:,}" if t < 1000 else f"{t/1000:.0f}K" if t < 1_000_000 else f"{t/1_000_000:.0f}M"
                        for t in log_ticks])
    return ax


def feature_distributions(df, cols, ax_row=None):
    n = len(cols)
    if ax_row is None:
        fig, ax_row = plt.subplots(1, n, figsize=(4.8 * n, 3.4), sharey=True)
    for ax, col in zip(ax_row, cols):
        legit = df.loc[df.y == 0, col]
        fraud = df.loc[df.y == 1, col]
        ax.hist(legit, bins=80, alpha=0.55, color=LEGIT, density=True,
                label="Legit", edgecolor=PAPER, lw=0.3)
        ax.hist(fraud, bins=40, alpha=0.7, color=FRAUD, density=True,
                label="Fraud", edgecolor=PAPER, lw=0.3)
        ax.set_title(col, color=INK)
        ax.set_xlabel(col)
        ax.legend(loc="upper right", fontsize=9)
    ax_row[0].set_ylabel("Density")
    plt.tight_layout()
    return ax_row


# ------------------------------------------------------------------
# Ranking diagnostics
# ------------------------------------------------------------------
def pr_curves(y_true, score_dict, ax=None, op_points=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, s in score_dict.items():
        p, r, _ = precision_recall_curve(y_true, s)
        col = MODEL_PALETTE.get(name, PRIMARY)
        ax.plot(r, p, label=name, linewidth=2, color=col, alpha=0.95)
    base = y_true.mean()
    ax.axhline(base, ls="--", color=MUTED, alpha=0.7,
               label=f"baseline ({base:.3%})")
    if op_points:
        for name, (recall, precision) in op_points.items():
            col = MODEL_PALETTE.get(name, PRIMARY)
            ax.scatter([recall], [precision], s=90, color=col,
                       edgecolor=INK, lw=0.9, zorder=5)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall curves: every supervised model dominates the prevalence baseline",
                 color=INK)
    ax.legend(loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    return ax


def roc_curves(y_true, score_dict, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, s in score_dict.items():
        fpr, tpr, _ = roc_curve(y_true, s)
        col = MODEL_PALETTE.get(name, PRIMARY)
        ax.plot(fpr, tpr, label=name, linewidth=2, color=col, alpha=0.95)
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, alpha=0.7, label="Chance")
    ax.set_xlabel("False positive rate (zoomed: only the low-FPR region matters here)")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC zoom on FPR < 5%: HistGB and Keras pull cleanly above the LRs",
                 color=INK)
    ax.set_xlim(0, 0.05)
    ax.legend(loc="lower right")
    return ax


def f1_vs_threshold(y_true, scores, title, ax=None):
    p, r, t = precision_recall_curve(y_true, scores)
    f1 = 2 * p * r / (p + r + 1e-12)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, p[:-1], label="Precision", color=PRIMARY, linewidth=2.0)
    ax.plot(t, r[:-1], label="Recall", color=ACCENT, linewidth=2.0)
    ax.plot(t, f1[:-1], label="F1", color=WARN, linewidth=2.4)
    star = int(np.argmax(f1[:-1]))
    ax.axvline(t[star], color=MUTED, linestyle="--", lw=1.0)
    ax.text(t[star], 1.02, f"F1* = {f1[star]:.2f} @ t={t[star]:.3f}",
            ha="center", fontsize=9, color=INK,
            bbox=dict(boxstyle="round,pad=0.25", fc=PAPER, ec=MUTED, lw=0.7))
    ax.set_xlabel("Calibrated probability threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.10)
    ax.set_title(title, color=INK)
    ax.legend(loc="lower left", fontsize=9)
    return ax, float(t[star])


def confusion(y_true, y_pred, title, ax=None, costs=None):
    """Standard 2x2; if `costs=(c_FP, c_FN)` annotates dollar cost per cell."""
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4.2))
    sns.heatmap(cm, annot=False, cmap="Reds", cbar=False,
                xticklabels=["Legit", "Fraud"],
                yticklabels=["Legit", "Fraud"],
                linewidths=0.6, linecolor=PAPER, ax=ax)
    cmax = cm.max()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = int(cm[i, j])
            txt_color = PAPER if v > cmax * 0.55 else INK
            ax.text(j + 0.5, i + 0.5, f"{v:,}", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=txt_color)
    if costs is not None:
        c_fp, c_fn = costs
        TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        ax.text(1.5, 0.5, f"\n${FP * c_fp:,.0f}", ha="center", va="center",
                fontsize=8, color=WARN, fontweight="medium")
        ax.text(0.5, 1.5, f"\n${FN * c_fn:,.0f}", ha="center", va="center",
                fontsize=8, color=WARN, fontweight="medium")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, color=INK)
    return ax


def score_distributions(y_true, scores, name, ax=None, threshold=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(scores[y_true == 0], bins=80, alpha=0.55, color=LEGIT,
            density=True, label="Legit", edgecolor=PAPER, lw=0.3)
    ax.hist(scores[y_true == 1], bins=40, alpha=0.75, color=FRAUD,
            density=True, label="Fraud", edgecolor=PAPER, lw=0.3)
    ax.set_yscale("log")
    # plain-text log ticks
    import matplotlib.ticker as mt
    ax.yaxis.set_major_formatter(mt.FuncFormatter(
        lambda v, _: f"{v:.0e}".replace("e+0", "e").replace("e-0", "e-")))
    if threshold is not None:
        ax.axvline(threshold, color=INK, ls="--", lw=1.2,
                   label=f"threshold = {threshold:.3f}")
    ax.set_xlabel("Calibrated probability score")
    ax.set_ylabel("Density (log)")
    ax.set_title(f"Score distribution — {name}", color=INK)
    ax.legend(loc="upper center", fontsize=9)
    return ax


def model_comparison(df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    df = df.sort_values("F1")
    y = np.arange(len(df))
    h = 0.25

    has_p_ci = {"precision_lo", "precision_hi"}.issubset(df.columns)
    has_r_ci = {"recall_lo", "recall_hi"}.issubset(df.columns)
    has_f_ci = {"F1_lo", "F1_hi"}.issubset(df.columns)

    ax.barh(y - h, df["precision"], height=h, label="Precision",
            color=PRIMARY, alpha=0.92, edgecolor=INK, lw=0.4)
    ax.barh(y, df["recall"], height=h, label="Recall",
            color=ACCENT, alpha=0.92, edgecolor=INK, lw=0.4)
    ax.barh(y + h, df["F1"], height=h, label="F1",
            color=WARN, alpha=0.92, edgecolor=INK, lw=0.4)

    if has_p_ci:
        ax.errorbar(df["precision"], y - h,
                    xerr=[df["precision"] - df["precision_lo"], df["precision_hi"] - df["precision"]],
                    fmt="none", ecolor=INK, capsize=2.5, lw=0.7)
    if has_r_ci:
        ax.errorbar(df["recall"], y,
                    xerr=[df["recall"] - df["recall_lo"], df["recall_hi"] - df["recall"]],
                    fmt="none", ecolor=INK, capsize=2.5, lw=0.7)
    if has_f_ci:
        ax.errorbar(df["F1"], y + h,
                    xerr=[df["F1"] - df["F1_lo"], df["F1_hi"] - df["F1"]],
                    fmt="none", ecolor=INK, capsize=2.5, lw=0.7)

    for i, (a, b, c) in enumerate(zip(df["precision"], df["recall"], df["F1"])):
        ax.text(a + 0.005, i - h, f"{a:.2f}", va="center", fontsize=8.5)
        ax.text(b + 0.005, i, f"{b:.2f}", va="center", fontsize=8.5)
        ax.text(c + 0.005, i + h, f"{c:.2f}", va="center", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"])
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Score (with paired-bootstrap 95% CI caps when available)")
    ax.set_title("Test-set metrics by model: HistGB and Keras NN are tied within the bootstrap noise",
                 color=INK)
    ax.legend(loc="lower right")
    return ax


# ------------------------------------------------------------------
# NEW: domain-specific helpers
# ------------------------------------------------------------------
def cost_threshold_sweep(y_true, scores_dict, cost_ratios=(1, 5, 20),
                         ax=None, mean_amount=88.5):
    """Net savings ($) vs threshold across cost ratios, per model.

    Cost model: FN costs `mean_amount * ratio` (missed fraud, full chargeback);
    FP costs `mean_amount * 1` (one declined-card customer-friction event).
    Net savings = avoided_fraud - friction_cost over the validation set.
    """
    if ax is None:
        fig, ax = plt.subplots(1, len(cost_ratios),
                                figsize=(4.6 * len(cost_ratios), 4),
                                sharey=True)
    if len(cost_ratios) == 1:
        ax = [ax]
    for axi, ratio in zip(ax, cost_ratios):
        for name, s in scores_dict.items():
            ts = np.linspace(0.005, 0.995, 200)
            net = []
            for t in ts:
                pred = (s >= t).astype(int)
                FP = int(((pred == 1) & (y_true == 0)).sum())
                FN = int(((pred == 0) & (y_true == 1)).sum())
                cost_fp = FP * mean_amount
                # net savings = chargebacks avoided minus friction
                avoided = (y_true.sum() - FN) * mean_amount * ratio
                net.append(avoided - cost_fp)
            net = np.array(net)
            col = MODEL_PALETTE.get(name, PRIMARY)
            axi.plot(ts, net, lw=1.6, color=col, label=name, alpha=0.95)
            star = ts[int(np.argmax(net))]
            axi.scatter([star], [net.max()], s=70, color=col,
                        edgecolor=INK, lw=0.7, zorder=5)
        axi.set_title(f"FN:FP cost ratio = {ratio}:1", color=INK)
        axi.set_xlabel("Threshold")
        axi.axhline(0, color=MUTED, lw=0.6)
    ax[0].set_ylabel("Net savings ($)")
    ax[-1].legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    return ax


def pareto_frontier(y_true, scores_dict, ax=None, n_thresh=200):
    """FN-rate vs FP-rate for each model, with the convex frontier marked."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, s in scores_dict.items():
        fpr, tpr, _ = roc_curve(y_true, s)
        fnr = 1 - tpr
        col = MODEL_PALETTE.get(name, PRIMARY)
        ax.plot(fpr, fnr, lw=1.8, color=col, label=name, alpha=0.92)
    # cost-optimal corner markers (closest to origin in each direction)
    for ratio, marker, label in [(1, "o", "1:1"), (5, "s", "5:1"), (20, "^", "20:1")]:
        for name, s in scores_dict.items():
            fpr, tpr, _ = roc_curve(y_true, s)
            fnr = 1 - tpr
            costs = ratio * fnr + fpr
            i_opt = int(np.argmin(costs))
            col = MODEL_PALETTE.get(name, PRIMARY)
            ax.scatter([fpr[i_opt]], [fnr[i_opt]], s=55, color=col,
                       marker=marker, edgecolor=INK, lw=0.6, zorder=5)
    legend_models = ax.legend(loc="upper right", title="Model")
    ax.add_artist(legend_models)
    legend_markers = [
        plt.Line2D([0], [0], marker="o", color="white", markerfacecolor=MUTED,
                   markersize=8, markeredgecolor=INK, label="opt @ 1:1"),
        plt.Line2D([0], [0], marker="s", color="white", markerfacecolor=MUTED,
                   markersize=8, markeredgecolor=INK, label="opt @ 5:1"),
        plt.Line2D([0], [0], marker="^", color="white", markerfacecolor=MUTED,
                   markersize=8, markeredgecolor=INK, label="opt @ 20:1"),
    ]
    ax.legend(handles=legend_markers, loc="lower left", title="Cost-optimal point",
              fontsize=9)
    ax.set_xlim(0, 0.05)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("False positive rate (customer-friction rate)")
    ax.set_ylabel("False negative rate (missed-fraud rate)")
    ax.set_title("Pareto frontier of error trade-offs: cost-optimal points shift left as FN cost rises",
                 color=INK)
    return ax
