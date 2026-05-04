"""Plot helpers."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, confusion_matrix, roc_curve,
)

LEGIT = "#4c78a8"
FRAUD = "#e45756"
ACCENT = "#2c5f8d"


def apply_style():
    sns.set_theme(style="whitegrid", context="notebook")
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "axes.titleweight": "semibold",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
    })


def class_balance(y, ax=None):
    counts = np.bincount(y)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(["legitimate", "fraud"], counts, color=[LEGIT, FRAUD])
    ax.set_yscale("log")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c:,}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("count (log)")
    ax.set_title("class balance")
    return ax


def feature_distributions(df, cols, ax_row=None):
    """Histogram of each col, fraud vs legit overlaid, density-normalised."""
    n = len(cols)
    if ax_row is None:
        fig, ax_row = plt.subplots(1, n, figsize=(4.5 * n, 3))
    for ax, col in zip(ax_row, cols):
        for c, name, color in [(0, "legit", LEGIT), (1, "fraud", FRAUD)]:
            ax.hist(df.loc[df.y == c, col], bins=60, alpha=0.55,
                    label=name, color=color, density=True)
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("density")
        ax.legend()
    return ax_row


def pr_curves(y_true, score_dict, ax=None):
    """All models on one PR plot."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    for name, s in score_dict.items():
        p, r, _ = precision_recall_curve(y_true, s)
        ax.plot(r, p, label=name, linewidth=2)
    base = y_true.mean()
    ax.axhline(base, ls="--", color="grey", alpha=0.6,
               label=f"baseline ({base:.3%})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("precision-recall curves (validation)")
    ax.legend(loc="best")
    return ax


def roc_curves(y_true, score_dict, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    for name, s in score_dict.items():
        fpr, tpr, _ = roc_curve(y_true, s)
        ax.plot(fpr, tpr, label=name, linewidth=2)
    ax.plot([0, 1], [0, 1], ls="--", color="grey", alpha=0.6, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC curves (validation)")
    ax.set_xlim(0, 0.05)  # zoom: at this imbalance, only the low-FPR region matters
    ax.legend(loc="lower right")
    return ax


def f1_vs_threshold(y_true, scores, title, ax=None):
    p, r, t = precision_recall_curve(y_true, scores)
    f1 = 2 * p * r / (p + r + 1e-12)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    ax.plot(t, p[:-1], label="precision", color=ACCENT, linewidth=2)
    ax.plot(t, r[:-1], label="recall", color="#7e57c2", linewidth=2)
    ax.plot(t, f1[:-1], label="F1", color=FRAUD, linewidth=2)
    star = int(np.argmax(f1[:-1]))
    ax.axvline(t[star], color="grey", linestyle="--",
               label=f"best F1 {f1[star]:.2f} @ {t[star]:.3f}")
    ax.set_xlabel("threshold")
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    return ax, float(t[star])


def confusion(y_true, y_pred, title, ax=None):
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["legit", "fraud"],
                yticklabels=["legit", "fraud"],
                linewidths=0.5, linecolor="white", ax=ax)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    return ax


def score_distributions(y_true, scores, name, ax=None):
    """Log-density histograms of model scores, fraud vs legit."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    ax.hist(scores[y_true == 0], bins=80, alpha=0.55, color=LEGIT,
            density=True, label="legit")
    ax.hist(scores[y_true == 1], bins=40, alpha=0.7, color=FRAUD,
            density=True, label="fraud")
    ax.set_yscale("log")
    ax.set_xlabel("model score")
    ax.set_ylabel("density (log)")
    ax.set_title(f"score distribution — {name}")
    ax.legend()
    return ax


def model_comparison(df, ax=None):
    """Horizontal grouped bar of precision/recall/F1."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    df = df.sort_values("F1")
    y = np.arange(len(df))
    h = 0.25
    ax.barh(y - h, df["precision"], height=h, label="precision", color=ACCENT)
    ax.barh(y, df["recall"], height=h, label="recall", color="#7e57c2")
    ax.barh(y + h, df["F1"], height=h, label="F1", color=FRAUD)
    for i, (a, b, c) in enumerate(zip(df["precision"], df["recall"], df["F1"])):
        ax.text(a + 0.005, i - h, f"{a:.2f}", va="center", fontsize=8)
        ax.text(b + 0.005, i, f"{b:.2f}", va="center", fontsize=8)
        ax.text(c + 0.005, i + h, f"{c:.2f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("score")
    ax.set_title("test-set metrics by model")
    ax.legend(loc="lower right")
    return ax
