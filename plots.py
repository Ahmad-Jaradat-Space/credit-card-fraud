"""Plot helpers."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, confusion_matrix


def class_balance(y, ax=None):
    counts = np.bincount(y)
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["legitimate", "fraud"], counts, color=["#4c78a8", "#e45756"])
    ax.set_yscale("log")
    for i, c in enumerate(counts):
        ax.text(i, c, f"{c:,}", ha="center", va="bottom")
    ax.set_ylabel("count (log)")
    return ax


def pr_curve(y_true, scores, label, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    p, r, _ = precision_recall_curve(y_true, scores)
    ax.plot(r, p, label=label)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.legend()
    return ax


def f1_vs_threshold(y_true, scores, ax=None):
    p, r, t = precision_recall_curve(y_true, scores)
    f1 = 2 * p * r / (p + r + 1e-12)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    ax.plot(t, f1[:-1])
    ax.set_xlabel("threshold")
    ax.set_ylabel("F1")
    star = np.argmax(f1[:-1])
    ax.axvline(t[star], color="grey", linestyle="--",
               label=f"best at {t[star]:.3f} (F1 {f1[star]:.2f})")
    ax.legend()
    return ax, t[star]


def confusion(y_true, y_pred, title, ax=None):
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        _, ax = plt.subplots(figsize=(3.5, 3))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["legit", "fraud"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["legit", "fraud"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    return ax
