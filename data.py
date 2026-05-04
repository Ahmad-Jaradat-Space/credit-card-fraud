"""Load the ULB credit-card fraud dataset.

Originally hosted on Kaggle, but it's also mirrored on OpenML, which
doesn't require auth. We pull it once and cache the CSV under data/.

Columns: Time, V1..V28 (PCA-projected anonymised features), Amount, Class.
Class is 0 = legitimate, 1 = fraud.
"""

import os

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "data", "creditcard.csv")


def download():
    os.makedirs(os.path.dirname(CSV), exist_ok=True)
    if os.path.exists(CSV):
        return
    print("downloading creditcard from openml")
    bunch = fetch_openml(
        name="creditcard", version=1, as_frame=True, parser="auto",
    )
    df = bunch.frame.copy()
    # OpenML may name the label "Class" or "class"; normalise.
    if "Class" not in df.columns and "class" in df.columns:
        df = df.rename(columns={"class": "Class"})
    df["Class"] = df["Class"].astype(int)
    df.to_csv(CSV, index=False)


def load(seed=0):
    """Return train/val/test splits, stratified.

    `Amount` is scaled; V1..V28 are PCA components from the original
    paper and already roughly standardised, so they're left alone.
    (The OpenML mirror drops the `Time` column, so we don't have it.)
    """
    download()
    df = pd.read_csv(CSV)
    X = df.drop(columns="Class").values.astype(np.float32)
    y = df["Class"].values.astype(np.int64)

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.20, stratify=y_tv, random_state=seed
    )

    # scale Amount (last column); V1..V28 are already PCA-scaled
    amt = X_train.shape[1] - 1
    scaler = StandardScaler().fit(X_train[:, [amt]])
    for arr in (X_train, X_val, X_test):
        arr[:, [amt]] = scaler.transform(arr[:, [amt]])

    return X_train, y_train, X_val, y_val, X_test, y_test


FEATURE_NAMES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
