# Credit Card Fraud Detection

This is the second of two capstones I built after finishing Andrew Ng's Machine Learning Specialization. The first one (forest cover type) was a roughly balanced multiclass problem that exercised everything from logistic regression up to gradient-boosted trees. The point of this one is the opposite: a binary problem with an extreme class imbalance, where most of the work is figuring out the right metrics and the right threshold instead of the right model.

The dataset is the well-known ULB credit-card fraud set: 284,807 transactions over two days in September 2013, of which 492 (~0.17%) are fraudulent. The features V1..V28 are anonymised PCA components produced by the original authors; only `Time` and `Amount` are raw. I pull it from OpenML so the download doesn't need a Kaggle account.

In one notebook I compare:

- logistic regression I wrote from scratch in numpy, with per-class loss weights
- the same in sklearn
- a small neural net in Keras, also class-weighted
- histogram gradient boosting (sklearn)
- a Gaussian anomaly-detection baseline I wrote from scratch (the unsupervised approach from Part III of the specialization), with epsilon picked by maximising F1 on the validation set

The interesting bit isn't which model wins — they all land near each other on F1. It's the threshold sweep and the precision-recall curves, which are exactly the diagnostic the course emphasises for skewed datasets.

## Running it

Tested on macOS with Python 3.12.

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

The first cell of the notebook downloads the data via `sklearn.datasets.fetch_openml` and caches it as `data/creditcard.csv`. The `data/` folder is gitignored.

## What's where

- `notebook.ipynb` — runs top to bottom
- `data.py` — OpenML download, stratified split, scaling
- `models.py` — the from-scratch class-weighted logistic regression and Gaussian anomaly detector
- `plots.py` — small matplotlib helpers (PR curve, F1-vs-threshold, confusion matrix)
