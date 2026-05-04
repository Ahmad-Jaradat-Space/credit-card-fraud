# Credit Card Fraud Detection

This is the second of two capstones I built after finishing Andrew Ng's Machine Learning Specialization. The first one (forest cover type) was a roughly balanced multiclass problem that exercised everything from logistic regression up to gradient-boosted trees. The point of this one is the opposite: a binary problem with an extreme class imbalance, where most of the work is figuring out the right metrics and the right threshold instead of the right model.

The dataset is the well-known ULB credit-card fraud set: 284,807 transactions over two days in September 2013, of which 492 (~0.17%) are fraudulent. Features V1..V28 are anonymised PCA components produced by the original authors; only `Amount` is raw. I pull it from OpenML so the download doesn't need a Kaggle account.

The central claim the notebook puts on trial is that **the model isn't the lever; the threshold is**. All four supervised candidates land near each other on PR-AUC; what actually changes the operational outcome is picking a defensible decision threshold on the model's score.

## How the notebook is laid out

The notebook reads as a short paper with five sections:

1. **Introduction** — the cost asymmetry between missed fraud and false alarms, and what the deliverable actually is (a model *plus* an operating threshold).
2. **Data** — class imbalance (so why accuracy is broken here), single-feature signal in V14/V17, and the explicit modelling hypothesis.
3. **Methods** — five candidates, all class-weighted: logistic regression (from scratch and with sklearn), a small Keras NN, histogram gradient boosting, and a Gaussian anomaly detector from scratch as the unsupervised baseline from Part III of the course.
4. **Results** — combined PR/ROC curves, F1-vs-threshold panels, score distributions for each model, and the final operational ledger (precision/recall/F1 plus the test-set confusion matrix at the chosen threshold).
5. **Conclusion** — what the evidence supports, which model I'd actually ship and why, and what would change in real deployment (drift, adversarial behaviour, cost asymmetry).

Every plot is read out loud: a one-line setup before the cell, a finding-style title on the figure, and a 2–4 sentence takeaway after — *what to look at, what it means, what it indicates next.*

## Running it

Tested on macOS with Python 3.12.

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

The first cell of the notebook downloads the data via `sklearn.datasets.fetch_openml` and caches it as `data/creditcard.csv`. The `data/` folder is gitignored. The `notebook.ipynb` in this repo is already executed, so GitHub renders all outputs and plots inline — you can read it through without running anything.

## What's where

- `notebook.ipynb` — runs top to bottom
- `data.py` — OpenML download, stratified split, scaling
- `models.py` — the from-scratch class-weighted logistic regression and Gaussian anomaly detector
- `plots.py` — small matplotlib helpers (PR curve, F1-vs-threshold, confusion matrix, score distributions)
