"""Stability-accuracy frontier for DCT against regularizer sweeps and ensembles.

Per trial: hold out 30% as a test set, split the rest into two disjoint halves,
and inside each half reserve 20% for early stopping. Three models per arm:
  A  half 1, seed a
  B  half 2, seed a        (A, B): differ in training sample and seed
  C  half 1, seed b        (A, C): differ in seed only
All arms see the same splits in a trial, so differences are paired by trial.

Arms (dropout 0.3, one pass unless stated):
  standard, dropout sweep, weight-decay sweep         the regularization frontier
  DCT lam=l (5 passes); DCT lam=5 (2 passes)   the method; a gradient-noise null
  5 passes lam=0                               DCT's compute without the penalty
  bagging-5, seed-ens-5                        rungs that average instead of shrink
  data=25/50/75%                               how much data an agreement gain buys

Training uses early stopping on a validation fifth of each half. With
--fixed-epochs 30 every model instead trains on the whole half for 30 epochs,
the original notebook's protocol, as a robustness run (results/fixed30/).

Writes results/<dataset>.csv (one row per arm x pair type x trial) and
results/preds/<dataset>/trial_<t>.npz with every model's test probabilities, its
mask variance, and the test labels, for the mechanism tests in analysis.py.
Finished datasets are skipped on rerun.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml, load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dct import mask_variance, pair_metrics, predict_proba, train

N_JOBS = 6
BASE = dict(n_passes=1, lam=0.0, weight_decay=0.0, dropout=0.3)
ARMS = {
    "standard": BASE,
    "dropout=0.1": BASE | dict(dropout=0.1),
    "dropout=0.5": BASE | dict(dropout=0.5),
    "dropout=0.7": BASE | dict(dropout=0.7),
    "wd=1": BASE | dict(weight_decay=1.0),
    "wd=3": BASE | dict(weight_decay=3.0),
    "wd=5": BASE | dict(weight_decay=5.0),
    "wd=10": BASE | dict(weight_decay=10.0),
    "wd=20": BASE | dict(weight_decay=20.0),
    "wd=30": BASE | dict(weight_decay=30.0),
    "5 passes": BASE | dict(n_passes=5),
    "DCT lam=0.5": BASE | dict(n_passes=5, lam=0.5),
    "DCT lam=1.5": BASE | dict(n_passes=5, lam=1.5),
    "DCT lam=5": BASE | dict(n_passes=5, lam=5.0),
    "DCT lam=15": BASE | dict(n_passes=5, lam=15.0),
    "DCT lam=5, 2 passes": BASE | dict(n_passes=2, lam=5.0),
    "bagging-5": BASE | dict(ensemble="bag", k=5),
    "seed-ens-5": BASE | dict(ensemble="seed", k=5),
    "data=25%": BASE | dict(frac=0.25),
    "data=50%": BASE | dict(frac=0.5),
    "data=75%": BASE | dict(frac=0.75),
}
FAMILY = {
    **{a: "dropout" for a in ARMS if a.startswith("dropout")},
    **{a: "wd" for a in ARMS if a.startswith("wd")},
    **{a: "DCT" for a in ARMS if a.startswith("DCT")},
    **{a: "data" for a in ARMS if a.startswith("data")},
    "standard": "dropout",
    "5 passes": "control",
    "bagging-5": "ensemble",
    "seed-ens-5": "ensemble",
}


def datasets():
    out = {
        "High Noise": make_classification(
            n_samples=700, n_features=15, n_informative=8, n_redundant=3,
            flip_y=0.15, class_sep=0.65, random_state=42,
        ),
        "Small Sample": make_classification(
            n_samples=400, n_features=12, n_informative=8, n_redundant=2,
            flip_y=0.05, class_sep=0.8, random_state=43,
        ),
        "High Dim": make_classification(
            n_samples=600, n_features=40, n_informative=8, n_redundant=5,
            flip_y=0.08, class_sep=0.7, random_state=44,
        ),
        "Breast Cancer": load_breast_cancer(return_X_y=True),
    }  # fmt: skip
    adult = fetch_openml("adult", version=2, as_frame=True)
    X, y = adult.data, (adult.target == ">50K").astype(int).to_numpy()
    cats = X.select_dtypes("category").columns
    enc = ColumnTransformer(
        [("cat", OneHotEncoder(min_frequency=50, handle_unknown="infrequent_if_exist",
                               sparse_output=False), cats)],
        remainder="passthrough",
    )  # fmt: skip
    # A 3000-row subsample keeps Adult at the scale of the other datasets.
    idx = np.random.default_rng(0).choice(len(X), 3000, replace=False)
    out["Adult (3k)"] = (enc.fit_transform(X.iloc[idx]).astype(float), y[idx])
    return out


class Half:
    """One training half: scaler fit on it, a train/val split inside it.

    With a fixed epoch budget there is nothing to validate, so the whole half
    is used for training (as in the original notebook)."""

    def __init__(self, X, y, seed, val_split=True):
        self.scaler = StandardScaler().fit(X)
        Xs = torch.tensor(self.scaler.transform(X)).float()
        ys = torch.tensor(y)
        if val_split:
            tr, va = train_test_split(
                np.arange(len(X)), test_size=0.2, random_state=seed
            )
        else:
            tr = va = np.arange(len(X))
        self.Xtr, self.ytr, self.Xval, self.yval = Xs[tr], ys[tr], Xs[va], ys[va]

    def transform(self, X):
        return torch.tensor(self.scaler.transform(X)).float()


def fit_arm(cfg, half, Xte, seed, fixed_epochs=None):
    """Fit one arm on one half. Returns (test probs, mask variance or None, epochs)."""
    cfg = dict(cfg)
    if fixed_epochs:
        cfg |= dict(patience=None, max_epochs=fixed_epochs)
    ensemble, k, frac = cfg.pop("ensemble", None), cfg.pop("k", 1), cfg.pop("frac", 1.0)
    rng = np.random.default_rng(seed)
    Xtr, ytr = half.Xtr, half.ytr
    if frac < 1:
        keep = rng.choice(len(Xtr), int(frac * len(Xtr)), replace=False)
        Xtr, ytr = Xtr[keep], ytr[keep]
    probs, epochs = [], []
    for m in range(k):
        Xm, ym = Xtr, ytr
        if ensemble == "bag":
            boot = rng.integers(0, len(Xtr), len(Xtr))
            Xm, ym = Xtr[boot], ytr[boot]
        model, ep = train(Xm, ym, half.Xval, half.yval, **cfg, seed=seed + 100 * m)
        probs.append(predict_proba(model, Xte))
        epochs.append(ep)
    mv = mask_variance(model, Xte, seed=seed) if k == 1 else None
    return np.mean(probs, 0), mv, float(np.mean(epochs))


def one_trial(X, y, trial, preds_dir, fixed_epochs=None, arms=None):
    torch.set_num_threads(1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=trial)
    X1, X2, y1, y2 = train_test_split(Xtr, ytr, test_size=0.5, random_state=trial * 2)
    halves = [
        Half(X1, y1, trial, not fixed_epochs),
        Half(X2, y2, trial, not fixed_epochs),
    ]
    base = 1000 * trial
    rows, store = [], {"y": yte}
    path = preds_dir / f"trial_{trial}.npz"
    if arms and path.exists():  # adding arms to a finished run: keep what is there
        store = dict(np.load(path))
    for arm, cfg in ARMS.items():
        if arms and arm not in arms:
            continue
        fits = {
            "A": fit_arm(
                cfg, halves[0], halves[0].transform(Xte), base + 0, fixed_epochs
            ),
            "B": fit_arm(
                cfg, halves[1], halves[1].transform(Xte), base + 1, fixed_epochs
            ),
            "C": fit_arm(
                cfg, halves[0], halves[0].transform(Xte), base + 2, fixed_epochs
            ),
        }
        for tag, (p, mv, ep) in fits.items():
            store[f"{arm}/{tag}/p"] = p.astype(np.float32)
            if mv is not None:
                store[f"{arm}/{tag}/mv"] = mv.astype(np.float32)
        epochs = np.mean([f[2] for f in fits.values()])
        for pair, (u, v) in {"sample": ("A", "B"), "seed": ("A", "C")}.items():
            m = pair_metrics(fits[u][0], fits[v][0], yte)
            rows.append(dict(arm=arm, trial=trial, pair=pair, epochs=epochs) | m)
    np.savez_compressed(path, **store)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument(
        "--arms",
        nargs="*",
        default=None,
        help="run only these arms and merge them into existing results",
    )
    ap.add_argument(
        "--fixed-epochs",
        type=int,
        default=None,
        help="train every model for exactly this many epochs, no "
        "validation split (the original protocol); default is early stopping",
    )
    args = ap.parse_args()
    for name, (X, y) in datasets().items():
        if args.datasets and name not in args.datasets:
            continue
        path = args.out / f"{name}.csv"
        if path.exists() and not args.arms:
            continue
        preds_dir = args.out / "preds" / name
        preds_dir.mkdir(parents=True, exist_ok=True)
        res = Parallel(n_jobs=N_JOBS)(
            delayed(one_trial)(X, y, t, preds_dir, args.fixed_epochs, args.arms)
            for t in range(args.trials)
        )
        df = pd.DataFrame([r for rows in res for r in rows])
        df.insert(0, "dataset", name)
        if args.arms and path.exists():
            old = pd.read_csv(path)
            df = pd.concat([old[~old.arm.isin(args.arms)], df], ignore_index=True)
        df.to_csv(path, index=False)
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    # Importable-module entry so joblib pickles workers by reference.
    import experiment

    experiment.main()
