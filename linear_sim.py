"""Linear-logistic simulation: where the DCT penalty has a closed form.

Data: x ~ N(0, Sigma) in d = 30 dimensions with eigenvalues 1/k. The true
coefficient lies on the five largest eigendirections (scenario `top`: shrinkage
removes noise and keeps signal) or the five smallest (`bottom`: shrinkage eats
the signal). y ~ Bernoulli(sigmoid(beta'x)), with beta scaled so the Bayes
accuracy is 80%. Each trial draws two disjoint training halves and a 2,000-point
test set; the two fits are scored for accuracy and agreement.

Arms:
  ridge C=c          sklearn L2 logistic regression, the plain shrinkage frontier
  dropout q=q        logistic regression with Wager, Wang & Liang's (2013)
                     closed-form dropout penalty
                       q/(1-q)/2 * mean_i p_i(1-p_i) sum_j w_j^2 x_ij^2
  DCT lam=l          the dropout q=0.3 penalty plus the delta-method DCT penalty
                       2 l q/(1-q) * mean_i (p_i(1-p_i))^2 sum_j w_j^2 x_ij^2
  DCT-masks lam=l    the actual DCT loss with sampled masks, 5 passes
                     (checks the closed form); `2 passes` is the gradient-noise null
  bagging-5          five bootstrap refits of dropout q=0.3, probabilities averaged
  DCT lam=5 + bagging-5

All deterministic arms are fit by L-BFGS to convergence, so the comparison is
between penalties, not optimizers. Writes results/linear.csv.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy.optimize import brentq
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from dct import pair_metrics

D, N_TEST, Q_BASE, N_JOBS = 30, 2000, 0.3, 6
RIDGE_C = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100]
DROPOUT_Q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DCT_LAM = [0.5, 1.5, 5, 15, 50]
MASK_LAM = [0.5, 1.5, 5, 15]


def bayes_scale(acc_target=0.8):
    """Variance v of beta'x such that E[sigmoid(|N(0, v)|)] equals the target."""

    def bayes_acc(v):
        z = norm.rvs(size=200_000, random_state=0) * np.sqrt(v)
        return np.mean(1 / (1 + np.exp(-np.abs(z))))

    return brentq(lambda v: bayes_acc(v) - acc_target, 0.01, 1000)


def make_problem(scenario, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(D, D)))
    eig = 1 / np.arange(1, D + 1)
    cov = Q @ np.diag(eig) @ Q.T
    idx = np.arange(5) if scenario == "top" else np.arange(D - 5, D)
    direction = Q[:, idx].sum(1)
    v = bayes_scale()
    beta = direction * np.sqrt(v / (direction @ cov @ direction))
    return cov, beta


def sample(cov, beta, n, rng):
    X = rng.multivariate_normal(np.zeros(D), cov, size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-X @ beta))).astype(int)
    return X, y


def fit_penalized(X, y, q, lam, max_iter=300):
    """Logistic regression with the closed-form dropout (+ DCT) penalty, by L-BFGS."""
    Xt, yt = torch.tensor(X), torch.tensor(y, dtype=torch.float64)
    w = torch.zeros(D, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    coef = q / (1 - q)
    opt = torch.optim.LBFGS([w, b], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = Xt @ w + b
        p = torch.sigmoid(z)
        ce = torch.nn.functional.binary_cross_entropy_with_logits(z, yt)
        pq = p * (1 - p)
        ridge = (Xt**2 * w**2).sum(1)
        pen = 0.5 * coef * (pq * ridge).mean() + 2 * lam * coef * (pq**2 * ridge).mean()
        loss = ce + pen
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy(), b.item()


def fit_masks(X, y, q, lam, n_passes, seed, steps=1500, lr=0.05):
    """The actual DCT loss on a logistic model with sampled input-dropout masks."""
    torch.manual_seed(seed)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    w = torch.zeros(D, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        masks = (torch.rand(n_passes, *Xt.shape, dtype=torch.float64) > q) / (1 - q)
        z = (masks * Xt) @ w + b  # (n_passes, n)
        ce = torch.nn.functional.cross_entropy(
            torch.stack([torch.zeros_like(z), z], -1).reshape(-1, 2),
            yt.repeat(n_passes),
        )
        p = torch.sigmoid(z)
        pairs = [
            ((p[i] - p[j]) ** 2).mean()
            for i in range(n_passes)
            for j in range(i + 1, n_passes)
        ]
        # MSE over the two-class probability vector equals the squared gap in p
        pen = torch.stack(pairs).mean() if lam > 0 else 0.0
        (ce + lam * pen).backward()
        opt.step()
    return w.detach().numpy(), b.item()


def probs(w, b, X):
    p1 = 1 / (1 + np.exp(-(X @ w + b)))
    return np.stack([1 - p1, p1], 1)


def arms():
    out = {f"ridge C={c}": ("ridge", c) for c in RIDGE_C}
    out |= {f"dropout q={q}": ("dropout", q) for q in DROPOUT_Q}
    out |= {f"DCT lam={lam}": ("DCT", lam) for lam in DCT_LAM}
    out |= {f"DCT-masks lam={lam}": ("DCT-masks", lam) for lam in MASK_LAM}
    out["DCT-masks lam=5, 2 passes"] = ("DCT-masks-2", 5)
    out["bagging-5"] = ("bagging", 0)
    out["DCT lam=5 + bagging-5"] = ("DCT+bagging", 5)
    return out


def fit_one(family, param, X, y, rng, seed):
    if family == "ridge":
        m = LogisticRegression(C=param, max_iter=5000).fit(X, y)
        return m.coef_[0], m.intercept_[0]
    if family == "dropout":
        return fit_penalized(X, y, param, 0.0)
    if family == "DCT":
        return fit_penalized(X, y, Q_BASE, param)
    if family == "DCT-masks":
        return fit_masks(X, y, Q_BASE, param, 5, seed)
    if family == "DCT-masks-2":
        return fit_masks(X, y, Q_BASE, param, 2, seed)
    raise ValueError(family)


def predict_arm(family, param, X, y, Xte, rng, seed):
    if family in ("bagging", "DCT+bagging"):
        lam = param if family == "DCT+bagging" else 0.0
        ps = []
        for _ in range(5):
            boot = rng.integers(0, len(X), len(X))
            w, b = fit_penalized(X[boot], y[boot], Q_BASE, lam)
            ps.append(probs(w, b, Xte))
        return np.mean(ps, 0)
    w, b = fit_one(family, param, X, y, rng, seed)
    return probs(w, b, Xte)


def one_trial(scenario, n_half, trial):
    torch.set_num_threads(1)
    cov, beta = make_problem(scenario)
    rng = np.random.default_rng(10_000 * trial + n_half + (scenario == "top"))
    X1, y1 = sample(cov, beta, n_half, rng)
    X2, y2 = sample(cov, beta, n_half, rng)
    Xte, yte = sample(cov, beta, N_TEST, rng)
    rows = []
    for arm, (family, param) in arms().items():
        p1 = predict_arm(family, param, X1, y1, Xte, rng, 2 * trial)
        p2 = predict_arm(family, param, X2, y2, Xte, rng, 2 * trial + 1)
        rows.append(
            dict(
                scenario=scenario,
                n=n_half,
                arm=arm,
                family=family,
                param=param,
                trial=trial,
            )
            | pair_metrics(p1, p2, yte)
        )
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--out", type=Path, default=Path("results/linear.csv"))
    args = ap.parse_args()
    cells = [
        (s, n, t)
        for s in ("top", "bottom")
        for n in (100, 300)
        for t in range(args.trials)
    ]
    res = Parallel(n_jobs=N_JOBS)(delayed(one_trial)(*c) for c in cells)
    df = pd.DataFrame([r for rows in res for r in rows])
    args.out.parent.mkdir(exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    import linear_sim

    linear_sim.main()
