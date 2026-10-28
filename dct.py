"""Dropout-consistency training (DCT) and the measurements used to judge it.

Loss for n stochastic forward passes z_1..z_n of the same batch (dropout on):
    mean_i CE(z_i, y) + lam * mean_{i<j} MSE(softmax z_i, softmax z_j)
With lam = 0 this is n-pass training at the same compute.

The pairwise term equals twice the (unbiased) variance of the n predictions, for
every n >= 2, so n changes the gradient noise of the penalty but not its expected
value. For a linear layer the mask variance of the logit is a data-weighted ridge
penalty on the weights (Wager, Wang & Liang 2013); Wei, Kakade & Ma (2020) give
the deep-net version. See the README for the formalization these functions check.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, dropout=0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


def pairwise_prob_mse(logits):
    probs = [F.softmax(z, dim=1) for z in logits]
    pairs = [
        F.mse_loss(probs[i], probs[j])
        for i in range(len(probs))
        for j in range(i + 1, len(probs))
    ]
    return torch.stack(pairs).mean()


def dct_loss(model, x, y, n_passes, lam):
    logits = [model(x) for _ in range(n_passes)]
    ce = torch.stack([F.cross_entropy(z, y) for z in logits]).mean()
    if lam > 0 and n_passes > 1:
        return ce + lam * pairwise_prob_mse(logits)
    return ce


def train(
    X, y, Xval, yval, *, n_passes, lam, weight_decay, dropout, seed,
    lr=2e-3, patience=5, max_epochs=100, batch_size=32,
):  # fmt: skip
    """Train the MLP with early stopping on validation cross-entropy.

    Returns the model at its best validation epoch and the number of epochs
    run. Early stopping (rather than a fixed epoch budget) is what makes arms
    with stronger penalties comparable: they converge more slowly. With
    `patience=None` the model trains for exactly `max_epochs` and the final
    weights are returned (the original notebook's protocol).
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = MLP(X.shape[1], output_dim=int(y.max()) + 1, dropout=dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, stale, epoch = np.inf, model.state_dict(), 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for idx in torch.randperm(len(X), generator=gen).split(batch_size):
            opt.zero_grad()
            dct_loss(model, X[idx], y[idx], n_passes, lam).backward()
            opt.step()
        if patience is None:
            continue
        model.eval()
        with torch.no_grad():
            val = F.cross_entropy(model(Xval), yval).item()
        if val < best - 1e-4:
            best, stale = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if patience is not None:
        model.load_state_dict(best_state)
    return model, epoch


@torch.no_grad()
def predict_proba(model, X):
    model.eval()
    return F.softmax(model(X), 1).numpy()


@torch.no_grad()
def mask_variance(model, X, n_masks=200, seed=0):
    """Per-point variance across dropout masks of the class-1 logit margin.

    This is the quantity DCT penalizes (up to the softmax Jacobian). Returned on
    the logit scale so that it can be compared with the across-model variance of
    the same margin without the shared [p(1-p)]^2 factor.
    """
    torch.manual_seed(seed)
    model.train()
    z = torch.stack([model(X) for _ in range(n_masks)])
    margin = z[..., 1] - z[..., 0]
    model.eval()
    return margin.var(0, unbiased=True).numpy()


def logit_margin(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p[:, 1]) - np.log(p[:, 0])


def pair_metrics(p1, p2, y):
    """Accuracy and agreement of two models, plus the checks on agreement.

    `kappa` is Cohen's kappa between the two label vectors: agreement above
    what the two models' own label frequencies would give by chance, so it does
    not reward both models collapsing toward the majority class.
    `excess_agree` is agreement above what the two accuracies would give if the
    models erred independently. `soft_agree` is exp(-symmetric KL), the old
    metric, which also rises when both models are merely less confident, so
    `confidence` is reported alongside it.
    """
    k = p1.shape[1]
    h1, h2 = p1.argmax(1), p2.argmax(1)
    a1, a2 = (h1 == y).mean(), (h2 == y).mean()
    agree = (h1 == h2).mean()
    indep = a1 * a2 + (1 - a1) * (1 - a2) / (k - 1)
    chance = sum((h1 == c).mean() * (h2 == c).mean() for c in range(k))
    eps = 1e-12
    lp1, lp2 = np.log(p1 + eps), np.log(p2 + eps)
    skl = 0.5 * ((p1 * (lp1 - lp2)).sum(1) + (p2 * (lp2 - lp1)).sum(1)).mean()
    return dict(
        acc=(a1 + a2) / 2,
        agree=agree,
        kappa=(agree - chance) / (1 - chance) if chance < 1 else np.nan,
        excess_agree=agree - indep,
        soft_agree=float(np.exp(-skl)),
        confidence=(p1.max(1).mean() + p2.max(1).mean()) / 2,
        pos_rate=((h1 == 1).mean() + (h2 == 1).mean()) / 2,
    )


def vote_metrics(hard, y, n_bins=10):
    """Vote-ensemble statistics from an (M, N) array of predicted labels."""
    k = int(max(hard.max(), y.max())) + 1
    counts = np.stack([(hard == c).sum(0) for c in range(k)], 1)
    return vote_metrics_from_counts(counts, y, n_bins)


def vote_metrics_from_counts(counts, y, n_bins=10):
    """Statistics of the vote ensemble over independently trained models.

    `counts[i, c]` is how many models voted class c on test point i; the number
    of models may differ across points. pi_c(x) is the share voting c. Jiang,
    Nagarajan, Baek & Kolter (2022): if the vote ensemble is class-aggregated
    calibrated, expected disagreement between two models equals expected test
    error, so agreement in excess of 1 - error measures vote-ensemble
    overconfidence.

    Returns the two sides of that identity (disagreement as the unbiased mean
    over model pairs, error as the mean over models), the class-aggregated
    calibration error, and a signed overconfidence: mean vote confidence minus
    accuracy of the majority vote.
    """
    counts = np.asarray(counts, dtype=float)
    M = counts.sum(1, keepdims=True)
    pi = counts / M
    same = (counts * (counts - 1)).sum(1) / (M[:, 0] * (M[:, 0] - 1))
    disagree = (1 - same).mean()
    hits = np.take_along_axis(counts, y[:, None], 1)[:, 0]
    error = 1 - (hits / M[:, 0]).mean()
    conf = pi.max(1)
    majority_acc = (pi.argmax(1) == y).mean()
    # Class-aggregated calibration: pool all (x, class) pairs, bin by pi_c(x).
    flat_pi = pi.ravel()
    flat_hit = (y[:, None] == np.arange(pi.shape[1])[None, :]).ravel().astype(float)
    bins = np.minimum((flat_pi * n_bins).astype(int), n_bins - 1)
    cace = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            cace += m.mean() * abs(flat_pi[m].mean() - flat_hit[m].mean())
    return dict(
        disagree_identity=disagree,
        error=error,
        cace=cace,
        overconfidence=conf.mean() - majority_acc,
    )
