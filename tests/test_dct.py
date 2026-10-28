import numpy as np
import pytest
import torch
import torch.nn.functional as F

from dct import (
    MLP,
    dct_loss,
    logit_margin,
    mask_variance,
    pair_metrics,
    pairwise_prob_mse,
    vote_metrics,
)


def test_pairwise_mse_is_twice_variance_for_any_n():
    # Mean over unordered pairs of (a_i - a_j)^2 equals 2 * unbiased variance,
    # exactly, for every n >= 2. So n_passes changes only the gradient noise of
    # the penalty, not its expected value.
    torch.manual_seed(0)
    for n in (2, 5):
        probs = torch.rand(n, 8, 2)
        probs = probs / probs.sum(-1, keepdim=True)
        logits = [torch.log(p) for p in probs]
        var = probs.var(0, unbiased=True).mean()
        assert pairwise_prob_mse(logits).item() == pytest.approx(
            2 * var.item(), rel=1e-5
        )


def test_linear_dropout_variance_is_weighted_ridge():
    # For logit w'(m * x)/(1-q) with m_j ~ Bernoulli(1-q), the variance over
    # masks is q/(1-q) * sum_j w_j^2 x_j^2: a data-weighted L2 penalty.
    rng = np.random.default_rng(0)
    q, w, x = 0.3, rng.normal(size=6), rng.normal(size=6)
    masks = rng.random((200_000, 6)) > q
    logits = (masks * x) @ w / (1 - q)
    assert logits.var() == pytest.approx(q / (1 - q) * np.sum(w**2 * x**2), rel=0.02)


def test_delta_method_prob_variance():
    # Var_m(p) ~= [p(1-p)]^2 Var_m(z) when the logit noise is small.
    rng = np.random.default_rng(1)
    z0, sd = 1.2, 0.05
    z = z0 + sd * rng.normal(size=400_000)
    p = 1 / (1 + np.exp(-z))
    p0 = 1 / (1 + np.exp(-z0))
    assert p.var() == pytest.approx((p0 * (1 - p0)) ** 2 * sd**2, rel=0.02)


def test_lam_zero_is_mean_ce_over_passes():
    torch.manual_seed(0)
    model = MLP(4, dropout=0.0)
    x, y = torch.randn(10, 4), torch.randint(0, 2, (10,))
    assert dct_loss(model, x, y, 3, 0.0).item() == pytest.approx(
        F.cross_entropy(model(x), y).item(), rel=1e-6
    )


def test_mask_variance_matches_penalty_object():
    # The function used for the mechanism test measures the same thing the
    # loss penalizes: variance across masks of the model's output.
    torch.manual_seed(0)
    model = MLP(5, dropout=0.3)
    x = torch.randn(32, 5)
    mv = mask_variance(model, x, n_masks=2000)
    assert not model.training
    model.train()
    with torch.no_grad():
        z = torch.stack([model(x) for _ in range(2000)])
    direct = (z[..., 1] - z[..., 0]).var(0).numpy()
    assert np.corrcoef(mv, direct)[0, 1] > 0.95


def test_pair_metrics_hand_example():
    y = np.array([0, 1, 1, 0])
    p1 = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.7, 0.3]])
    p2 = np.array([[0.8, 0.2], [0.3, 0.7], [0.3, 0.7], [0.4, 0.6]])
    m = pair_metrics(p1, p2, y)
    assert m["acc"] == pytest.approx((0.75 + 0.75) / 2)
    assert m["agree"] == pytest.approx(0.5)
    assert m["excess_agree"] == pytest.approx(0.5 - (0.75**2 + 0.25**2))
    # labels h1 = 0,1,0,0 and h2 = 0,1,1,1: chance = .75*.25 + .25*.75 = .375
    assert m["kappa"] == pytest.approx((0.5 - 0.375) / (1 - 0.375))
    assert m["pos_rate"] == pytest.approx((0.25 + 0.75) / 2)


def test_hard_agreement_invariant_to_temperature_soft_is_not():
    rng = np.random.default_rng(0)
    z1, z2 = rng.normal(size=(2, 500)) * 2
    y = (rng.random(500) < 0.5).astype(int)

    def probs(z, T):
        p1 = 1 / (1 + np.exp(-z / T))
        return np.stack([1 - p1, p1], 1)

    base = pair_metrics(probs(z1, 1), probs(z2, 1), y)
    hot = pair_metrics(probs(z1, 3), probs(z2, 3), y)
    assert hot["agree"] == base["agree"] and hot["kappa"] == base["kappa"]
    assert hot["soft_agree"] > base["soft_agree"] + 0.05


def test_vote_identity_and_calibration_on_hand_example():
    # Two points, four models. Votes for class 1: 1/4 and 3/4.
    hard = np.array([[0, 1], [0, 1], [0, 1], [1, 0]])
    y = np.array([0, 1])
    m = vote_metrics(hard, y, n_bins=4)
    # Of the 6 model pairs at each point, 3 disagree.
    assert m["disagree_identity"] == pytest.approx(0.5)
    assert m["error"] == pytest.approx(2 / 8)
    # Vote confidence .75 at both points, majority vote right at both.
    assert m["overconfidence"] == pytest.approx(0.75 - 1.0)
    assert m["cace"] == pytest.approx(0.25)


def test_vote_identity_matches_pairwise_disagreement():
    # The counts formula is the mean disagreement over unordered model pairs.
    rng = np.random.default_rng(0)
    hard = (rng.random((200, 50)) < np.linspace(0.1, 0.9, 50)).astype(int)
    y = np.zeros(50, dtype=int)
    m = vote_metrics(hard, y)
    pairs = [
        (hard[i] != hard[j]).mean()
        for i in range(len(hard))
        for j in range(i + 1, len(hard))
    ]
    assert m["disagree_identity"] == pytest.approx(np.mean(pairs), rel=1e-9)


def test_logit_margin_inverts_softmax():
    z = np.array([-2.0, 0.5, 3.0])
    p1 = 1 / (1 + np.exp(-z))
    p = np.stack([1 - p1, p1], 1)
    assert np.allclose(logit_margin(p), z, atol=1e-5)
