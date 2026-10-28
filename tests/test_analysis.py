import numpy as np
import pandas as pd
import pytest

from analysis import interp_at, matched_frontier


def test_frontier_staircase():
    # Points: (acc, y). (0.70, 0.90) dominates (0.65, 0.80), which drops out;
    # the frontier interpolates between the remaining points, is flat to the
    # left of the least accurate one, and is undefined beyond the most accurate.
    acc = np.array([0.60, 0.65, 0.70, 0.80])
    y = np.array([0.95, 0.80, 0.90, 0.70])
    assert interp_at(acc, y, 0.65) == pytest.approx(0.925)
    assert interp_at(acc, y, 0.75) == 0.80
    assert interp_at(acc, y, 0.55) == 0.95
    assert np.isnan(interp_at(acc, y, 0.85))


def test_matched_frontier_recovers_known_gap():
    # Sweep arms a, b trade accuracy for agreement: (0.6, 0.9) and (0.8, 0.7).
    # The target sits at accuracy 0.7 with agreement 0.85, i.e. +0.05 above
    # the line between them. Trial noise is paired (same shift in all arms).
    rng = np.random.default_rng(0)
    rows = []
    for t in range(40):
        e = rng.normal(0, 0.01)
        for arm, acc, ag in (("a", 0.6, 0.9), ("b", 0.8, 0.7), ("x", 0.7, 0.85)):
            rows.append(dict(trial=t, arm=arm, acc=acc + e, agree=ag + e))
    est, se, cover = matched_frontier(pd.DataFrame(rows), ["a", "b"], "x", n_boot=200)
    assert abs(est - 0.05) < 0.005
    assert se < 0.01 and cover == 1.0
