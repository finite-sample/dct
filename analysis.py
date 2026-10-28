"""Turn the raw runs into the tables the README reports.

Reads results/<dataset>.csv, results/preds/<dataset>/trial_<t>.npz and
results/linear.csv. Writes:
  results/paired_vs_standard.csv   paired change against `standard`, mean +- SE
  results/matched_frontier.csv     agreement of each target arm minus the sweep's
                                   agreement interpolated at the target's accuracy
  results/seed_vs_sample.csv       disagreement split into seed-only and sample parts
  results/votes.csv                vote-ensemble identity, calibration, overconfidence
  results/targeting.csv            does the variance reduction land where mask
                                   variance was high? (diff-in-diff vs dropout 0.5)
  results/temperature.csv          hard agreement under temperature scaling
  results/data_equivalents.csv     agreement gains expressed as extra data
  results/linear_summary.csv       the linear sim's frontier and mapping tests
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from dct import logit_margin, pair_metrics, vote_metrics_from_counts
from experiment import ARMS, datasets

RES = Path("results")
REF = "standard"
TARGETS = [a for a in ARMS if a.startswith("DCT")] + ["bagging-5", "seed-ens-5"]
SWEEPS = {
    "dropout": [REF] + [a for a in ARMS if a.startswith("dropout")],
    "wd": [REF] + [a for a in ARMS if a.startswith("wd")],
}
SWEEPS["single-pass"] = SWEEPS["dropout"] + SWEEPS["wd"][1:]
METRICS = [
    "acc",
    "agree",
    "kappa",
    "excess_agree",
    "soft_agree",
    "confidence",
    "pos_rate",
    "epochs",
]


def load_runs():
    frames = [
        pd.read_csv(p) for p in sorted(RES.glob("*.csv")) if p.stem in datasets_names()
    ]
    return pd.concat(frames, ignore_index=True)


def datasets_names():
    names = ["High Noise", "Small Sample", "High Dim", "Breast Cancer", "Adult (3k)"]
    return [n for n in names if (RES / f"{n}.csv").exists()]


def paired(df, ref=REF, by=("dataset", "pair"), metrics=METRICS):
    """Mean and paired SE of (arm - ref) within each group, matched by trial."""
    rows = []
    for keys, g in df.groupby(list(by)):
        base = g[g.arm == ref].set_index("trial")[metrics]
        for arm, h in g.groupby("arm"):
            d = h.set_index("trial")[metrics].sub(base, axis=0)
            row = dict(zip(by, keys)) | dict(
                arm=arm, n_trials=int(d.notna().any(axis=1).sum())
            )
            for m in metrics:
                row[f"d_{m}"], row[f"se_{m}"] = d[m].mean(), d[m].sem()
            rows.append(row)
    return pd.DataFrame(rows)


def interp_at(acc, y, x):
    """y of a sweep's frontier at accuracy x.

    The frontier keeps the sweep points no other point beats on both accuracy
    and y, interpolates linearly between them, and is flat to the left of its
    most agreeing point: at a lower accuracy you could still pick that point.
    NaN beyond the sweep's highest accuracy, where nothing can be matched.
    """
    if len(acc) == 0:
        return np.nan
    o = np.argsort(-acc)
    acc, y = acc[o], y[o]
    keep = y >= np.maximum.accumulate(y)  # not beaten by any more accurate point
    acc, y = acc[keep][::-1], y[keep][::-1]
    if x > acc[-1]:
        return np.nan
    return float(np.interp(x, acc, y, left=y[0]))


def matched_frontier(df, sweep_arms, target, metric="agree", n_boot=1000, seed=0):
    """Target's agreement minus the sweep's agreement at the target's accuracy.

    Both sides are means over trials; the SE is a bootstrap over trials, so it
    carries the pairing. Accuracy matching uses the sweep's mean curve, not the
    nearest grid point.
    """
    yw = df.pivot(index="trial", columns="arm", values=metric)
    usable = set(yw.columns[yw.notna().mean() > 0.5])  # drop arms that collapsed
    arms = [a for a in sweep_arms if a in usable] + [target]
    if target not in usable or len(arms) < 3:
        return np.nan, np.nan, 0.0
    acc = df.pivot(index="trial", columns="arm", values="acc")[arms].values
    y = yw[arms].values
    keep = np.isfinite(acc).all(1) & np.isfinite(y).all(1)
    acc, y = acc[keep], y[keep]
    rng = np.random.default_rng(seed)

    def stat(rows):
        a, m = acc[rows].mean(0), y[rows].mean(0)
        return m[-1] - interp_at(a[:-1], m[:-1], a[-1])

    n = len(acc)
    est = stat(np.arange(n))
    boots = np.array([stat(rng.integers(0, n, n)) for _ in range(n_boot)])
    boots = boots[~np.isnan(boots)]
    se = boots.std() if len(boots) > n_boot // 2 else np.nan
    return est, se, len(boots) / n_boot


def frontier_table(df):
    rows = []
    for (ds, pair), g in df.groupby(["dataset", "pair"]):
        for sweep, arms in SWEEPS.items():
            for target in TARGETS:
                for metric in ("agree", "kappa"):
                    est, se, cover = matched_frontier(g, arms, target, metric)
                    rows.append(
                        dict(
                            dataset=ds,
                            pair=pair,
                            sweep=sweep,
                            target=target,
                            metric=metric,
                            delta=est,
                            se=se,
                            t=est / se if se else np.nan,
                            in_range=cover,
                        )
                    )
    return pd.DataFrame(rows)


def seed_vs_sample(df):
    """Disagreement of the seed-only pair and of the cross-sample pair, and the
    sample component (their difference), each as a paired change vs standard."""
    wide = df.pivot_table(
        index=["dataset", "arm", "trial"], columns="pair", values="agree"
    ).reset_index()
    wide["dis_seed"] = 1 - wide["seed"]
    wide["dis_sample"] = 1 - wide["sample"]
    wide["dis_sample_only"] = wide["dis_sample"] - wide["dis_seed"]
    wide["pair"] = "both"
    cols = ["dis_seed", "dis_sample", "dis_sample_only"]
    out = paired(wide, by=("dataset", "pair"), metrics=cols).drop(columns="pair")
    levels = wide.groupby(["dataset", "arm"])[cols].mean().reset_index()
    return out.merge(levels, on=["dataset", "arm"])


def test_indices(n, trial):
    return train_test_split(np.arange(n), test_size=0.3, random_state=trial)[1]


def load_preds(ds, n):
    """Per arm: lists over test points of per-model probabilities and mask variances.

    Each test point is seen by the models of every trial whose test set holds it
    (about 30% of trials x 2 halves), so the vote ensemble per point pools across
    trials. Only models A and B are used: they come from different halves.
    """
    files = sorted(
        (RES / "preds" / ds).glob("trial_*.npz"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    y = np.full(n, -1)
    probs = {a: [[] for _ in range(n)] for a in ARMS}
    mvar = {a: [[] for _ in range(n)] for a in ARMS}
    for f in files:
        z = np.load(f)
        idx = test_indices(n, int(f.stem.split("_")[1]))
        y[idx] = z["y"]
        for a in ARMS:
            for tag in ("A", "B"):
                p = z[f"{a}/{tag}/p"]
                for j, i in enumerate(idx):
                    probs[a][i].append(p[j])
                if f"{a}/{tag}/mv" in z.files:
                    mv = z[f"{a}/{tag}/mv"]
                    for j, i in enumerate(idx):
                        mvar[a][i].append(mv[j])
    seen = y >= 0
    return y, probs, mvar, seen


def vote_table(ds, y, probs, seen):
    rows = []
    for a in ARMS:
        counts = np.array(
            [
                np.bincount(np.argmax(probs[a][i], 1), minlength=2)
                for i in np.flatnonzero(seen)
            ]
        )
        ok = counts.sum(1) >= 2
        m = vote_metrics_from_counts(counts[ok], y[seen][ok])
        rows.append(
            dict(
                dataset=ds,
                arm=a,
                n_points=int(ok.sum()),
                models_per_point=counts[ok].sum(1).mean(),
            )
            | m
        )
    return pd.DataFrame(rows)


def point_stats(probs, mvar, seen):
    """Across-model variance of the logit margin, mean |margin|, mean mask variance."""
    var_s, margin, var_m = [], [], []
    for i in np.flatnonzero(seen):
        z = logit_margin(np.array(probs[i]))
        var_s.append(z.var(ddof=1) if len(z) > 1 else np.nan)
        margin.append(abs(z.mean()))
        var_m.append(np.mean(mvar[i]) if mvar[i] else np.nan)
    return np.array(var_s), np.array(margin), np.array(var_m)


def targeting_table(ds, probs, mvar, seen, n_bins=5):
    """M1. Does an arm's reduction in across-model variance (vs standard) land
    where baseline mask variance was high? Reduction is the log ratio of the two
    variances per point, so a uniform rescaling of the logits gives a constant
    and a correlation of zero; only targeted reduction gives a positive one.
    Reported within bins of baseline |margin| and for dropout and weight decay
    too, so DCT can be compared with them."""
    vs0, mg0, vm0 = point_stats(probs[REF], mvar[REF], seen)
    ok = ~np.isnan(vs0) & ~np.isnan(vm0)
    bins = pd.qcut(mg0[ok], n_bins, labels=False, duplicates="drop")
    rows = [
        dict(
            dataset=ds,
            arm=REF,
            bin="all",
            stat="rho(Var_S, Var_m)",
            value=spearmanr(vs0[ok], vm0[ok]).statistic,
        )
    ]
    for b in np.unique(bins):
        m = bins == b
        rows.append(
            dict(
                dataset=ds,
                arm=REF,
                bin=int(b),
                stat="rho(Var_S, Var_m)",
                value=spearmanr(vs0[ok][m], vm0[ok][m]).statistic,
            )
        )
    for a in ("DCT lam=5", "DCT lam=15", "dropout=0.5", "dropout=0.7", "wd=10"):
        vs, _, _ = point_stats(probs[a], mvar[a], seen)
        red = np.log(vs0[ok]) - np.log(vs[ok])  # positive = variance reduced
        rows.append(
            dict(
                dataset=ds,
                arm=a,
                bin="all",
                stat="rho(reduction, Var_m0)",
                value=spearmanr(red, vm0[ok]).statistic,
            )
        )
        rows.append(
            dict(
                dataset=ds,
                arm=a,
                bin="all",
                stat="mean log reduction",
                value=red.mean(),
            )
        )
        for b in np.unique(bins):
            m = bins == b
            rows.append(
                dict(
                    dataset=ds,
                    arm=a,
                    bin=int(b),
                    stat="rho(reduction, Var_m0)",
                    value=spearmanr(red[m], vm0[ok][m]).statistic,
                )
            )
    return pd.DataFrame(rows)


def temperature_table(ds, n):
    """M2. Rescale logits by T: hard agreement must be unchanged, soft must move."""
    rows = []
    for f in sorted((RES / "preds" / ds).glob("trial_*.npz")):
        z = np.load(f)
        pa, pb, y = z[f"{REF}/A/p"], z[f"{REF}/B/p"], z["y"]
        for T in (1.0, 0.5, 2.0):
            za, zb = logit_margin(pa) / T, logit_margin(pb) / T
            qa = 1 / (1 + np.exp(-za))
            qb = 1 / (1 + np.exp(-zb))
            m = pair_metrics(np.stack([1 - qa, qa], 1), np.stack([1 - qb, qb], 1), y)
            rows.append(
                dict(dataset=ds, T=T, agree=m["agree"], soft_agree=m["soft_agree"])
            )
    return pd.DataFrame(rows).groupby(["dataset", "T"]).mean().reset_index()


def data_equivalents(df):
    """M6. Fit agreement against log(training fraction) on the data arms and
    express every arm's agreement as the fraction of data that would produce it."""
    rows = []
    g = df[df.pair == "sample"].groupby(["dataset", "arm"])[["agree", "acc"]].mean()
    for ds in g.index.get_level_values(0).unique():
        fr = {"data=25%": 0.25, "data=50%": 0.5, "data=75%": 0.75, REF: 1.0}
        x = np.log([fr[a] for a in fr])
        ya = np.array([g.loc[(ds, a), "agree"] for a in fr])
        slope, icpt = np.polyfit(x, ya, 1)
        for a in ARMS:
            eq = np.exp((g.loc[(ds, a), "agree"] - icpt) / slope)
            rows.append(
                dict(
                    dataset=ds,
                    arm=a,
                    agree=g.loc[(ds, a), "agree"],
                    data_equivalent=eq,
                    slope_per_doubling=slope * np.log(2),
                )
            )
    return pd.DataFrame(rows)


def linear_summary():
    df = pd.read_csv(RES / "linear.csv")
    rows = []
    for (sc, n), g in df.groupby(["scenario", "n"]):
        sweeps = {
            "dropout": sorted(g[g.family == "dropout"].arm.unique()),
            "ridge": sorted(g[g.family == "ridge"].arm.unique()),
        }
        targets = [a for a in g.arm.unique() if a.startswith("DCT") or "bagging" in a]
        for sweep, arms in sweeps.items():
            for t in targets:
                for metric in ("agree", "kappa"):
                    est, se, cover = matched_frontier(g, arms, t, metric, n_boot=500)
                    rows.append(
                        dict(
                            scenario=sc,
                            n=n,
                            sweep=sweep,
                            target=t,
                            metric=metric,
                            delta=est,
                            se=se,
                            t=est / se if se else np.nan,
                            in_range=cover,
                        )
                    )
        # Mapping test: which dropout rate reproduces DCT lam=5's agreement?
        means = g.groupby("arm")[["acc", "agree", "confidence"]].mean()
        drop = g[g.family == "dropout"].groupby("param")[["acc", "agree"]].mean()
        target = means.loc["DCT lam=5"]
        order = np.argsort(drop.agree.values)
        q_eq = np.interp(
            target.agree, drop.agree.values[order], drop.index.values[order]
        )
        acc_at_q = np.interp(q_eq, drop.index.values, drop.acc.values)
        rows.append(
            dict(
                scenario=sc,
                n=n,
                sweep="dropout",
                target="DCT lam=5",
                metric="q_equivalent",
                delta=q_eq,
                se=np.nan,
                t=target.acc - acc_at_q,
                in_range=1.0,
            )
        )
    return pd.DataFrame(rows)


def main():
    import argparse

    global RES
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=RES)
    RES = ap.parse_args().results
    df = load_runs()
    paired(df).to_csv(RES / "paired_vs_standard.csv", index=False)
    frontier_table(df).to_csv(RES / "matched_frontier.csv", index=False)
    seed_vs_sample(df).to_csv(RES / "seed_vs_sample.csv", index=False)
    data_equivalents(df).to_csv(RES / "data_equivalents.csv", index=False)
    votes, targ, temp = [], [], []
    sizes = {k: len(v[1]) for k, v in datasets().items()}
    for ds in datasets_names():
        y, probs, mvar, seen = load_preds(ds, sizes[ds])
        votes.append(vote_table(ds, y, probs, seen))
        targ.append(targeting_table(ds, probs, mvar, seen))
        temp.append(temperature_table(ds, sizes[ds]))
        print(f"mechanism tables: {ds}", flush=True)
    pd.concat(votes).to_csv(RES / "votes.csv", index=False)
    pd.concat(targ).to_csv(RES / "targeting.csv", index=False)
    pd.concat(temp).to_csv(RES / "temperature.csv", index=False)
    if (RES / "linear.csv").exists():
        linear_summary().to_csv(RES / "linear_summary.csv", index=False)
    print("wrote results/*.csv")


if __name__ == "__main__":
    main()
