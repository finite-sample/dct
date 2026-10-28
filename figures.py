"""Figures and README tables from results/*.csv.

results/frontier.png         MLP: paired change in accuracy vs agreement, per dataset
results/linear_frontier.png  linear sim: accuracy vs agreement for every sweep
results/mechanism.png        seed vs sample, vote overconfidence, targeting
results/tables.md            the README's tables; spliced into README.md between
                             <!-- tables:start --> and <!-- tables:end -->
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RES = Path("results")
DATASETS = ["Adult (3k)", "Breast Cancer", "High Noise", "Small Sample", "High Dim"]
# Categorical slots in fixed order; the same family keeps its hue in every figure.
C = {
    "dropout": "#2a78d6",
    "wd": "#eb6834",
    "DCT": "#1baf7a",
    "ensemble": "#eda100",
    "data": "#9a9893",
    "ridge": "#6250d6",
    "text": "#0b0b0b",
    "muted": "#52514e",
}
SWEEP = {
    "dropout": ["dropout=0.1", "standard", "dropout=0.5", "dropout=0.7"],
    "wd": ["standard", "wd=1", "wd=3", "wd=10"],
    "DCT": ["DCT lam=0.5", "DCT lam=1.5", "DCT lam=5", "DCT lam=15"],
    "data": ["data=25%", "data=50%", "data=75%", "standard"],
}
SHORT = {
    "DCT lam=5": "DCT λ=5",
    "DCT lam=15": "λ=15",
    "wd=10": "wd=10",
    "wd=3": "wd=3",
    "dropout=0.7": "q=0.7",
    "dropout=0.5": "q=0.5",
    "seed-ens-5": "seed ens.",
    "bagging-5": "bagging",
    "data=50%": "½ data",
    "DCT lam=5, 2 passes": "2 passes",
}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": "#c3c2b7",
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#e6e5e1",
        "grid.linewidth": 0.5,
        "xtick.color": C["muted"],
        "ytick.color": C["muted"],
        "text.color": C["text"],
        "axes.labelcolor": C["muted"],
        "legend.frameon": False,
        "legend.fontsize": 8,
    }
)


def pct(x):
    return 100 * x


def frontier_figure():
    p = pd.read_csv(RES / "paired_vs_standard.csv")
    p = p[p.pair == "sample"].set_index(["dataset", "arm"])
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.6), sharey=False)
    for ax, ds in zip(axes, DATASETS):
        g = p.loc[ds]
        for fam, arms in SWEEP.items():
            arms = [a for a in arms if a in g.index]
            x, y = pct(g.loc[arms, "d_acc"]), pct(g.loc[arms, "d_agree"])
            ls = "--" if fam == "data" else "-"
            ax.plot(
                x,
                y,
                ls,
                color=C[fam],
                lw=1.5,
                marker="o",
                ms=4,
                zorder=3,
                alpha=0.6 if fam == "data" else 1,
                label={
                    "dropout": "dropout rate",
                    "wd": "weight decay",
                    "DCT": "DCT λ (5 passes)",
                    "data": "less data",
                }[fam],
            )
            ax.errorbar(
                x,
                y,
                xerr=pct(g.loc[arms, "se_acc"]),
                yerr=pct(g.loc[arms, "se_agree"]),
                fmt="none",
                ecolor=C[fam],
                elinewidth=0.5,
                alpha=0.6,
                zorder=2,
            )
        for arm, mk in (
            ("bagging-5", "s"),
            ("seed-ens-5", "D"),
            ("DCT lam=5, 2 passes", "o"),
        ):
            r = g.loc[arm]
            fam = "DCT" if arm.startswith("DCT") else "ensemble"
            ax.errorbar(
                pct(r.d_acc),
                pct(r.d_agree),
                xerr=pct(r.se_acc),
                yerr=pct(r.se_agree),
                fmt=mk,
                color=C[fam],
                mfc="white" if arm.startswith("DCT") else C[fam],
                ms=5,
                elinewidth=0.5,
                zorder=4,
                label={
                    "bagging-5": "bagging-5",
                    "seed-ens-5": "seed ensemble-5",
                    "DCT lam=5, 2 passes": "DCT λ=5, 2 passes",
                }[arm],
            )
        for arm in ("DCT lam=5", "wd=10", "seed-ens-5", "bagging-5", "dropout=0.7"):
            if ds == "Breast Cancer":  # everything sits at the ceiling; labels collide
                break
            r = g.loc[arm]
            ax.annotate(
                SHORT[arm],
                (pct(r.d_acc), pct(r.d_agree)),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
                color=C["muted"],
            )
        ax.axhline(0, color="#c3c2b7", lw=0.6)
        ax.axvline(0, color="#c3c2b7", lw=0.6)
        ax.set_title(ds)
        ax.grid(True, zorder=0)
        lo = min(-2.5, pct(g.loc[SWEEP["data"][1:], "d_acc"].min()) - 0.5)
        ax.set_xlim(lo, max(2.5, pct(g.d_acc.max()) + 0.8))
        ax.set_ylim(
            min(-3, pct(g.loc["data=50%", "d_agree"]) - 1),
            max(5, pct(g.d_agree.max()) + 1),
        )
    axes[0].set_ylabel("Δ agreement vs standard (points)")
    axes[2].set_xlabel("Δ accuracy vs standard (points)")
    hd, lb = axes[0].get_legend_handles_labels()
    fig.legend(hd, lb, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        "Two models trained on disjoint halves, paired change against standard "
        "(dropout 0.3, early stopping); mean ± SE over 50 trials",
        y=1.02,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(RES / "frontier.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def linear_figure():
    df = pd.read_csv(RES / "linear.csv")
    g = (
        df.groupby(["scenario", "n", "family", "param", "arm"])[["acc", "agree"]]
        .mean()
        .reset_index()
    )
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.5))
    fams = [
        ("ridge", "ridge", "ridge (C)"),
        ("dropout", "dropout", "dropout (q)"),
        ("DCT", "DCT", "DCT λ, closed form"),
        ("DCT-masks", "DCT", "DCT λ, sampled masks"),
    ]
    for ax, (sc, n) in zip(
        axes.ravel(), [("top", 100), ("top", 300), ("bottom", 100), ("bottom", 300)]
    ):
        h = g[(g.scenario == sc) & (g.n == n)]
        for fam, col, label in fams:
            s = h[h.family == fam].sort_values("param")
            hollow = fam == "DCT-masks"
            ax.plot(
                pct(s.acc),
                pct(s.agree),
                "-" if not hollow else ":",
                color=C[col],
                lw=1.5,
                marker="o",
                ms=4,
                mfc="white" if hollow else C[col],
                label=label,
                zorder=3,
            )
        for arm, mk, label in (
            ("bagging-5", "s", "bagging-5 of dropout 0.3"),
            ("DCT-masks lam=5, 2 passes", "^", "DCT λ=5, 2 passes"),
        ):
            r = h[h.arm == arm].iloc[0]
            ax.plot(
                pct(r.acc),
                pct(r.agree),
                mk,
                color=C["ensemble" if "bag" in arm else "DCT"],
                ms=6,
                mfc="white",
                label=label,
                zorder=4,
            )
        for arm, txt in (
            ("DCT lam=5", "λ=5"),
            ("dropout q=0.7", "q=0.7"),
            ("dropout q=0.3", "q=0.3"),
            ("ridge C=0.1", "C=0.1"),
            ("DCT lam=0.5", "λ=0.5"),
        ):
            r = h[h.arm == arm].iloc[0]
            ax.annotate(
                txt,
                (pct(r.acc), pct(r.agree)),
                xytext=(4, -8),
                textcoords="offset points",
                fontsize=7,
                color=C["muted"],
            )
        ax.set_title(
            f"signal on the {'largest' if sc == 'top' else 'smallest'} "
            f"eigendirections, "
            f"n = {n} per half"
        )
        ax.grid(True, zorder=0)
        ax.set_xlabel("accuracy (%)")
        ax.set_ylabel("agreement (%)")
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(
        "Linear-logistic simulation: every sweep traces the same "
        "accuracy–agreement curve",
        y=1.0,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(RES / "linear_frontier.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def family(arm):
    if arm.startswith("DCT") or arm == "5 passes":
        return "DCT"
    if arm.startswith("wd"):
        return "wd"
    if arm.startswith("dropout") or arm == "standard":
        return "dropout"
    if arm.startswith("data"):
        return "data"
    return "ensemble"


def mechanism_figure():
    shown = [
        "dropout=0.5",
        "dropout=0.7",
        "wd=3",
        "wd=10",
        "DCT lam=1.5",
        "DCT lam=5",
        "DCT lam=15",
        "bagging-5",
        "seed-ens-5",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    # (a) seed-only vs cross-sample disagreement change
    s = pd.read_csv(RES / "seed_vs_sample.csv")
    s = s[s.arm.isin(shown)]
    ax = axes[0]
    lim = pct(max(abs(s.d_dis_seed).max(), abs(s.d_dis_sample).max())) + 0.5
    ax.plot([-lim, lim], [-lim, lim], color="#c3c2b7", lw=0.8, zorder=1)
    ax.axhline(0, color="#e6e5e1", lw=0.6)
    ax.axvline(0, color="#e6e5e1", lw=0.6)
    for fam in ("dropout", "wd", "DCT", "ensemble"):
        h = s[s.arm.map(family) == fam]
        ax.scatter(
            pct(h.d_dis_seed),
            pct(h.d_dis_sample),
            s=22,
            color=C[fam],
            zorder=3,
            edgecolor="white",
            lw=0.6,
            label={
                "dropout": "dropout rate",
                "wd": "weight decay",
                "DCT": "DCT",
                "ensemble": "ensembles",
            }[fam],
        )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Δ disagreement, same data / different seed (points)")
    ax.set_ylabel("Δ disagreement, different data (points)")
    ax.set_title(
        "(a) Which variance does each method remove?\n"
        "below the diagonal: sample variance also fell",
        fontsize=9,
    )
    ax.legend(loc="upper left")
    # (b) vote overconfidence vs agreement gain
    v = pd.read_csv(RES / "votes.csv").set_index(["dataset", "arm"])
    p = pd.read_csv(RES / "paired_vs_standard.csv")
    p = p[p.pair == "sample"].set_index(["dataset", "arm"])
    ax = axes[1]
    ax.axhline(0, color="#e6e5e1", lw=0.6)
    ax.axvline(0, color="#e6e5e1", lw=0.6)
    for fam in ("dropout", "wd", "DCT", "ensemble"):
        xs, ys = [], []
        for ds in DATASETS:
            for arm in shown:
                if family(arm) == fam:
                    xs.append(pct(p.loc[(ds, arm), "d_agree"]))
                    ys.append(
                        pct(
                            v.loc[(ds, arm), "overconfidence"]
                            - v.loc[(ds, "standard"), "overconfidence"]
                        )
                    )
        ax.scatter(xs, ys, s=22, color=C[fam], zorder=3, edgecolor="white", lw=0.6)
    allx = pct(p.loc[[(d, a) for d in DATASETS for a in shown], "d_agree"]).values
    ally = np.array(
        [
            pct(
                v.loc[(d, a), "overconfidence"]
                - v.loc[(d, "standard"), "overconfidence"]
            )
            for d in DATASETS
            for a in shown
        ]
    )
    b = np.polyfit(allx, ally, 1)
    xx = np.linspace(allx.min(), allx.max(), 2)
    ax.plot(xx, np.polyval(b, xx), color="#c3c2b7", lw=0.8, zorder=1)
    ax.annotate(
        f"slope {b[0]:.2f}",
        (xx[-1], np.polyval(b, xx[-1])),
        xytext=(-40, 4),
        textcoords="offset points",
        fontsize=7,
        color=C["muted"],
    )
    ax.set_xlabel("Δ agreement vs standard (points)")
    ax.set_ylabel("Δ vote-ensemble overconfidence (points)")
    ax.set_title(
        "(b) Agreement gains come with a more\nover-confident vote ensemble", fontsize=9
    )
    # (c) targeting
    t = pd.read_csv(RES / "targeting.csv")
    t = t[(t.bin == "all") & (t.stat == "rho(reduction, Var_m0)")]
    arms = ["dropout=0.5", "dropout=0.7", "wd=3", "wd=10", "DCT lam=5", "DCT lam=15"]
    arms = [a for a in arms if a in set(t.arm)]
    ax = axes[2]
    w = 0.8 / len(DATASETS)
    for i, ds in enumerate(DATASETS):
        vals = [t[(t.dataset == ds) & (t.arm == a)].value.iloc[0] for a in arms]
        ax.bar(
            np.arange(len(arms)) + (i - 2) * w,
            vals,
            w * 0.9,
            color=[C[family(a)] for a in arms],
            alpha=0.35 + 0.15 * i,
            zorder=3,
        )
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    ax.set_xticks(np.arange(len(arms)))
    ax.set_xticklabels([SHORT.get(a, a) for a in arms], fontsize=8)
    ax.set_ylabel("Spearman ρ(variance reduction, baseline mask variance)")
    ax.set_title(
        "(c) Is the variance reduction concentrated where\nmask variance was high? "
        "(one bar per dataset)",
        fontsize=9,
    )
    ax.grid(True, axis="y", zorder=0)
    fig.tight_layout()
    fig.savefig(RES / "mechanism.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fmt(d, se):
    return f"{100 * d:+.1f} ± {100 * se:.1f}"


def tables():
    out = []
    p = pd.read_csv(RES / "paired_vs_standard.csv")
    p = p[p.pair == "sample"].set_index(["dataset", "arm"])
    arms = [
        "dropout=0.5",
        "dropout=0.7",
        "wd=3",
        "wd=10",
        "5 passes",
        "DCT lam=1.5",
        "DCT lam=5",
        "DCT lam=15",
        "DCT lam=5, 2 passes",
        "bagging-5",
        "seed-ens-5",
        "data=50%",
    ]
    out.append(
        "**Table 1. Paired change against `standard`, two models on disjoint halves.** "
        "Percentage points, mean ± paired SE over 50 trials; "
        "Δ accuracy / Δ agreement.\n"
    )
    out.append("| Arm | " + " | ".join(DATASETS) + " |")
    out.append("|---|" + "---|" * len(DATASETS))
    for a in arms:
        cells = [
            f"{fmt(p.loc[(d, a), 'd_acc'], p.loc[(d, a), 'se_acc'])} / "
            f"{fmt(p.loc[(d, a), 'd_agree'], p.loc[(d, a), 'se_agree'])}"
            for d in DATASETS
        ]
        out.append(f"| `{a}` | " + " | ".join(cells) + " |")
    f = pd.read_csv(RES / "matched_frontier.csv")
    f = f[(f.pair == "sample") & (f.metric == "agree")].set_index(
        ["dataset", "sweep", "target"]
    )
    out.append(
        "\n**Table 2. Agreement relative to the single-pass frontier "
        "at the same accuracy.** "
        "The frontier is the best of the dropout-rate and weight-decay "
        "sweeps at each accuracy "
        "(upper envelope, linearly interpolated). Points, mean "
        "± bootstrap SE over trials; "
        "negative means the frontier does better.\n"
    )
    out.append("| Arm | " + " | ".join(DATASETS) + " |")
    out.append("|---|" + "---|" * len(DATASETS))
    for a in [
        "DCT lam=1.5",
        "DCT lam=5",
        "DCT lam=15",
        "DCT lam=5, 2 passes",
        "bagging-5",
        "seed-ens-5",
    ]:
        cells = []
        for d in DATASETS:
            r = f.loc[(d, "single-pass", a)]
            cells.append(
                "more accurate than any frontier point"
                if np.isnan(r.delta)
                else fmt(r.delta, r.se)
            )
        out.append(f"| `{a}` | " + " | ".join(cells) + " |")
    s = pd.read_csv(RES / "seed_vs_sample.csv").set_index(["dataset", "arm"])
    out.append(
        "\n**Table 3. Where the disagreement goes.** Change vs "
        "`standard` in disagreement between "
        "two seeds on the same half (seed) and between the two "
        "halves (sample), in points. "
        "`standard` row: levels.\n"
    )
    out.append("| Arm | " + " | ".join(DATASETS) + " |")
    out.append("|---|" + "---|" * len(DATASETS))
    cells = [
        f"{100 * s.loc[(d, 'standard'), 'dis_seed']:.1f} seed, "
        f"{100 * s.loc[(d, 'standard'), 'dis_sample']:.1f} sample"
        for d in DATASETS
    ]
    out.append("| `standard` (levels) | " + " | ".join(cells) + " |")
    for a in [
        "dropout=0.5",
        "wd=10",
        "DCT lam=5",
        "DCT lam=15",
        "seed-ens-5",
        "bagging-5",
    ]:
        cells = [
            f"{100 * s.loc[(d, a), 'd_dis_seed']:+.1f} seed, "
            f"{100 * s.loc[(d, a), 'd_dis_sample']:+.1f} sample"
            for d in DATASETS
        ]
        out.append(f"| `{a}` | " + " | ".join(cells) + " |")
    v = pd.read_csv(RES / "votes.csv").set_index(["dataset", "arm"])
    out.append(
        "\n**Table 4. The vote ensemble.** Over the ~30 models "
        "that saw each test point: pairwise "
        "disagreement, mean error, and overconfidence (mean vote "
        "share of the majority class "
        "minus majority-vote accuracy). Under a calibrated vote "
        "ensemble disagreement would equal "
        "error (Jiang et al. 2022).\n"
    )
    out.append("| Arm | " + " | ".join(DATASETS) + " |")
    out.append("|---|" + "---|" * len(DATASETS))
    for a in ["standard", "wd=10", "DCT lam=5", "seed-ens-5"]:
        cells = [
            f"{100 * v.loc[(d, a), 'disagree_identity']:.1f} "
            f"/ {100 * v.loc[(d, a), 'error']:.1f} / "
            f"{100 * v.loc[(d, a), 'overconfidence']:+.1f}"
            for d in DATASETS
        ]
        out.append(f"| `{a}` | " + " | ".join(cells) + " |")
    ls = pd.read_csv(RES / "linear_summary.csv")
    out.append(
        "\n**Table 5. Linear simulation: DCT λ=5 against the "
        "sweeps.** Δ agreement at matched "
        "accuracy (points ± SE), and the dropout rate whose agreement "
        "equals DCT λ=5's, with "
        "the accuracy gap between them.\n"
    )
    out.append(
        "| Scenario, n | vs dropout sweep | vs ridge sweep | "
        "equivalent q | accuracy gap |"
    )
    out.append("|---|---|---|---|---|")
    for (sc, n), g in ls.groupby(["scenario", "n"]):
        q = g[(g.metric == "q_equivalent") & (g.target == "DCT lam=5")].iloc[0]
        dd = g[
            (g.metric == "agree") & (g.target == "DCT lam=5") & (g.sweep == "dropout")
        ].iloc[0]
        dr = g[
            (g.metric == "agree") & (g.target == "DCT lam=5") & (g.sweep == "ridge")
        ].iloc[0]
        out.append(
            f"| {sc}, {n} | {fmt(dd.delta, dd.se)} | {fmt(dr.delta, dr.se)} | "
            f"{q.delta:.2f} | {100 * q.t:+.1f} |"
        )
    fixed = RES / "fixed30" / "paired_vs_standard.csv"
    if fixed.exists():
        q = pd.read_csv(fixed)
        q = q[q.pair == "sample"].set_index(["dataset", "arm"])
        out.append(
            "\n**Table 6. Protocol check: fixed 30 epochs, no "
            "validation split (the original "
            "protocol).** Δ accuracy / Δ agreement vs `standard`, points.\n"
        )
        out.append("| Arm | " + " | ".join(DATASETS) + " |")
        out.append("|---|" + "---|" * len(DATASETS))
        for a in ["dropout=0.5", "wd=10", "DCT lam=5", "DCT lam=15", "seed-ens-5"]:
            cells = [
                f"{fmt(q.loc[(d, a), 'd_acc'], q.loc[(d, a), 'se_acc'])} / "
                f"{fmt(q.loc[(d, a), 'd_agree'], q.loc[(d, a), 'se_agree'])}"
                for d in DATASETS
            ]
            out.append(f"| `{a}` | " + " | ".join(cells) + " |")
    text = "\n".join(out) + "\n"
    (RES / "tables.md").write_text(text)
    readme = Path("README.md")
    if readme.exists() and "<!-- tables:start -->" in readme.read_text():
        new = re.sub(
            r"<!-- tables:start -->.*<!-- tables:end -->",
            "<!-- tables:start -->\n" + text + "<!-- tables:end -->",
            readme.read_text(),
            flags=re.S,
        )
        readme.write_text(new)


if __name__ == "__main__":
    frontier_figure()
    linear_figure()
    mechanism_figure()
    tables()
    print("wrote results/frontier.png, linear_frontier.png, mechanism.png, tables.md")
