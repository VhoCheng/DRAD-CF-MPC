#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys
import traceback
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parent

RESULT = ROOT / "revision_experiments/results"

DEST = (
    ROOT / "revision_experiments/figures/TNSE_V7"
)

REAL = DEST / "REAL_EXP47_EXP42"
DEMO = DEST / "SYNTHETIC_DEMO_ONLY"

for folder in [REAL, DEMO]:
    folder.mkdir(parents=True, exist_ok=True)

C = {
    "ink": "#243D52",
    "muted": "#8A9AA6",
    "blue": "#356D94",
    "teal": "#18857F",
    "amber": "#CC944A",
    "rose": "#B95D64",
    "light": "#E4EDF0"
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.2,
    "axes.titlesize": 10.2,
    "axes.titleweight": "semibold",
    "axes.labelsize": 9.1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#BDCCD4",
    "axes.linewidth": .75,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white"
})

RNG = np.random.default_rng(20261023)


def read(path):
    path = RESULT / path

    if not path.exists():
        raise FileNotFoundError(
            f"Missing data: {path}"
        )

    return pd.read_csv(path)


def style(ax, axis="x"):
    ax.grid(
        axis=axis,
        color="#DFE8ED",
        linewidth=.65,
        alpha=.8
    )

    ax.set_axisbelow(True)
    ax.tick_params(
        length=3,
        color="#9BAEB8"
    )


def panel(ax, title):
    ax.set_title(
        title,
        loc="left",
        pad=15,
        color=C["ink"]
    )


def export(fig, name, demo=False):

    folder = DEMO if demo else REAL

    if demo:
        fig.suptitle(
            "SYNTHETIC DEMO | "
            "Illustration only, not experimental evidence",
            y=1.04,
            fontsize=12,
            color=C["rose"],
            fontweight="bold"
        )

    for ext in ["pdf", "svg", "png"]:
        fig.savefig(
            folder / f"{name}.{ext}",
            dpi=220,
            bbox_inches="tight",
            pad_inches=.23
        )

    plt.close(fig)
    print("[OK]", name, flush=True)


def interval(ax, pos, mean, low, high, color):

    mean = float(mean)
    low = float(low)
    high = float(high)

    ax.errorbar(
        mean,
        pos,
        xerr=[
            [max(0, mean-low)],
            [max(0, high-mean)]
        ],
        fmt="o",
        color=color,
        ms=5.5,
        lw=1.8,
        capsize=3,
        zorder=5
    )


def get_data():

    d = read(
        Path(
            "47_independent_closed_loop/full/"
            "paired_outcomes.csv"
        )
    )

    keys = [
        "pattern", "budget", "onset", "seed"
    ]

    wide = d.pivot(
        index=keys,
        columns="method",
        values="PAS"
    )

    if len(wide) != 400:
        raise ValueError(
            "Expected 400 paired scenarios."
        )

    return d, wide


# ===================================================
# REAL FIGURE 1
# TWO-SCALE FOREST PLOT
# ===================================================

def figure_1(wide):

    s = read(
        Path(
            "47_independent_closed_loop/full/"
            "paired_statistics.csv"
        )
    )

    s = (
        s[s.A == "FIXED_EP"]
        .set_index("B")
    )

    fig, axs = plt.subplots(
        1, 2,
        figsize=(13.4, 4.7),
        layout="constrained",
        gridspec_kw={
            "width_ratios": [1.18, .9]
        }
    )

    methods = [
        "CDP",
        "LEGACY_EP",
        "FIXED_EP_NO_DYNAMIC",
        "FIXED_EP_HEURISTIC_GUARD"
    ]

    names = [
        "Degree baseline (CDP)",
        "Legacy selector",
        "Without dynamic risk",
        "Heuristic Guard"
    ]

    for i, method in enumerate(methods):

        r = s.loc[method]

        interval(
            axs[0],
            i,
            r.gain_B_minus_A,
            r.CI_low,
            r.CI_high,
            C["blue"]
        )

    axs[0].set_yticks(
        range(len(methods)),
        names
    )

    axs[0].invert_yaxis()

    axs[0].axvline(
        0,
        color=C["rose"],
        linestyle="--",
        linewidth=1
    )

    axs[0].set_xlim(-.003, .003)

    axs[0].set_xlabel(
        "PAS(comparator) - PAS(Fixed EP)"
    )

    panel(
        axs[0],
        "(a) Near-zero effects | magnified axis"
    )

    style(axs[0])

    # Separate scale for NO_GUARD
    r = s.loc["FIXED_EP_NO_GUARD"]

    interval(
        axs[1],
        0,
        r.gain_B_minus_A,
        r.CI_low,
        r.CI_high,
        C["rose"]
    )

    axs[1].set_yticks(
        [0],
        ["Without Safe-Guard"]
    )

    axs[1].set_ylim(-.7, .7)
    axs[1].set_xlim(-.032, .006)

    axs[1].axvline(
        0,
        color=C["rose"],
        linestyle="--",
        linewidth=1
    )

    axs[1].set_xlabel(
        "PAS(comparator) - PAS(Fixed EP)"
    )

    panel(
        axs[1],
        "(b) Guard tradeoff | separate scale"
    )

    style(axs[1])

    export(
        fig,
        "R01_exp47_split_scale_forest"
    )


# ===================================================
# REAL FIGURE 2
# EFFECT + ACTION DIVERGENCE + DISTRIBUTION
# ===================================================

def figure_2(d, wide):

    methods = [
        "CDP",
        "FIXED_EP_NO_DYNAMIC",
        "FIXED_EP_NO_GUARD",
        "FIXED_EP_HEURISTIC_NO_GUARD"
    ]

    labels = [
        "CDP",
        "No dynamic",
        "No Guard",
        "Heuristic no Guard"
    ]

    diff = read(
        Path(
            "47_independent_closed_loop/full/"
            "first_action_divergence.csv"
        )
    ).set_index("method")

    fig, axs = plt.subplots(
        1, 3,
        figsize=(15.5, 4.6),
        layout="constrained",
        gridspec_kw={
            "width_ratios": [1.2, .78, 1.2]
        }
    )

    counts = []

    for method in methods:

        delta = wide[method] - wide["FIXED_EP"]

        counts.append([
            int((delta < -1e-12).sum()),
            int((abs(delta) <= 1e-12).sum()),
            int((delta > 1e-12).sum())
        ])

    counts = np.asarray(counts)

    y = np.arange(len(methods))
    left = np.zeros(len(methods))

    for j, (name, color) in enumerate([
        ("Comparator better", C["rose"]),
        ("Identical PAS", C["light"]),
        ("Comparator worse", C["teal"])
    ]):

        pct = counts[:, j] / 4.0

        axs[0].barh(
            y,
            pct,
            left=left,
            height=.52,
            color=color,
            edgecolor="white",
            linewidth=.7,
            label=name
        )

        for i, value in enumerate(pct):

            if value >= 10:
                axs[0].text(
                    left[i] + value/2,
                    i,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=C["ink"]
                )

        left += pct

    axs[0].set_yticks(y, labels)
    axs[0].invert_yaxis()
    axs[0].set_xlim(0, 100)

    axs[0].set_xlabel(
        "Share of 400 paired scenarios (%)"
    )

    panel(
        axs[0],
        "(a) Better / tied / worse"
    )

    style(axs[0])

    axs[0].legend(
        loc="upper center",
        bbox_to_anchor=(.5, -.23),
        ncol=2,
        frameon=False,
        fontsize=7.4
    )

    # Action divergence
    values = (
        diff.loc[
            methods,
            "first_action_difference_from_FIXED_EP"
        ] * 100
    )

    axs[1].hlines(
        y, 0, values,
        color="#CCD9DE",
        linewidth=2.5
    )

    axs[1].scatter(
        values,
        y,
        s=65,
        color=[
            C["blue"], C["blue"],
            C["rose"], C["amber"]
        ],
        zorder=3
    )

    axs[1].set_yticks(
        y,
        [""] * len(y)
    )

    axs[1].invert_yaxis()
    axs[1].set_xlim(0, 110)

    for i, value in enumerate(values):

        axs[1].text(
            value + 2,
            i,
            f"{value:.1f}%",
            fontsize=8,
            va="center"
        )

    axs[1].set_xlabel(
        "First-action difference (%)"
    )

    panel(
        axs[1],
        "(b) Policy separation"
    )

    style(axs[1])

    # Paired distribution without cluttered lines
    deltas = [
        (
            wide[method] -
            wide["FIXED_EP"]
        ).to_numpy()
        for method in methods
    ]

    axs[2].boxplot(
        deltas,
        vert=False,
        showfliers=False,
        patch_artist=True,
        widths=.5,
        boxprops={
            "facecolor": "#D9E8EB",
            "edgecolor": C["blue"]
        },
        medianprops={
            "color": C["rose"],
            "linewidth": 1.7
        },
        whiskerprops={
            "color": C["muted"]
        },
        capprops={
            "color": C["muted"]
        }
    )

    axs[2].axvline(
        0,
        color=C["rose"],
        linestyle="--",
        linewidth=1
    )

    axs[2].set_yticks(
        range(1, 5),
        labels
    )

    axs[2].invert_yaxis()

    axs[2].set_xlabel(
        "PAS(comparator) - PAS(Fixed EP)"
    )

    panel(
        axs[2],
        "(c) Conditional difference spread"
    )

    style(axs[2])

    export(
        fig,
        "R02_exp47_action_divergence_and_ties"
    )

    # Save exact auditing statistics
    rows = []

    for i, method in enumerate(methods):

        rows.append({
            "method": method,
            "n": 400,
            "comparator_better": counts[i, 0],
            "equal": counts[i, 1],
            "comparator_worse": counts[i, 2],
            "first_action_diff_pct": values.iloc[i],
            "mean_delta": np.mean(deltas[i])
        })

    pd.DataFrame(rows).to_csv(
        REAL / "EXP47_AUDIT_STATS.csv",
        index=False
    )


# ===================================================
# REAL FIGURE 3
# EIGHT CONDITIONS WITH BOOTSTRAP CI
# ===================================================

def figure_3(wide):

    rows = []

    for key, group in wide.groupby(
        level=["pattern", "budget", "onset"],
        sort=True
    ):

        for method in [
            "CDP",
            "FIXED_EP_NO_GUARD"
        ]:

            delta = (
                group[method] -
                group["FIXED_EP"]
            ).to_numpy()

            # Within-condition seed bootstrap
            n = len(delta)
            indices = RNG.integers(
                0, n,
                size=(1500, n)
            )

            means = delta[indices].mean(axis=1)

            lo, hi = np.quantile(
                means,
                [.025, .975]
            )

            rows.append({
                "pattern": key[0],
                "budget": key[1],
                "onset": key[2],
                "method": method,
                "mean": delta.mean(),
                "low": lo,
                "high": hi
            })

    tab = pd.DataFrame(rows)

    tab.to_csv(
        REAL / "EXP47_CONDITIONAL_EFFECTS.csv",
        index=False
    )

    fig, axs = plt.subplots(
        1, 2,
        figsize=(13.5, 5.8),
        layout="constrained"
    )

    settings = tab[
        ["pattern", "budget", "onset"]
    ].drop_duplicates().reset_index(drop=True)

    configurations = [
        (
            "CDP",
            "CDP vs Fixed EP",
            C["blue"],
            (-.012, .012)
        ),
        (
            "FIXED_EP_NO_GUARD",
            "No Guard vs Fixed EP",
            C["rose"],
            (-.105, .035)
        )
    ]

    for j, (
        method, name, color, limits
    ) in enumerate(configurations):

        ax = axs[j]

        for i, setting in settings.iterrows():

            r = tab[
                (tab.pattern == setting.pattern) &
                (tab.budget == setting.budget) &
                (tab.onset == setting.onset) &
                (tab.method == method)
            ].iloc[0]

            interval(
                ax, i,
                r["mean"],
                r["low"],
                r["high"],
                color
            )

        names = [
            f"{r.pattern} | B={r.budget} T={r.onset}"
            for r in settings.itertuples()
        ]

        ax.set_yticks(
            range(8),
            names if j == 0 else [""]*8
        )

        ax.invert_yaxis()

        ax.axvline(
            0,
            color=C["muted"],
            linestyle="--",
            linewidth=1
        )

        ax.set_xlim(*limits)

        ax.set_xlabel(
            "Comparator PAS - Fixed EP PAS"
        )

        panel(
            ax,
            f"({chr(97+j)}) {name}"
        )

        style(ax)

    export(
        fig,
        "R03_exp47_eight_conditions_bootstrap"
    )


# ===================================================
# REAL FIGURE 4
# GUARD TRADEOFF UNDER 24 CONDITIONS
# ===================================================

def figure_4():

    d = read(
        Path(
            "42_final_guard_audit/"
            "42_guard_tradeoff_by_condition.csv"
        )
    )

    tail = read(
        Path(
            "42_final_guard_audit/"
            "42_tail_risk_and_regret.csv"
        )
    )

    pivot = d.pivot(
        index="condition",
        columns="case_size",
        values="guard_benefit"
    ).reindex(
        columns=[14, 57, 118, 300]
    )

    fig, axs = plt.subplots(
        1, 2,
        figsize=(13.7, 5.7),
        layout="constrained",
        gridspec_kw={
            "width_ratios": [1.1, 1.05]
        }
    )

    ax = axs[0]

    z = pivot.to_numpy(float)

    lim = max(
        abs(np.nanmin(z)),
        abs(np.nanmax(z)),
        .001
    )

    im = ax.imshow(
        z,
        norm=TwoSlopeNorm(
            vmin=-lim,
            vcenter=0,
            vmax=lim
        ),
        cmap="RdBu",
        aspect="auto"
    )

    ax.set_xticks(
        range(4),
        [14, 57, 118, 300]
    )

    ax.set_yticks(
        range(len(pivot)),
        [
            s.replace("_", " ")
            for s in pivot.index
        ]
    )

    for i in range(z.shape[0]):
        for j in range(z.shape[1]):

            value = z[i, j]

            ax.text(
                j, i,
                f"{value:+.3f}",
                ha="center",
                va="center",
                fontsize=7.7,
                color=(
                    "white"
                    if abs(value) > .7*lim
                    else C["ink"]
                )
            )

    ax.set_xlabel("IEEE power buses")

    panel(
        ax,
        "(a) NO_GUARD - FULL PAS"
    )

    fig.colorbar(
        im,
        ax=ax,
        fraction=.04,
        pad=.035,
        label="Positive favors Guard"
    )

    ax = axs[1]

    for method, color, marker in [
        ("CDP", C["muted"], "o"),
        ("FULL", C["teal"], "s"),
        ("NO_GUARD", C["rose"], "D")
    ]:

        q = tail[
            tail.method == method
        ].sort_values("case_size")

        ax.plot(
            q.mean_PAS,
            q.CVaR95_PAS,
            color=color,
            marker=marker,
            linewidth=1.5,
            markersize=5,
            label=method
        )

    ax.set_xlabel(
        "Mean PAS (lower is better)"
    )

    ax.set_ylabel(
        "CVaR95 PAS (lower is better)"
    )

    panel(
        ax,
        "(b) Mean-versus-tail tradeoff"
    )

    style(ax, "both")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(.5, -.19),
        ncol=3,
        frameon=False,
        fontsize=8
    )

    export(
        fig,
        "R04_exp42_guard_conditional_tradeoff"
    )


# ===================================================
# SYNTHETIC FIGURE 1
# NO DIAGONAL WATERMARK
# ===================================================

def demo_1():

    rng = np.random.default_rng(20261023)

    common = rng.normal(
        0, .095, 240
    )

    names = [
        "CDP",
        "Full",
        "No dynamic",
        "No Guard"
    ]

    colors = [
        C["muted"],
        C["teal"],
        C["blue"],
        C["rose"]
    ]

    values = []

    for mean, std in [
        (.48, .05),
        (.43, .035),
        (.46, .045),
        (.41, .12)
    ]:

        v = np.clip(
            mean + common +
            rng.normal(0, std, 240),
            0, 1
        )

        values.append(v)

    fig, (a, b) = plt.subplots(
        1, 2,
        figsize=(13, 4.8),
        layout="constrained"
    )

    for i, (v, color) in enumerate(
        zip(values, colors)
    ):

        violin = a.violinplot(
            [v],
            positions=[i],
            vert=False,
            showextrema=False,
            widths=.56
        )

        body = violin["bodies"][0]
        body.set_facecolor(color)
        body.set_alpha(.28)
        body.set_edgecolor("none")

        local_rng = np.random.default_rng(
            i + 4
        )

        pos = i + local_rng.uniform(
            -.12, .12, 90
        )

        a.scatter(
            v[:90],
            pos,
            s=6,
            color=color,
            alpha=.19,
            rasterized=True
        )

        q = np.quantile(
            v,
            [.25, .5, .75]
        )

        a.plot(
            [q[0], q[2]],
            [i, i],
            color=color,
            linewidth=5,
            solid_capstyle="round"
        )

        a.scatter(
            q[1],
            i,
            s=18,
            color="white",
            edgecolor=C["ink"],
            zorder=7
        )

        x = np.sort(v)

        b.step(
            x,
            np.arange(
                1, len(x)+1
            ) / len(x),
            where="post",
            color=color,
            linewidth=1.8,
            label=names[i]
        )

    a.set_yticks(
        range(4),
        names
    )

    a.invert_yaxis()

    a.set_xlabel(
        "Mock PAS loss (lower better)"
    )

    panel(
        a,
        "(a) Raincloud distributions"
    )

    style(a)

    b.set_xlabel("Mock PAS loss")

    b.set_ylabel(
        "Empirical cumulative fraction"
    )

    panel(
        b,
        "(b) ECDF: distribution and tail"
    )

    style(b, "both")

    b.legend(
        loc="upper center",
        bbox_to_anchor=(.5, -.17),
        ncol=4,
        frameon=False,
        fontsize=7.6
    )

    pd.DataFrame({
        name: value
        for name, value in zip(
            names, values
        )
    }).to_csv(
        DEMO / "SYNTHETIC_SAMPLES.csv",
        index=False
    )

    export(
        fig,
        "SIM01_clean_raincloud_ECDF",
        demo=True
    )


# ===================================================
# SYNTHETIC FIGURE 2
# PARAMETER LANDSCAPE
# ===================================================

def demo_2():

    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)

    xx, yy = np.meshgrid(x, y)

    z = (
        .04
        - .1*(xx-.5)**2
        - .07*(yy-.36)**2
        + .002*np.sin(7*xx)*np.cos(6*yy)
    )

    fig, (a, b) = plt.subplots(
        1, 2,
        figsize=(13, 4.75),
        layout="constrained"
    )

    im = a.contourf(
        xx, yy, z,
        levels=18,
        cmap="YlGnBu"
    )

    a.contour(
        xx, yy, z,
        levels=8,
        colors="white",
        alpha=.6,
        linewidths=.7
    )

    a.scatter(
        [.5],
        [.36],
        marker="x",
        color=C["rose"],
        s=80,
        linewidths=2
    )

    a.set_xlabel(
        "Synthetic gamma"
    )

    a.set_ylabel(
        "Synthetic kappa"
    )

    panel(
        a,
        "(a) Hypothetical performance surface"
    )

    fig.colorbar(
        im,
        ax=a,
        shrink=.86,
        label="Assumed PAS gain"
    )

    for kappa, color in [
        (.2, C["blue"]),
        (.36, C["teal"]),
        (.65, C["amber"])
    ]:

        curve = (
            .04
            - .1*(x-.5)**2
            - .07*(kappa-.36)**2
            + .002*np.sin(7*x)*np.cos(6*kappa)
        )

        b.plot(
            x, curve,
            color=color,
            linewidth=2,
            label=f"kappa = {kappa:.2f}"
        )

    b.axhline(
        0,
        color=C["muted"],
        linestyle="--"
    )

    b.set_xlabel(
        "Synthetic gamma"
    )

    b.set_ylabel(
        "Assumed PAS gain"
    )

    panel(
        b,
        "(b) Parameter profile and plateau"
    )

    style(b, "both")

    b.legend(
        loc="upper center",
        bbox_to_anchor=(.5, -.19),
        ncol=3,
        frameon=False,
        fontsize=7.5
    )

    export(
        fig,
        "SIM02_clean_parameter_landscape",
        demo=True
    )


# ===================================================
# RUN ALL
# ===================================================

if __name__ == "__main__":

    errors = []

    d, wide = get_data()

    jobs = [
        ("Exp47 forest", figure_1, (wide,)),
        ("Exp47 action divergence", figure_2, (d, wide)),
        ("Exp47 conditional bootstrap", figure_3, (wide,)),
        ("Exp42 Guard tradeoff", figure_4, ()),
        ("Synthetic raincloud", demo_1, ()),
        ("Synthetic parameter landscape", demo_2, ())
    ]

    for name, function, args in jobs:

        print("[RUN]", name, flush=True)

        try:
            function(*args)
        except Exception as exc:
            traceback.print_exc()
            errors.append((name, str(exc)))

    print("\nOUTPUT:", DEST)
    print("ERRORS:", errors)

    if errors:
        sys.exit(2)

    print("ALL FIGURES GENERATED SUCCESSFULLY")
