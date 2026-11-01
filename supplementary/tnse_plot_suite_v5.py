# -*- coding: utf-8 -*-
"""TNSE V5: publication-style plots from archived experiments."""

from pathlib import Path
import json
import traceback
import csv

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
import networkx as nx

ROOT = Path.cwd()
EXP = ROOT / "revision_experiments"
R = EXP / "results"
W = EXP / "topology_workspaces"
OUT = EXP / "figures" / "TNSE_V5_PUBLICATION"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#21364C"
BLUE = "#356C93"
TEAL = "#178585"
AMBER = "#C78A39"
RED = "#B95555"
SLATE = "#8498A8"
PALE = "#D8E4E9"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#AEBBC4",
    "axes.linewidth": .7,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.facecolor": "white",
    "savefig.facecolor": "white"
})

MANIFEST = []


def read(pattern):
    fs = sorted(R.glob(pattern))
    if len(fs) != 1:
        raise FileNotFoundError(
            f"{pattern}: expected 1, found {len(fs)}"
        )
    print("SOURCE:", fs[0].relative_to(ROOT), flush=True)
    return pd.read_csv(fs[0])


def lab(ax, text):
    ax.set_title(
        text, loc="left",
        fontweight="bold",
        color=INK, pad=13
    )


def grid(ax, axis="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color="#E4EAEE", lw=.6)
    ax.tick_params(
        axis="both", length=2, color=SLATE
    )


def save(fig, name, source):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(
            OUT / f"{name}.{ext}",
            dpi=270,
            bbox_inches="tight",
            pad_inches=.13
        )
    plt.close(fig)
    MANIFEST.append((name, source))
    print("[SAVED]", name, flush=True)


def interval(ax, x, y, lo, hi, color=TEAL):
    if not np.all(np.isfinite([x, y, lo, hi])):
        return
    ax.errorbar(
        x, y,
        xerr=[
            [max(0, x-lo)],
            [max(0, hi-x)]
        ],
        fmt="o", ms=4.5,
        color=color, ecolor=color,
        elinewidth=1.6,
        capsize=2.5, zorder=3
    )


def heat(ax, pivot, text, vcenter=None,
         fmt=".2f", cmap=None):
    z = pivot.to_numpy(dtype=float)
    kw = {}

    if vcenter is not None and np.isfinite(z).any():
        mx = max(
            abs(np.nanmin(z)-vcenter),
            abs(np.nanmax(z)-vcenter),
            1e-8
        )
        kw["norm"] = TwoSlopeNorm(
            vmin=vcenter-mx,
            vcenter=vcenter,
            vmax=vcenter+mx
        )

    im = ax.imshow(
        z,
        cmap=cmap or (
            "RdBu_r" if vcenter is not None
            else "YlGnBu"
        ),
        aspect="auto", **kw
    )

    ax.set_xticks(
        np.arange(len(pivot.columns)),
        [
            str(x).replace("_", " ")
            for x in pivot.columns
        ]
    )
    ax.set_yticks(
        np.arange(len(pivot.index)),
        [
            str(x).replace("_", " ")
            for x in pivot.index
        ]
    )
    ax.tick_params(axis="x", labelrotation=20)
    lab(ax, text)

    if z.size <= 35:
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                if np.isfinite(z[i, j]):
                    normv = im.norm(z[i, j])
                    ax.text(
                        j, i,
                        format(z[i, j], fmt),
                        ha="center",
                        va="center",
                        fontsize=7.1,
                        color=(
                            "white"
                            if normv > .75 or normv < .18
                            else INK
                        )
                    )
    return im


# ======================================================
# A01 — REAL BA / ER / WS NETWORK TOPOLOGY
# ======================================================

def figure_a():
    fig = plt.figure(
        figsize=(13.8, 7.2),
        layout="constrained"
    )
    gs = GridSpec(
        2, 3, figure=fig,
        height_ratios=[1.55, 1],
        hspace=.15, wspace=.12
    )

    for j, t in enumerate(("BA", "ER", "WS")):
        ff = sorted(
            W.glob(
                f"06_topology_full_*/{t}_seed42/graph.json"
            )
        )
        if len(ff) != 1:
            raise FileNotFoundError(
                f"{t}: graph.json"
            )

        obj = json.loads(ff[0].read_text())
        g = nx.Graph()
        g.add_nodes_from(range(obj["nodes"]))
        g.add_edges_from(obj["edge_list"])

        pos = nx.spring_layout(
            g, seed=19, iterations=95
        )
        deg = dict(g.degree())
        hubs = sorted(
            deg, key=deg.get, reverse=True
        )[:6]

        ax = fig.add_subplot(gs[0, j])
        ax.set_aspect("equal")
        ax.axis("off")

        nx.draw_networkx_edges(
            g, pos, ax=ax,
            edge_color="#91A6B5",
            width=.5, alpha=.37
        )

        oth = [v for v in g if v not in hubs]
        nx.draw_networkx_nodes(
            g, pos, ax=ax,
            nodelist=oth,
            node_size=[
                8+1.6*deg[v] for v in oth
            ],
            node_color=BLUE,
            alpha=.85, linewidths=0
        )

        nx.draw_networkx_nodes(
            g, pos, ax=ax,
            nodelist=hubs,
            node_size=[
                14+2.6*deg[v] for v in hubs
            ],
            node_color=AMBER,
            edgecolors="white",
            linewidths=.4
        )

        lab(
            ax,
            f"({chr(97+j)}) {t} network"
            f"  ·  {g.number_of_nodes()} nodes"
        )

        ax.text(
            .5, -.02,
            f"{g.number_of_edges()} links  |  "
            f"mean degree "
            f"{np.mean(list(deg.values())):.2f}",
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=8, color=INK
        )

        ax = fig.add_subplot(gs[1, j])
        lab(
            ax,
            f"({chr(100+j)}) Degree-tail distribution"
        )

        for seed, alpha in (
            (42, 1), (43, .55), (44, .4)
        ):
            fp = (
                ff[0].parent.parent /
                f"{t}_seed{seed}/graph.json"
            )
            obj2 = json.loads(fp.read_text())

            h = nx.Graph()
            h.add_nodes_from(range(obj2["nodes"]))
            h.add_edges_from(obj2["edge_list"])

            a = np.array([
                v for _, v in h.degree()
            ])
            x = np.arange(1, a.max()+1)

            ax.step(
                x,
                np.array([
                    (a >= v).mean()
                    for v in x
                ]),
                where="post",
                color=(
                    TEAL if seed == 42 else SLATE
                ),
                lw=1.9 if seed == 42 else .95,
                alpha=alpha,
                label=f"seed {seed}"
            )

        ax.set(
            xscale="log",
            yscale="log",
            xlabel="Node degree k",
            ylabel="P(K ≥ k)",
            ylim=(.006, 1.1)
        )
        ax.legend(
            frameon=False,
            fontsize=7, ncol=3,
            loc="lower left"
        )
        grid(ax, "both")

    save(
        fig,
        "A01_network_topologies_degree_tails",
        "Exp06 archived BA/ER/WS graphs"
    )


# ======================================================
# B02 — GENERALIZATION: THREE DIMENSIONS
# ======================================================

def figure_b():
    topo = read(
        "07_topology_audit_*_summary.csv"
    )
    prop = read(
        "15_transmission_full/mechanism_gain.csv"
    )
    scales = read(
        "38_same_state_full/paired_statistics.csv"
    )

    fig, axes = plt.subplots(
        1, 3, figsize=(14, 4.4),
        layout="constrained",
        gridspec_kw={
            "width_ratios": [1, 1.3, 1.05]
        }
    )

    a = axes[0]
    d = topo.set_index(
        "topology"
    ).reindex(["BA", "ER", "WS"])

    for j, (t, r) in enumerate(d.iterrows()):
        interval(
            a,
            float(r.relative_gain_pct), j,
            float(r.bootstrap_pct_CI_low),
            float(r.bootstrap_pct_CI_high),
            TEAL if r.relative_gain_pct > 0
            else RED
        )

    a.set_yticks(
        range(3), list(d.index)
    )
    a.invert_yaxis()
    a.axvline(
        0, color=SLATE,
        ls="--", lw=1
    )
    a.set_xlabel(
        "Legacy EP gain vs CDP (%)"
    )
    lab(a, "(a) Communication topology")
    grid(a, "x")

    a = axes[1]
    d = prop.set_index("mechanism")
    ys = np.arange(len(d))

    a.barh(
        ys-.17,
        d.Robust_vs_CDP_pct,
        height=.31,
        label="Robust vs CDP",
        color=TEAL
    )
    a.barh(
        ys+.17,
        d.Legacy_vs_CDP_pct,
        height=.31,
        label="Legacy vs CDP",
        color=AMBER
    )
    a.axvline(
        0, color=SLATE, lw=1
    )
    a.set_yticks(
        ys,
        [
            v.replace("_", " ")
            for v in d.index
        ]
    )
    a.invert_yaxis()
    a.legend(
        frameon=False, fontsize=7,
        loc="lower right"
    )
    a.set_xlabel(
        "Legacy EP improvement (%)"
    )
    lab(a, "(b) Propagation mechanisms")
    grid(a, "x")

    a = axes[2]
    d = scales[
        scales.comparison.str.contains(
            "vs CDP", regex=False
        )
    ].sort_values(
        ["case_size", "comparison"]
    )

    for i, n in enumerate(
        (14, 57, 118, 300)
    ):
        q = d[d.case_size == n]
        for _, r in q.iterrows():
            robust = "Robust" in r.comparison
            interval(
                a,
                float(r.mean_PAS_gain),
                i+(.11 if robust else -.11),
                float(r.CI_low),
                float(r.CI_high),
                TEAL if robust else AMBER
            )

    a.set_yticks(
        range(4),
        [
            f"IEEE {n}"
            for n in (14, 57, 118, 300)
        ]
    )
    a.invert_yaxis()
    a.axvline(
        0, color=SLATE,
        ls="--", lw=1
    )
    a.set_xlabel(
        "CDP − method PAS (95% CI)"
    )
    lab(
        a,
        "(c) Cross-scale paired evaluation"
    )
    grid(a, "x")

    save(
        fig,
        "B02_generalization_forest",
        "Exp07/15 legacy EP and Exp38 paired PAS"
    )


# ======================================================
# C03 — EXP47 VIOLIN, PAIRED DOTS, FOREST PLOT
# ======================================================

def figure_c():
    samples = read(
        "47_independent_closed_loop/"
        "full/paired_outcomes.csv"
    )
    stats = read(
        "47_independent_closed_loop/"
        "full/paired_statistics.csv"
    )

    ix = [
        "pattern", "budget",
        "onset", "seed"
    ]
    wide = samples.pivot_table(
        index=ix,
        columns="method",
        values="PAS",
        aggfunc="mean"
    )

    methods = [
        "CDP",
        "FIXED_EP",
        "FIXED_EP_NO_GUARD",
        "FIXED_EP_HEURISTIC_NO_GUARD"
    ]
    xwide = wide[methods].dropna()
    rng = np.random.default_rng(18)

    fig, axes = plt.subplots(
        1, 3,
        figsize=(14.2, 4.8),
        layout="constrained",
        gridspec_kw={
            "width_ratios": [
                1.05, 1.13, 1.12
            ]
        }
    )

    a = axes[0]
    values = [
        xwide[s].values for s in methods
    ]

    v = a.violinplot(
        values,
        positions=range(len(values)),
        showextrema=False,
        widths=.78
    )

    for body, c in zip(
        v["bodies"],
        [SLATE, TEAL, AMBER, RED]
    ):
        body.set_facecolor(c)
        body.set_alpha(.45)
        body.set_edgecolor("none")

    a.boxplot(
        values,
        positions=range(len(values)),
        widths=.17,
        showfliers=False,
        patch_artist=True,
        boxprops={
            "facecolor": "white",
            "edgecolor": INK
        },
        medianprops={"color": RED}
    )

    a.set_xticks(
        range(4),
        [
            "CDP",
            "FIXED\nEP",
            "No\nGuard",
            "Heuristic\nNo Guard"
        ]
    )
    a.set_ylabel(
        "Terminal PAS (lower is better)"
    )
    lab(
        a,
        "(a) Distribution across matched settings"
    )
    grid(a)

    a = axes[1]
    diffs = {
        "CDP vs Full":
            xwide["CDP"]-xwide["FIXED_EP"],
        "No Guard vs Full":
            xwide["FIXED_EP_NO_GUARD"]-
            xwide["FIXED_EP"],
        "Heuristic vs Full":
            xwide["FIXED_EP_HEURISTIC_NO_GUARD"]-
            xwide["FIXED_EP"]
    }

    for j, (k, s) in enumerate(
        diffs.items()
    ):
        arr = s.to_numpy()
        y = (
            j+rng.uniform(
                -.16, .16, size=len(arr)
            )
        )
        a.scatter(
            arr, y,
            s=6,
            color=[BLUE, AMBER, RED][j],
            alpha=.27,
            rasterized=True
        )
        a.scatter(
            [np.mean(arr)], [j],
            marker="D", s=50,
            color=INK, zorder=5
        )

    a.axvline(
        0, color=SLATE,
        ls="--", lw=1
    )
    a.set_yticks(
        range(3), list(diffs)
    )
    a.set_xlabel(
        "PAS(reference) − PAS(Fixed EP)"
    )
    lab(a, "(b) Matched PAS contrasts")
    grid(a, "x")

    a = axes[2]
    d = stats[
        stats.A == "FIXED_EP"
    ].sort_values(
        "gain_B_minus_A"
    ).reset_index(drop=True)

    for i, r in d.iterrows():
        interval(
            a,
            float(r.gain_B_minus_A), i,
            float(r.CI_low),
            float(r.CI_high),
            TEAL if r.gain_B_minus_A > 0
            else RED
        )

    a.set_yticks(
        range(len(d)),
        [
            s.replace(
                "FIXED_EP_", ""
            ).replace("FIXED_EP", "Full")
            for s in d.B
        ]
    )
    a.invert_yaxis()
    a.axvline(
        0, color=SLATE,
        ls="--", lw=1
    )
    a.set_xlabel(
        "PAS(B) − PAS(Fixed EP)"
    )
    lab(
        a,
        "(c) Predefined paired 95% intervals"
    )
    grid(a, "x")

    save(
        fig,
        "C03_exp47_matched_distribution",
        "Exp47 independent paired experiments"
    )


# ======================================================
# D04 — SAFE-GUARD MEAN / TAIL-RISK / CONDITION
# ======================================================

def figure_d():
    tail = read(
        "42_final_guard_audit/"
        "42_tail_risk_and_regret.csv"
    )
    guard = read(
        "42_final_guard_audit/"
        "42_guard_tradeoff_by_condition.csv"
    )

    fig, axes = plt.subplots(
        1, 3, figsize=(14.3, 4.55),
        layout="constrained",
        gridspec_kw={
            "width_ratios": [1, 1, 1.2]
        }
    )

    a = axes[0]
    for method, color in [
        ("CDP", SLATE),
        ("FULL", TEAL),
        ("NO_GUARD", AMBER)
    ]:
        q = tail[tail.method == method]
        a.scatter(
            q.mean_PAS,
            q.CVaR95_PAS,
            c=color,
            s=q.case_size*.28+32,
            alpha=.85,
            label=method,
            edgecolors="white",
            linewidths=.6
        )
        for r in q.itertuples():
            a.annotate(
                str(r.case_size),
                (
                    r.mean_PAS,
                    r.CVaR95_PAS
                ),
                fontsize=6,
                xytext=(3, 3),
                textcoords="offset points"
            )

    a.set(
        xlabel="Mean PAS (lower better)",
        ylabel="CVaR95 PAS (lower better)"
    )
    a.legend(
        frameon=False, fontsize=7
    )
    lab(a, "(a) Mean–tail risk landscape")
    grid(a, "both")

    a = axes[1]
    for method, color in [
        ("FULL", TEAL),
        ("NO_GUARD", AMBER)
    ]:
        q = tail[
            tail.method == method
        ].sort_values("case_size")

        a.plot(
            q.case_size,
            100*q.per_run_worse_than_CDP_rate,
            "-o", lw=1.9, ms=5,
            color=color,
            label=method
        )

    a.set_xticks([14, 57, 118, 300])
    a.set(
        xlabel="IEEE bus count",
        ylabel="Worse-than-CDP event rate (%)"
    )
    a.legend(
        frameon=False, fontsize=7
    )
    lab(a, "(b) Adverse-event frequency")
    grid(a)

    a = axes[2]
    p = guard.pivot(
        index="condition",
        columns="case_size",
        values="guard_benefit"
    ).reindex(
        columns=[14, 57, 118, 300]
    )

    im = heat(
        a, p,
        "(c) Guard benefit by perturbation",
        vcenter=0,
        fmt=".3f"
    )
    a.set_xlabel("IEEE test system")

    fig.colorbar(
        im, ax=a, shrink=.72,
        label="NO_GUARD PAS − FULL PAS"
    )

    save(
        fig,
        "D04_guard_tailrisk_boundary",
        "Exp42 guard tradeoff and tail risk"
    )


# ======================================================
# E05 — THREE-PANEL PARAMETER SENSITIVITY
# ======================================================

def figure_e():
    d = read(
        "02_safeguard_full_*_parameter_aggregate.csv"
    )

    pairs = [
        (
            "mean_EP_gain_pct",
            "(a) Legacy EP gain vs CDP (%)",
            0, ".1f"
        ),
        (
            "mean_switch_rate",
            "(b) Accepted-switch frequency",
            None, ".2f"
        ),
        (
            "mean_negative_PAS_rate",
            "(c) Negative PAS-gain frequency",
            None, ".2f"
        )
    ]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(13.7, 4.1),
        layout="constrained"
    )

    for ax, (
        col, labstr, ctr, fmt
    ) in zip(axes, pairs):
        p = d.pivot(
            index="kappa",
            columns="gamma",
            values=col
        ).sort_index(
            ascending=False
        )

        im = heat(
            ax, p, labstr,
            ctr, fmt
        )
        ax.set(
            xlabel="gamma (switch margin)",
            ylabel="kappa (conservativeness)"
        )
        fig.colorbar(
            im, ax=ax,
            fraction=.042, pad=.03
        )

    save(
        fig,
        "E05_hyperparameter_surfaces",
        "Exp02 legacy EP sensitivity"
    )


# ======================================================
# F06 — DELAYED AND IMPERFECT OBSERVATIONS
# ======================================================

def figure_f():
    d = read(
        "09_observation_audit_*_aggregate.csv"
    )
    q = read(
        "40_paired_observation_full/"
        "paired_stress_statistics.csv"
    )

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12.9, 8),
        layout="constrained"
    )

    metrics = [
        (
            "mean_delta_PAS",
            "(a) PAS degradation under sensing gaps",
            0, ".3f"
        ),
        (
            "policy_change_rate",
            "(b) Decision changes from sensing gaps",
            None, ".2f"
        ),
        (
            "PAS_worsening_rate_vs_clean",
            "(c) Fraction with worse PAS",
            None, ".2f"
        )
    ]

    for ax, (
        metric, label, mid, fmt
    ) in zip(axes.flat[:3], metrics):
        p = d.pivot(
            index="delay_steps",
            columns="missing_rate",
            values=metric
        ).sort_index(ascending=False)

        im = heat(
            ax, p, label,
            mid, fmt
        )
        ax.set(
            xlabel="Missing-observation fraction",
            ylabel="Observation delay (steps)"
        )
        fig.colorbar(
            im, ax=ax,
            fraction=.044, pad=.03
        )

    ax = axes[1, 1]
    p = q.pivot(
        index="condition",
        columns="case_size",
        values="delta_PAS_stress_minus_clean"
    ).reindex(
        columns=[14, 57, 118, 300]
    )

    im = heat(
        ax, p,
        "(d) Independently paired stress effect",
        0, ".3f"
    )
    ax.set_xlabel("IEEE test system")
    fig.colorbar(
        im, ax=ax,
        fraction=.044,
        pad=.03,
        label="Stress PAS − clean PAS"
    )

    save(
        fig,
        "F06_imperfect_observation_dashboard",
        "Exp09 and Exp40 observation stress"
    )


# ======================================================
# G07 — FOUR-PANEL SCALABILITY
# ======================================================

def figure_g():
    t = read(
        "35_37_final_evidence/"
        "37_serial_timing_summary.csv"
    ).sort_values("case_size")

    p = read(
        "38_same_state_full/paired_statistics.csv"
    )

    d = read(
        "35_37_final_evidence/"
        "35_physical_overview.csv"
    )

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12.7, 7.8),
        layout="constrained"
    )

    a = axes[0, 0]

    lines = [
        (
            "mean_stage1",
            "Stage 1 mean", BLUE, "-"
        ),
        (
            "p95_stage1",
            "Stage 1 P95", BLUE, "--"
        ),
        (
            "mean_stage2",
            "Stage 2 mean", AMBER, "-"
        ),
        (
            "p95_stage2",
            "Stage 2 P95", AMBER, "--"
        )
    ]

    for field, labstr, c, ls in lines:
        a.plot(
            t.case_size,
            t[field],
            ls,
            marker="o",
            ms=4,
            lw=1.8,
            color=c,
            label=labstr
        )

    a.set_xticks(t.case_size)
    a.set(
        xlabel="IEEE bus count",
        ylabel="Measured computation time (s)"
    )
    a.legend(
        ncol=2, fontsize=7,
        frameon=False
    )
    lab(
        a,
        "(a) Stage-wise runtime, including P95"
    )
    grid(a)

    a = axes[0, 1]
    v = t.mean_three_method_wall

    b = a.bar(
        t.case_size.astype(str),
        v,
        color=[
            SLATE, BLUE, TEAL, INK
        ],
        width=.6
    )
    a.bar_label(
        b, fmt="%.1f s",
        padding=4, fontsize=8
    )

    a.set_ylim(0, 1.22*max(v))
    a.set(
        xlabel="IEEE bus count",
        ylabel="Measured batch wall time (s)"
    )
    lab(a, "(b) Three-method batch timing")
    grid(a)

    a = axes[1, 0]
    nlist = [14, 57, 118, 300]

    for i, n in enumerate(nlist):
        part = p[
            (p.case_size == n) &
            p.comparison.str.contains(
                "vs CDP", regex=False
            )
        ]

        for r in part.itertuples():
            robust = "Robust" in r.comparison
            interval(
                a,
                r.mean_PAS_gain,
                i+(.11 if robust else -.11),
                r.CI_low,
                r.CI_high,
                TEAL if robust else AMBER
            )

    a.set_yticks(
        range(4),
        [f"IEEE {x}" for x in nlist]
    )
    a.invert_yaxis()
    a.axvline(
        0, color=SLATE,
        ls="--"
    )
    a.set_xlabel(
        "CDP PAS − method PAS (95% CI)"
    )
    lab(
        a,
        "(c) Independently paired PAS gain"
    )
    grid(a, "x")

    a = axes[1, 1]

    for method, c in [
        ("CDP", SLATE),
        ("Legacy-K5", AMBER),
        ("Confirm15-Robust", TEAL)
    ]:
        q = d[
            d.variant == method
        ].sort_values("case_size")

        if not q.empty:
            a.plot(
                q.case_size,
                100*q.mean_unserved,
                "-o",
                color=c, ms=4,
                lw=1.8,
                label=method
            )

    a.set_xticks(nlist)
    a.set(
        xlabel="IEEE bus count",
        ylabel="Unserved-load proxy (%)"
    )
    a.legend(
        frameon=False,
        fontsize=7
    )
    lab(
        a,
        "(d) Physical service impact proxy"
    )
    grid(a)

    save(
        fig,
        "G07_scalability_and_runtime",
        "Exp35/37/38; measured computation only"
    )


# ======================================================
# H08 — AC OPF AND PHYSICAL CONSTRAINT COVERAGE
# ======================================================

def figure_h():
    d = read(
        "45_acopf_corrective_dispatch/full_summary.csv"
    )
    r = read(
        "46_final_closeout/"
        "46_opf_recovery_summary.csv"
    )

    sizes = [14, 57, 118, 300]
    profiles = [
        "Morning",
        "Solar-Peak",
        "Evening-Peak"
    ]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12.9, 7.5),
        layout="constrained"
    )

    a = axes[0, 0]
    p = d.pivot(
        index="case_size",
        columns="profile",
        values="opf_success"
    ).reindex(
        index=sizes,
        columns=profiles
    ).astype(float)

    a.imshow(
        p, cmap="BrBG",
        vmin=0, vmax=1,
        aspect="auto"
    )
    a.set_xticks(
        range(3), profiles
    )
    a.set_yticks(
        range(4),
        [str(n) for n in sizes]
    )

    for i in range(4):
        for j in range(3):
            a.text(
                j, i,
                "solved"
                if p.iloc[i, j] > .5
                else "failed",
                ha="center", va="center",
                fontsize=8,
                weight="bold"
            )

    lab(
        a,
        "(a) AC OPF convergence by profile"
    )
    a.set_ylabel("IEEE buses")

    a = axes[0, 1]
    p = d.pivot(
        index="case_size",
        columns="profile",
        values="new_voltage_violations"
    ).reindex(
        index=sizes,
        columns=profiles
    )

    im = heat(
        a, p,
        "(b) Residual voltage-limit checks",
        None, ".0f"
    )
    fig.colorbar(
        im, ax=a,
        fraction=.04, pad=.025
    )

    a = axes[1, 0]
    z = d.drop_duplicates(
        "case_size"
    ).set_index(
        "case_size"
    ).reindex(sizes)

    x = np.arange(4)
    bottom = np.zeros(4)

    metrics = [
        (
            "branches_with_placeholder_RATING_A",
            AMBER, "Placeholder thermal rating"
        ),
        (
            "branches_with_nonplaceholder_RATING_A",
            TEAL, "Usable thermal rating"
        ),
        (
            "branches_without_RATING_A",
            SLATE, "Missing rating"
        )
    ]

    for col, co, label in metrics:
        y = z[col].to_numpy(dtype=float)

        a.bar(
            x, y,
            bottom=bottom,
            color=co,
            width=.55,
            label=label
        )
        bottom += y

    a.set_xticks(
        x, [str(n) for n in sizes]
    )
    a.set(
        xlabel="IEEE buses",
        ylabel="Branches"
    )
    a.legend(
        fontsize=7,
        frameon=False
    )
    lab(
        a,
        "(c) Thermal-rating evidence coverage"
    )
    grid(a)

    a = axes[1, 1]
    r = r.sort_values("case_size")
    yp = np.arange(len(r))

    pc = (
        100*r.load_curtailment_fraction.to_numpy()
    )

    for i, row in enumerate(r.itertuples()):
        c = [AMBER, RED][i]

        a.hlines(
            i, 0, pc[i],
            color=c, lw=4
        )
        a.scatter(
            pc[i], i,
            c=c, s=65,
            zorder=3
        )
        a.annotate(
            f"{pc[i]:.2f}%  /  "
            f"{row.curtailed_MW:,.1f} MW",
            (pc[i], i),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=8
        )

    a.set_yticks(
        yp,
        [
            f"IEEE {n}"
            for n in r.case_size
        ]
    )
    a.set_ylim(
        -.6, len(r)-.4
    )
    a.set_xlim(
        0,
        max(14.5, pc.max()*1.65)
    )
    a.set_xlabel(
        "Load curtailment for recovered point (%)"
    )
    lab(
        a,
        "(d) Exp46 feasible-point recovery"
    )
    grid(a, "x")

    save(
        fig,
        "H08_ACOPF_constraint_coverage",
        "Exp45/46; thermal constraints incomplete"
    )


# ======================================================
# I09 — MECHANISM ABLATION AND CONDITION EFFECT
# ======================================================

def figure_i():
    d = read(
        "39_mechanism_full/condition_summary.csv"
    )
    p = read(
        "39_mechanism_full/"
        "clean_paired_statistics.csv"
    )

    fig, axes = plt.subplots(
        1, 3,
        figsize=(13.8, 4.6),
        layout="constrained"
    )

    a = axes[0]
    q = d[
        d.condition == "clean"
    ].pivot(
        index="case_size",
        columns="method",
        values="mean_PAS"
    )

    for name1, c in [
        ("CDP", SLATE),
        ("FULL", TEAL),
        ("NO_CF", AMBER),
        ("NO_DYNAMIC", BLUE),
        ("NO_GUARD", RED)
    ]:
        if name1 in q:
            a.plot(
                q.index, q[name1],
                "-o",
                label=name1,
                ms=3.6,
                color=c,
                lw=1.7
            )

    a.set_xticks(q.index)
    a.set(
        xlabel="IEEE buses",
        ylabel="Mean PAS (lower better)"
    )
    a.legend(
        fontsize=7,
        frameon=False,
        ncol=2
    )
    lab(
        a,
        "(a) Ablation on clean observations"
    )
    grid(a)

    a = axes[1]
    q = d[
        d.case_size == 118
    ].pivot(
        index="condition",
        columns="method",
        values="mean_PAS"
    )

    q.plot.bar(
        ax=a,
        rot=35,
        width=.78,
        color=[
            SLATE, TEAL, AMBER,
            BLUE, RED
        ][:len(q.columns)]
    )
    a.set(
        xlabel="",
        ylabel="Mean PAS"
    )
    a.legend(
        frameon=False,
        fontsize=6.5,
        ncol=2
    )
    lab(
        a,
        "(b) Observation-condition sensitivity"
    )
    grid(a)

    a = axes[2]
    q = p[
        p.comparison == "FULL vs CDP"
    ].sort_values(
        "case_size"
    ).reset_index(drop=True)

    for i, r in q.iterrows():
        interval(
            a,
            float(r.absolute_gain),
            i,
            float(r.CI_low),
            float(r.CI_high),
            TEAL
        )

    a.set_yticks(
        range(len(q)),
        [
            f"IEEE {int(x)}"
            for x in q.case_size
        ]
    )
    a.invert_yaxis()
    a.axvline(
        0, color=SLATE,
        ls="--"
    )
    a.set_xlabel(
        "CDP PAS − FULL PAS (95% CI)"
    )
    lab(
        a,
        "(c) Clean paired improvement"
    )
    grid(a, "x")

    save(
        fig,
        "I09_mechanism_ablation",
        "Exp39; NO_CF is not fully independent"
    )


# ======================================================
# J10 — MEAN / P95 / CVaR95 TAIL-RISK CURVES
# ======================================================

def figure_j():
    d = read(
        "42_final_guard_audit/"
        "42_tail_risk_and_regret.csv"
    )

    methods = [
        "CDP", "FULL", "NO_GUARD"
    ]
    sizes = [14, 57, 118, 300]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(13.8, 4.1),
        layout="constrained"
    )

    metrics = [
        (
            "mean_PAS",
            "(a) Average impact"
        ),
        (
            "P95_PAS",
            "(b) P95 severity"
        ),
        (
            "CVaR95_PAS",
            "(c) Worst-tail mean (CVaR95)"
        )
    ]

    for ax, (metric, subtitle) in zip(
        axes, metrics
    ):
        for method, c in zip(
            methods,
            [SLATE, TEAL, AMBER]
        ):
            q = d[
                d.method == method
            ].sort_values("case_size")

            ax.plot(
                q.case_size,
                q[metric],
                "-o",
                color=c,
                lw=1.8,
                ms=5,
                label=method
            )

        ax.set_xticks(sizes)
        ax.set_xlabel("IEEE buses")
        ax.set_ylabel(
            "PAS (lower better)"
        )
        lab(ax, subtitle)
        grid(ax)

    axes[0].legend(
        frameon=False,
        ncol=3,
        fontsize=7
    )

    save(
        fig,
        "J10_guard_quantile_risk",
        "Exp42 mean/P95/CVaR95 across scales"
    )


# ======================================================
# MAIN — GENERATE ALL REAL EXPERIMENT FIGURES
# ======================================================

JOBS = [
    figure_a,
    figure_b,
    figure_c,
    figure_d,
    figure_e,
    figure_f,
    figure_g,
    figure_h,
    figure_i,
    figure_j
]

if __name__ == "__main__":

    if not R.exists():
        raise SystemExit(
            f"Missing experiment data: {R}"
        )

    failed = []

    for idx, func in enumerate(JOBS, 1):
        print(
            f"\n[{idx}/{len(JOBS)}] "
            f"{func.__name__}",
            flush=True
        )
        try:
            func()
        except Exception as e:
            traceback.print_exc()
            failed.append(
                (func.__name__, str(e))
            )

    manifest = OUT / "FIGURE_MANIFEST.tsv"

    with manifest.open(
        "w", encoding="utf8"
    ) as f:
        writer = csv.writer(
            f, delimiter="\t"
        )
        writer.writerow([
            "figure",
            "source_and_limitation"
        ])
        writer.writerows(MANIFEST)

    print("\n" + "="*65)
    print("TNSE V5 FIGURE GENERATION COMPLETED")
    print("="*65)

    print(
        "Successful figure groups:",
        len(MANIFEST)
    )
    print(
        "Failed:",
        failed
    )
    print(
        "Output:",
        OUT
    )

    if failed:
        raise SystemExit(1)
