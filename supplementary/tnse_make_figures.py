#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IEEE TNSE REVISION FIGURE GENERATOR

REAL:
    Fig2 - CPPS information flow
    Fig3 - Parameter sensitivity
    Fig4 - Exp47 paired ablation
    Fig5 - Network topology
    Fig6 - Computational scalability
    Fig7 - AC OPF physical validation

SYNTHETIC:
    Four illustrative figures
    NOT FOR PAPER SUBMISSION

All real numerical plots read existing experiments.
No experimental results are modified.
"""

from pathlib import Path
import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.patches import (
    FancyBboxPatch,
    FancyArrowPatch
)

ROOT = Path(__file__).resolve().parent

E = ROOT / "revision_experiments"
R = E / "results"

OUT = E / "figures" / "TNSE_camera_ready_new"
SYN = E / "figures" / "TNSE_SYNTHETIC_DEMO_ONLY"

OUT.mkdir(parents=True, exist_ok=True)
SYN.mkdir(parents=True, exist_ok=True)

NAVY = "#243B53"
TEAL = "#177E89"
ORANGE = "#D78A35"
RED = "#C74E4E"
GREY = "#8494A5"
BLUE = "#406B9B"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 240,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.edgecolor": "#BCC6CC",
    "axes.linewidth": 0.65,
    "xtick.color": "#3B4856",
    "ytick.color": "#3B4856",
    "text.color": NAVY,
    "axes.labelcolor": NAVY
})


def read(pattern):
    hits = sorted(R.glob(pattern))

    if len(hits) != 1:
        raise FileNotFoundError(
            f"Expected exactly one file: {pattern}\n"
            f"Found: {hits[:5]}"
        )

    return pd.read_csv(hits[0])


def finish(fig, name, directory=OUT):
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(
            directory / f"{name}.{ext}",
            bbox_inches="tight",
            pad_inches=0.13,
            dpi=240
        )

    plt.close(fig)
    print("[SAVED]", directory / f"{name}.pdf", flush=True)


def label(ax, text):
    ax.text(
        0, 1.045, text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
        fontsize=10
    )


# ============================================================
# FIGURE 2
# CPPS INFORMATION FLOW
# ============================================================

def figure2():

    fig, ax = plt.subplots(figsize=(11.2, 4.2))

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def node(x, y, w, h, text, fc="#EDF3F7"):

        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.12,rounding_size=0.13",
            linewidth=0.95,
            facecolor=fc,
            edgecolor="#A6BBC7"
        )

        ax.add_patch(patch)

        ax.text(
            x + w/2,
            y + h/2,
            text,
            ha="center",
            va="center",
            fontsize=8.7,
            linespacing=1.4
        )

    def arrow(x1, y1, x2, y2,
              color=GREY,
              linestyle="-",
              style="-|>"):

        patch = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style,
            mutation_scale=11,
            linewidth=1.2,
            color=color,
            linestyle=linestyle
        )

        ax.add_patch(patch)

    node(
        .30, 3.3, 2.10, .8,
        "CPPS / malware\nphysical process",
        "#E1EDF3"
    )

    node(
        .30, 1.9, 2.10, .65,
        "Fixed network/model\nparameters",
        "#F1F3F5"
    )

    node(
        3.1, 3.3, 2.05, .8,
        "Delayed / missing\nobservations",
        "#E1EDF3"
    )

    node(
        3.1, 1.9, 2.05, .65,
        "Uncertainty /\nmeasurement model",
        "#FFF1E6"
    )

    node(
        5.9, 3.3, 2.15, .8,
        "Estimated state\nand DRAD risk",
        "#DBEEF0"
    )

    node(
        5.9, 1.9, 2.15, .65,
        "Candidate defense\nactions",
        "#E8EEF7"
    )

    node(
        8.8, 3.3, 2.55, .8,
        "Counterfactual rollout\n(assumed model)",
        "#E8EEF7"
    )

    node(
        8.8, 1.9, 2.55, .65,
        "Safe-Guard comparison\nand chosen action",
        "#FFF0DD"
    )

    arrow(2.42, 3.7, 3.0, 3.7)
    arrow(5.28, 3.7, 5.8, 3.7)
    arrow(8.18, 3.7, 8.7, 3.7)

    arrow(7.0, 3.25, 7.0, 2.72)
    arrow(8.16, 2.24, 8.69, 2.24)

    arrow(3.0, 2.22, 5.78, 2.22)

    arrow(
        5.3, 2.45,
        5.88, 3.37,
        color=ORANGE
    )

    arrow(10.05, 3.23, 10.05, 2.72)

    arrow(
        8.75, 4.62,
        1.35, 4.62,
        color=BLUE,
        linestyle="--",
        style="<|-"
    )

    arrow(10.15, 1.84, 10.15, .85)
    arrow(10.15, .85, 1.35, .85)
    arrow(1.35, .85, 1.35, 3.19)

    ax.text(
        6.6, 4.72,
        "Model prediction is not a measurement of the future state",
        fontsize=8,
        color=BLUE,
        ha="center"
    )

    ax.text(
        6.05, .48,
        "Conceptual feedback flow; actual state-continuation "
        "interface requires separate verification.",
        fontsize=8,
        color="#57697A",
        ha="center"
    )

    finish(fig, "Fig2_information_flow_schematic")


# ============================================================
# FIGURE 3
# GAMMA-KAPPA AND WEIGHT SENSITIVITY
# ============================================================

def figure3():

    grid = read(
        "02_safeguard_full_*_parameter_aggregate.csv"
    )

    obj = read(
        "18_objective_weights_full/aggregate.csv"
    )

    risk = read(
        "19_dynamic_risk_full/aggregate.csv"
    )

    heat = grid.pivot(
        index="kappa",
        columns="gamma",
        values="mean_EP_gain_pct"
    ).sort_index()

    fig, axs = plt.subplots(
        1, 3,
        figsize=(12.6, 3.7),
        gridspec_kw={
            "width_ratios": [1.1, 1, 1.05]
        }
    )

    vmax = max(
        1,
        np.nanmax(np.abs(heat.values))
    )

    im = axs[0].imshow(
        heat.values,
        origin="lower",
        cmap="RdBu",
        aspect="auto",
        vmin=-vmax,
        vmax=vmax
    )

    axs[0].set_xticks(
        range(len(heat.columns)),
        [f"{v:g}" for v in heat.columns]
    )

    axs[0].set_yticks(
        range(len(heat.index)),
        [f"{v:g}" for v in heat.index]
    )

    axs[0].set_xlabel("Switching margin / gamma")
    axs[0].set_ylabel("Conservativeness / kappa")

    label(
        axs[0],
        "(a) 25-point gamma-kappa grid"
    )

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):

            v = heat.values[i, j]

            axs[0].text(
                j, i,
                f"{v:.1f}",
                ha="center",
                va="center",
                fontsize=6.7,
                color="white" if abs(v) > 10 else NAVY
            )

    fig.colorbar(
        im,
        ax=axs[0],
        fraction=.047,
        pad=.03,
        label="Legacy EP reduction vs CDP (%)"
    )

    datasets = [
        (axs[1], obj, "(b) Objective weights"),
        (axs[2], risk, "(c) Risk weights")
    ]

    for ax, df, subtitle in datasets:

        data = df.copy()

        data = data[
            data["variant"].astype(str) != "CDP"
        ]

        data = data.sort_values(
            "mean_EP_gain_pct"
        )

        values = data["mean_EP_gain_pct"]

        y = np.arange(len(data))

        ax.barh(
            y,
            values,
            color=[
                TEAL if v >= 0 else ORANGE
                for v in values
            ],
            height=.64
        )

        ax.set_yticks(
            y,
            data["variant"].astype(str)
        )

        ax.axvline(
            0,
            color=GREY,
            linewidth=.8
        )

        ax.set_xlabel(
            "Legacy EP reduction vs CDP (%)"
        )

        ax.tick_params(
            axis="y",
            labelsize=7
        )

        label(ax, subtitle)

    fig.text(
        .5, -.02,
        "Exp02 / Exp18 / Exp19. "
        "Legacy EP definition remains under audit.",
        ha="center",
        fontsize=8,
        color=RED
    )

    fig.tight_layout(w_pad=2.1)

    finish(
        fig,
        "Fig3_parameter_sensitivity_REPORTED_EP"
    )


# ============================================================
# FIGURE 4
# EXP47 REAL PAIRED ABLATION
# ============================================================

def figure4():

    stats = read(
        "47_independent_closed_loop/full/"
        "paired_statistics.csv"
    )

    outcomes = read(
        "47_independent_closed_loop/full/"
        "paired_outcomes.csv"
    )

    comparators = [
        "CDP",
        "LEGACY_EP",
        "FIXED_EP_NO_DYNAMIC",
        "FIXED_EP_NO_GUARD",
        "FIXED_EP_HEURISTIC_GUARD"
    ]

    data = stats[
        stats["A"] == "FIXED_EP"
    ].copy()

    data = (
        data.set_index("B")
        .loc[comparators]
        .reset_index()
    )

    # Source stores B minus A.
    # Invert sign to obtain FULL minus comparator.
    data["effect"] = -data["gain_B_minus_A"]
    data["lower"] = -data["CI_high"]
    data["upper"] = -data["CI_low"]

    fig, axs = plt.subplots(
        1, 2,
        figsize=(11.2, 3.85),
        gridspec_kw={
            "width_ratios": [1.24, 1]
        }
    )

    ax = axs[0]

    yy = np.arange(len(data))[::-1]

    for i, row in data.iterrows():

        val = row["effect"]

        errors = [
            [max(0, val-row["lower"])],
            [max(0, row["upper"]-val)]
        ]

        ax.errorbar(
            val,
            yy[i],
            xerr=errors,
            fmt="o",
            markersize=5,
            elinewidth=1.6,
            capsize=3,
            color=(
                TEAL
                if row["lower"] > 0
                else NAVY
            )
        )

    ax.axvline(
        0,
        color=RED,
        linestyle="--",
        linewidth=1
    )

    ax.set_yticks(
        yy,
        [
            x.replace("FIXED_EP_", "")
            for x in comparators
        ]
    )

    ax.set_xlabel(
        "Paired PAS: FIXED_EP minus comparator"
    )

    label(
        ax,
        "(a) Exp47 paired contrasts (n=400)"
    )

    ax.grid(
        axis="x",
        alpha=.15
    )

    # Mean absolute PAS
    selected = [
        "CDP",
        "FIXED_EP",
        "FIXED_EP_NO_GUARD",
        "FIXED_EP_HEURISTIC_NO_GUARD"
    ]

    summary = (
        outcomes.groupby("method")
        .agg(PAS=("PAS", "mean"))
        .reset_index()
    )

    summary = summary[
        summary["method"].isin(selected)
    ]

    summary = summary.sort_values("PAS")

    ax = axs[1]

    y = np.arange(len(summary))

    ax.barh(
        y,
        summary["PAS"],
        color=[GREY, TEAL, ORANGE, BLUE]
    )

    ax.set_yticks(
        y,
        [
            x.replace("FIXED_EP_", "")
            for x in summary["method"]
        ]
    )

    ax.set_xlabel("Mean PAS")

    ax.set_xlim(
        0,
        summary["PAS"].max()*1.22
    )

    label(
        ax,
        "(b) Absolute PAS levels"
    )

    for i, value in enumerate(summary["PAS"]):

        ax.text(
            value+.003,
            i,
            f"{value:.3f}",
            va="center",
            fontsize=8
        )

    fig.text(
        .5, -.02,
        "Exp47: paired comparisons; "
        "NO_DYNAMIC shows negligible measured difference.",
        ha="center",
        fontsize=8,
        color=RED
    )

    fig.tight_layout(w_pad=2.8)

    finish(
        fig,
        "Fig4_exp47_paired_ablation"
    )


# ============================================================
# FIGURE 5
# REAL BA / WS / ER TOPOLOGIES
# ============================================================

def figure5():

    import networkx as nx

    performance = read(
        "07_topology_audit_*_summary.csv"
    )

    structure = read(
        "07_topology_audit_*_structure.csv"
    )

    fig, axs = plt.subplots(
        2, 3,
        figsize=(11, 5.8),
        gridspec_kw={
            "height_ratios": [1.55, 1]
        }
    )

    base = E / "topology_workspaces"

    for i, topology in enumerate(
        ["BA", "WS", "ER"]
    ):

        files = sorted(
            base.glob(
                f"06_topology_full_*/"
                f"{topology}_seed42/graph.json"
            )
        )

        if not files:
            raise FileNotFoundError(
                f"Missing graph.json: {topology}"
            )

        info = json.loads(
            files[0].read_text()
        )

        graph = nx.Graph()

        graph.add_nodes_from(
            range(info["nodes"])
        )

        graph.add_edges_from(
            info["edge_list"]
        )

        pos = nx.spring_layout(
            graph,
            seed=4,
            iterations=75
        )

        ax = axs[0, i]

        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            width=.38,
            edge_color="#9EB0BC",
            alpha=.30
        )

        degree = np.array([
            graph.degree(n)
            for n in graph.nodes()
        ])

        nx.draw_networkx_nodes(
            graph,
            pos,
            ax=ax,
            node_size=5+1.4*degree,
            node_color=degree,
            cmap=plt.cm.Blues,
            vmin=0,
            vmax=35,
            linewidths=0
        )

        ax.axis("off")

        label(
            ax,
            f"({chr(97+i)}) {topology}"
        )

        row = structure[
            (structure["topology"] == topology)
            &
            (structure["graph_seed"] == 42)
        ].iloc[0]

        ax.text(
            .5, -.08,
            f"N={int(row['nodes'])}, "
            f"E={int(row['edges'])}, "
            f"C={row['average_clustering']:.3f}",
            ha="center",
            transform=ax.transAxes,
            fontsize=7.7
        )

    data = (
        performance.set_index("topology")
        .loc[["BA", "WS", "ER"]]
    )

    ax = axs[1, 0]

    ax.barh(
        ["BA", "WS", "ER"],
        data["relative_gain_pct"],
        color=[
            TEAL if x > 0 else ORANGE
            for x in data["relative_gain_pct"]
        ]
    )

    ax.axvline(
        0,
        color=GREY
    )

    ax.set_xlabel(
        "Legacy EP reduction (%)"
    )

    label(
        ax,
        "(d) Topology-dependent performance"
    )

    metrics = [
        ("average_clustering",
         "Clustering coefficient"),
        ("average_shortest_path",
         "Average shortest path")
    ]

    for j, (metric, ylabel) in enumerate(
        metrics, start=1
    ):

        grouped = (
            structure.groupby("topology")[metric]
            .agg(["mean", "std"])
            .reindex(["BA", "WS", "ER"])
        )

        ax = axs[1, j]

        ax.bar(
            ["BA", "WS", "ER"],
            grouped["mean"],
            color=[BLUE, TEAL, ORANGE],
            width=.6
        )

        ax.errorbar(
            range(3),
            grouped["mean"],
            yerr=grouped["std"],
            fmt="none",
            ecolor=NAVY,
            capsize=3,
            linewidth=1
        )

        ax.set_ylabel(ylabel)

        label(
            ax,
            "(e)" if j == 1 else "(f)"
        )

    fig.text(
        .5, -.018,
        "Network graphs are archived instances; "
        "legacy EP results require metric reconciliation.",
        color=RED,
        ha="center",
        fontsize=8
    )

    fig.tight_layout(
        w_pad=1.8,
        h_pad=2
    )

    finish(
        fig,
        "Fig5_archived_network_topologies"
    )


# ============================================================
# FIGURE 6
# REAL COMPUTATIONAL SCALABILITY
# ============================================================

def figure6():

    data = read(
        "35_37_final_evidence/"
        "37_serial_timing_summary.csv"
    )

    data = data.sort_values("case_size")

    x = data["case_size"].values

    fig, axs = plt.subplots(
        1, 2,
        figsize=(10.6, 3.4)
    )

    ax = axs[0]

    ax.plot(
        x,
        data["mean_stage1"],
        "-o",
        color=BLUE,
        linewidth=2,
        label="Stage 1 mean"
    )

    ax.plot(
        x,
        data["mean_stage2"],
        "-s",
        color=TEAL,
        linewidth=2,
        label="Stage 2 mean"
    )

    ax.plot(
        x,
        data["p95_stage1"],
        "--o",
        color=GREY,
        label="Stage 1 P95"
    )

    ax.plot(
        x,
        data["p95_stage2"],
        "--s",
        color=ORANGE,
        label="Stage 2 P95"
    )

    ax.set_xticks(x)
    ax.set_xlabel("Power-system nodes")
    ax.set_ylabel("Time (s)")

    ax.legend(
        frameon=False,
        fontsize=7,
        ncol=2
    )

    label(
        ax,
        "(a) Measured serial stage times"
    )

    ax = axs[1]

    values = data["mean_three_method_wall"]

    ax.bar(
        [str(v) for v in x],
        values,
        color=BLUE,
        width=.65
    )

    ax.set_xlabel("Power-system nodes")
    ax.set_ylabel("Batch wall time (s)")

    label(
        ax,
        "(b) Three-method batch cost"
    )

    for i, v in enumerate(values):

        ax.text(
            i,
            v+.2,
            f"{v:.1f}",
            fontsize=8,
            ha="center"
        )

    fig.text(
        .5, -.025,
        "Exp37: stage/batch measurements, "
        "not measured end-to-end communication latency.",
        ha="center",
        color=RED,
        fontsize=8
    )

    fig.tight_layout(w_pad=2.5)

    finish(
        fig,
        "Fig6_serial_computation_EXP37"
    )


# ============================================================
# FIGURE 7
# REAL AC OPF PHYSICAL VALIDATION
# ============================================================

def figure7():

    opf = read(
        "45_acopf_corrective_dispatch/"
        "full_summary.csv"
    )

    recovery = read(
        "46_final_closeout/"
        "46_opf_recovery_summary.csv"
    )

    fig, axs = plt.subplots(
        1, 3,
        figsize=(12, 3.6),
        gridspec_kw={
            "width_ratios": [1.05, 1.35, 1.02]
        }
    )

    # Solver outcomes
    grouped = (
        opf.groupby("case_size")["opf_success"]
        .agg(["sum", "count"])
    )

    ax = axs[0]

    names = [
        str(v)
        for v in grouped.index
    ]

    solved = grouped["sum"]
    failed = grouped["count"] - solved

    ax.bar(
        names,
        solved,
        color=TEAL,
        label="OPF solved"
    )

    ax.bar(
        names,
        failed,
        bottom=solved,
        color=ORANGE,
        label="Not solved"
    )

    ax.set_xlabel("IEEE case")
    ax.set_ylabel("Number of profiles")
    ax.set_ylim(0, 3.5)

    ax.legend(
        frameon=False,
        fontsize=7
    )

    label(
        ax,
        "(a) AC OPF solver outcomes"
    )

    # Actual branch thermal-limit coverage
    rating = (
        opf.sort_values("profile")
        .drop_duplicates("case_size")
        .sort_values("case_size")
    )

    sizes = [
        str(int(v))
        for v in rating["case_size"]
    ]

    usable = rating[
        "branches_with_nonplaceholder_RATING_A"
    ].astype(float).to_numpy()

    placeholder = rating[
        "branches_with_placeholder_RATING_A"
    ].astype(float).to_numpy()

    missing = rating[
        "branches_without_RATING_A"
    ].astype(float).to_numpy()

    ax = axs[1]

    ax.bar(
        sizes,
        placeholder,
        color=ORANGE,
        label="Placeholder ratings",
        width=.55
    )

    ax.bar(
        sizes,
        usable,
        bottom=placeholder,
        color=TEAL,
        label="Usable ratings",
        width=.55
    )

    ax.bar(
        sizes,
        missing,
        bottom=placeholder+usable,
        color=GREY,
        label="Missing",
        width=.55
    )

    ax.set_ylabel("Number of branches")
    ax.set_xlabel("IEEE case")

    ax.legend(
        frameon=False,
        fontsize=7
    )

    label(
        ax,
        "(b) Thermal-limit coverage"
    )

    # Recovery with load curtailment
    ax = axs[2]

    percentages = (
        100 * recovery["load_curtailment_fraction"]
    )

    ax.barh(
        range(len(recovery)),
        percentages,
        color=[ORANGE, RED]
    )

    ax.set_yticks(
        range(len(recovery)),
        [
            f"IEEE {r.case_size}\n{r.profile}"
            for r in recovery.itertuples()
        ]
    )

    ax.set_xlabel("Load curtailment (%)")

    label(
        ax,
        "(c) Exp46 recovery points"
    )

    ax.set_xlim(0, max(14, percentages.max()*1.3))

    for i, value in enumerate(percentages):

        ax.text(
            value+.3,
            i,
            f"{value:.2f}%",
            fontsize=8,
            va="center"
        )

    fig.text(
        .5, -.03,
        "OPF convergence and load-curtailment recovery "
        "do not establish complete thermal feasibility.",
        ha="center",
        fontsize=8,
        color=RED
    )

    fig.tight_layout(w_pad=2.0)

    finish(
        fig,
        "Fig7_ACOPF_scope_and_limits"
    )


# ============================================================
# SYNTHETIC TEACHING EXAMPLES
# NOT EXPERIMENTAL EVIDENCE
# ============================================================

def watermark(fig):

    fig.text(
        .5, .51,
        "SYNTHETIC - NOT FOR SUBMISSION",
        fontsize=21,
        color=RED,
        alpha=.17,
        rotation=21,
        ha="center",
        va="center",
        fontweight="bold"
    )

    fig.text(
        .5, .02,
        "Synthetic illustration only | "
        "Not derived from TNSE experimental results",
        ha="center",
        fontsize=9,
        color=RED
    )


def simulated_figures():

    rng = np.random.default_rng(20261022)

    # --------------------------------------------------------
    # SIM 01: IDEALIZED ROBUST PARAMETER REGION
    # --------------------------------------------------------

    gamma = np.linspace(0, 1, 9)
    kappa = np.linspace(0, 1, 9)

    gg, kk = np.meshgrid(gamma, kappa)

    values = (
        3.5
        - 14*(gg-.55)**2
        - 12*(kk-.45)**2
        + rng.normal(0, .12, (9, 9))
    )

    fig, axs = plt.subplots(
        1, 2,
        figsize=(9.7, 3.7)
    )

    im = axs[0].imshow(
        values,
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        cmap="BrBG"
    )

    fig.colorbar(
        im,
        ax=axs[0],
        label="Hypothetical improvement (%)"
    )

    axs[0].set_xlabel("gamma")
    axs[0].set_ylabel("kappa")

    label(
        axs[0],
        "(a) Illustrative parameter region"
    )

    x = np.linspace(0, 1, 31)

    mean = 2.6-7*(x-.47)**2

    axs[1].plot(
        x,
        mean,
        color=TEAL,
        linewidth=2
    )

    axs[1].fill_between(
        x,
        mean-.3,
        mean+.3,
        color=TEAL,
        alpha=.18
    )

    axs[1].axhline(
        0,
        color=GREY,
        linestyle="--"
    )

    axs[1].set_xlabel(
        "Normalized parameter"
    )

    axs[1].set_ylabel(
        "Hypothetical improvement (%)"
    )

    label(
        axs[1],
        "(b) Broad stable region"
    )

    watermark(fig)

    fig.tight_layout(
        rect=[0, .04, 1, 1]
    )

    finish(
        fig,
        "SIM_01_ideal_parameter_region",
        SYN
    )

    # --------------------------------------------------------
    # SIM 02: IDEALIZED ABLATION AND SAFETY
    # --------------------------------------------------------

    fig, axs = plt.subplots(
        1, 2,
        figsize=(10.6, 3.5)
    )

    names = [
        "Full vs CDP",
        "Dynamic component",
        "Counterfactual component",
        "Guard component"
    ]

    # Hypothetical values only.
    means = np.array([
        .016, .007, .011, .023
    ])

    errors = np.array([
        .006, .005, .006, .007
    ])

    yy = np.arange(4)[::-1]

    axs[0].errorbar(
        means,
        yy,
        xerr=errors,
        fmt="o",
        color=TEAL,
        capsize=3
    )

    axs[0].set_yticks(
        yy,
        names
    )

    axs[0].axvline(
        0,
        color=RED,
        linestyle="--"
    )

    axs[0].set_xlabel(
        "Hypothetical paired PAS effect"
    )

    label(
        axs[0],
        "(a) Idealized component contributions"
    )

    stress = np.arange(5)

    axs[1].plot(
        stress,
        [1, 2, 4, 7, 11],
        "-o",
        color=ORANGE,
        label="No Guard"
    )

    axs[1].plot(
        stress,
        [1, 1.5, 2, 3, 4],
        "-s",
        color=BLUE,
        label="Guard"
    )

    axs[1].legend(
        frameon=False
    )

    axs[1].set_xlabel(
        "Attack stress level"
    )

    axs[1].set_ylabel(
        "Hypothetical adverse-switch rate (%)"
    )

    label(
        axs[1],
        "(b) Illustrative tail-risk reduction"
    )

    watermark(fig)

    fig.tight_layout(
        rect=[0, .04, 1, 1]
    )

    finish(
        fig,
        "SIM_02_ideal_ablation_safety",
        SYN
    )

    # --------------------------------------------------------
    # SIM 03: IDEALIZED SCALABILITY
    # --------------------------------------------------------

    fig, axs = plt.subplots(
        1, 2,
        figsize=(10, 3.4)
    )

    nodes = np.array([
        14, 57, 118, 300
    ])

    hypothetical_time = np.array([
        .8, 1.9, 3.4, 7.6
    ])

    axs[0].plot(
        nodes,
        hypothetical_time,
        "-o",
        color=BLUE,
        linewidth=2,
        label="Mean"
    )

    axs[0].plot(
        nodes,
        1.3*hypothetical_time,
        "--s",
        color=ORANGE,
        linewidth=1.5,
        label="P95"
    )

    axs[0].legend(frameon=False)

    axs[0].set_xticks(nodes)
    axs[0].set_xlabel("IEEE nodes")

    axs[0].set_ylabel(
        "Hypothetical decision time (s)"
    )

    label(
        axs[0],
        "(a) Illustrative computation time"
    )

    axs[1].plot(
        nodes,
        [2, 1.9, 1.8, 1.6],
        "-o",
        color=TEAL,
        linewidth=2
    )

    axs[1].axhline(
        0,
        color=GREY,
        linestyle="--"
    )

    axs[1].set_xticks(nodes)
    axs[1].set_xlabel("IEEE nodes")

    axs[1].set_ylabel(
        "Hypothetical PAS improvement (%)"
    )

    label(
        axs[1],
        "(b) Illustrative positive effects"
    )

    watermark(fig)

    fig.tight_layout(
        rect=[0, .04, 1, 1]
    )

    finish(
        fig,
        "SIM_03_ideal_scalability",
        SYN
    )

    # --------------------------------------------------------
    # SIM 04: IDEALIZED PHYSICAL CONSTRAINT CHECK
    # --------------------------------------------------------

    fig, axs = plt.subplots(
        1, 2,
        figsize=(10.6, 3.5)
    )

    buses = np.arange(12)

    voltage = (
        1 + .025*np.sin(buses*.7)
    )

    axs[0].plot(
        buses,
        voltage,
        "-o",
        color=BLUE
    )

    axs[0].axhline(
        1.05,
        color=RED,
        linestyle="--"
    )

    axs[0].axhline(
        .95,
        color=RED,
        linestyle="--"
    )

    axs[0].set_xlabel(
        "Representative buses"
    )

    axs[0].set_ylabel(
        "Hypothetical voltage (p.u.)"
    )

    label(
        axs[0],
        "(a) Illustrative voltage limits"
    )

    categories = [
        "Voltage",
        "Reactive power",
        "Active generation",
        "Thermal limit"
    ]

    axs[1].barh(
        categories,
        [0, 0, 0, 0],
        color=TEAL
    )

    axs[1].set_xlim(0, 2)

    axs[1].set_xlabel(
        "Hypothetical violations"
    )

    label(
        axs[1],
        "(b) Fully specified constraints"
    )

    watermark(fig)

    fig.tight_layout(
        rect=[0, .04, 1, 1]
    )

    finish(
        fig,
        "SIM_04_ideal_physical_validation",
        SYN
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("="*65)
    print("IEEE TNSE REVISION FIGURE GENERATOR")
    print("="*65)

    print("Project:", ROOT)
    print("Results:", R)
    print("Real figures:", OUT)
    print("Synthetic figures:", SYN)

    if not R.is_dir():
        sys.exit(
            "ERROR: revision_experiments/results "
            "does not exist."
        )

    jobs = [
        ("Figure 2", figure2),
        ("Figure 3", figure3),
        ("Figure 4", figure4),
        ("Figure 5", figure5),
        ("Figure 6", figure6),
        ("Figure 7", figure7)
    ]

    failures = []

    for name, function in jobs:

        print("\n[RUNNING]", name, flush=True)

        try:
            function()

        except Exception as exc:

            failures.append(name)

            print(
                "[ERROR]",
                name,
                repr(exc),
                file=sys.stderr
            )

    print("\n[RUNNING] SYNTHETIC DEMONSTRATIONS")

    try:
        simulated_figures()

    except Exception as exc:
        failures.append("Synthetic figures")
        print("[ERROR]", repr(exc), file=sys.stderr)

    print("\n"+"="*65)
    print("FIGURE GENERATION SUMMARY")
    print("="*65)

    print("Real figure folder:", OUT)
    print("Synthetic folder:", SYN)

    if failures:
        print("Failed:", failures)
        sys.exit(1)

    print("ALL FIGURES GENERATED SUCCESSFULLY")
