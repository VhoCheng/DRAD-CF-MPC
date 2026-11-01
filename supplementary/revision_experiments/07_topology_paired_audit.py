from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "revision_experiments/results"
FIG = ROOT / "revision_experiments/figures"
WORK = ROOT / "revision_experiments/topology_workspaces"

OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

KEY = [
    "topology", "graph_seed", "pattern",
    "info", "budget", "onset"
]

METHODS = ["CDP", "DRAD-CF-MPC"]
TOPOLOGIES = ["BA", "ER", "WS"]

BOOTSTRAPS = 5000
RNG = np.random.default_rng(20261018)


def main():
    print("=" * 70)
    print("TNSE EXPERIMENT 07")
    print("PAIRED TOPOLOGY AND STRUCTURAL AUDIT")
    print("=" * 70)

    sources = sorted(
        OUT.glob("06_topology_full_*_summary.csv"),
        key=lambda p: p.stat().st_mtime
    )

    if not sources:
        raise FileNotFoundError(
            "Experiment 06 full summary not found."
        )

    source = sources[-1]
    prefix = source.name.removesuffix("_summary.csv")

    run_source = OUT / f"{prefix}_per_run.csv"
    workspace = WORK / prefix

    if not run_source.exists():
        raise FileNotFoundError(run_source)

    if not workspace.exists():
        raise FileNotFoundError(workspace)

    summary = pd.read_csv(source)
    runs = pd.read_csv(run_source)

    if len(summary) != 720:
        raise RuntimeError(
            f"Expected 720 summary rows, got {len(summary)}"
        )

    if len(runs) != 7200:
        raise RuntimeError(
            f"Expected 7200 per-run rows, got {len(runs)}"
        )

    if set(summary["topology"]) != set(TOPOLOGIES):
        raise RuntimeError("Unexpected topology labels.")

    if set(summary["strategy"]) != set(METHODS):
        raise RuntimeError("Unexpected method labels.")

    if summary.duplicated(KEY + ["strategy"]).any():
        raise RuntimeError("Duplicate summary rows.")

    if runs.duplicated(
        KEY + ["strategy", "run_index"]
    ).any():
        raise RuntimeError("Duplicate per-run rows.")

    counts = runs.groupby(
        KEY + ["strategy"]
    )["run_index"].nunique()

    if not np.all(counts.to_numpy() == 10):
        raise RuntimeError("Missing Monte Carlo runs.")

    # Verify the original EP definition from run-level records.
    recomputed = (
        runs.groupby(
            KEY + ["strategy"], as_index=False
        )
        .agg(
            mean_PAS=("PAS", "mean"),
            mean_PS=("PS", "mean"),
        )
    )

    check = summary.merge(
        recomputed,
        on=KEY + ["strategy"],
        validate="one_to_one"
    )

    reconstructed = (
        check["mean_PAS"] * check["mean_PS"]
    )

    max_error = float(np.max(np.abs(
        check["EP"] - reconstructed
    )))

    if max_error > 1e-10:
        raise RuntimeError(
            f"EP reconstruction failed: {max_error}"
        )

    print(
        "DATA AND EP VALIDATION PASSED:",
        f"max_error={max_error:.3e}"
    )

    paired = (
        summary.pivot(
            index=KEY,
            columns="strategy",
            values="EP",
        )
        .reset_index()
    )

    if len(paired) != 360:
        raise RuntimeError("Expected 360 paired settings.")

    paired["EP_gain"] = (
        paired["CDP"] - paired["DRAD-CF-MPC"]
    )

    # Directly paired actual PAS observations.
    run_paired = (
        runs.pivot(
            index=KEY + ["run_index"],
            columns="strategy",
            values="PAS",
        )
        .reset_index()
    )

    run_paired["negative_gain"] = (
        run_paired["DRAD-CF-MPC"]
        > run_paired["CDP"] + 1e-12
    )

    switch_rates = (
        summary[
            summary["strategy"] == "DRAD-CF-MPC"
        ]
        .groupby("topology")["switch_rate"]
        .mean()
    )

    negative_rates = (
        run_paired.groupby("topology")[
            "negative_gain"
        ]
        .mean()
    )

    topology_rows = []
    seed_rows = []
    structural_rows = []

    for topology in TOPOLOGIES:

        subset = paired[
            paired["topology"] == topology
        ].copy()

        seeds = sorted(
            subset["graph_seed"].unique()
        )

        if len(seeds) != 3:
            raise RuntimeError(
                f"{topology}: expected three graph seeds."
            )

        factors = sorted(set(
            (
                row.pattern,
                row.info,
                row.budget,
                row.onset,
            )
            for row in subset.itertuples()
        ))

        if len(factors) != 40:
            raise RuntimeError(
                f"{topology}: expected 40 factor combinations."
            )

        indexed = subset.set_index(
            ["graph_seed", "pattern", "info", "budget", "onset"]
        )

        cube = np.empty((3, 40, 2), dtype=float)

        for i, seed in enumerate(seeds):
            for j, factors_row in enumerate(factors):
                row = indexed.loc[
                    (seed,) + factors_row
                ]

                cube[i, j, 0] = float(row["CDP"])
                cube[i, j, 1] = float(
                    row["DRAD-CF-MPC"]
                )

        # Hierarchical exploratory bootstrap:
        # resample graphs, then factor settings inside each graph.
        sampled_graphs = RNG.integers(
            0, 3, size=(BOOTSTRAPS, 3)
        )

        sampled_factors = RNG.integers(
            0, 40, size=(BOOTSTRAPS, 3, 40)
        )

        draws = cube[
            sampled_graphs[:, :, None],
            sampled_factors
        ]

        bootstrap_means = draws.mean(
            axis=(1, 2)
        )

        bootstrap_gain = (
            bootstrap_means[:, 0]
            - bootstrap_means[:, 1]
        )

        bootstrap_gain_pct = (
            100.0
            * bootstrap_gain
            / np.maximum(
                bootstrap_means[:, 0], 1e-12
            )
        )

        mean_cdp = float(cube[:, :, 0].mean())
        mean_drad = float(cube[:, :, 1].mean())
        mean_gain = mean_cdp - mean_drad

        ci_low, ci_high = np.quantile(
            bootstrap_gain,
            [0.025, 0.975]
        )

        pct_low, pct_high = np.quantile(
            bootstrap_gain_pct,
            [0.025, 0.975]
        )

        topology_rows.append({
            "topology": topology,
            "graph_instances": 3,
            "paired_settings": 120,
            "mean_CDP_EP": mean_cdp,
            "mean_DRAD_EP": mean_drad,
            "mean_EP_gain": mean_gain,
            "relative_gain_pct": (
                100.0 * mean_gain / mean_cdp
            ),
            "bootstrap_gain_CI_low": ci_low,
            "bootstrap_gain_CI_high": ci_high,
            "bootstrap_pct_CI_low": pct_low,
            "bootstrap_pct_CI_high": pct_high,
            "better_settings": int(
                (subset["EP_gain"] > 1e-12).sum()
            ),
            "worse_settings": int(
                (subset["EP_gain"] < -1e-12).sum()
            ),
            "tied_settings": int(
                (subset["EP_gain"].abs() <= 1e-12).sum()
            ),
            "mean_switch_rate": float(
                switch_rates.loc[topology]
            ),
            "negative_paired_PAS_rate": float(
                negative_rates.loc[topology]
            ),
        })

        for i, seed in enumerate(seeds):
            mean_by_seed = cube[i].mean(axis=0)

            seed_cdp = float(mean_by_seed[0])
            seed_drad = float(mean_by_seed[1])

            seed_rows.append({
                "topology": topology,
                "graph_seed": int(seed),
                "mean_CDP_EP": seed_cdp,
                "mean_DRAD_EP": seed_drad,
                "EP_gain": seed_cdp - seed_drad,
                "relative_gain_pct": (
                    100.0
                    * (seed_cdp - seed_drad)
                    / max(seed_cdp, 1e-12)
                ),
            })

            graph_path = (
                workspace
                / f"{topology}_seed{seed}"
                / "graph.json"
            )

            graph_data = json.loads(
                graph_path.read_text(
                    encoding="utf-8"
                )
            )

            graph = nx.Graph()
            graph.add_nodes_from(range(118))
            graph.add_edges_from(
                graph_data["edge_list"]
            )

            if not nx.is_connected(graph):
                raise RuntimeError(
                    f"Disconnected graph: {topology}/{seed}"
                )

            degrees = np.asarray(
                [graph.degree(n) for n in graph.nodes()],
                dtype=float
            )

            structural_rows.append({
                "topology": topology,
                "graph_seed": int(seed),
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "average_degree": float(degrees.mean()),
                "maximum_degree": int(degrees.max()),
                "degree_variance": float(degrees.var()),
                "average_clustering": float(
                    nx.average_clustering(graph)
                ),
                "average_shortest_path": float(
                    nx.average_shortest_path_length(graph)
                ),
                "global_efficiency": float(
                    nx.global_efficiency(graph)
                ),
            })

    result = pd.DataFrame(topology_rows)
    per_seed = pd.DataFrame(seed_rows)
    structure = pd.DataFrame(structural_rows)

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S_%f")

    base = f"07_topology_audit_{timestamp}"

    summary_path = OUT / f"{base}_summary.csv"
    seed_path = OUT / f"{base}_by_seed.csv"
    structure_path = OUT / f"{base}_structure.csv"

    result.to_csv(summary_path, index=False)
    per_seed.to_csv(seed_path, index=False)
    structure.to_csv(structure_path, index=False)

    # Paper-style grouped comparison.
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
    })

    fig, ax = plt.subplots(figsize=(7.3, 4.2))

    x = np.arange(len(TOPOLOGIES))
    width = 0.34

    ax.bar(
        x - width / 2,
        result["mean_CDP_EP"],
        width,
        color="#AAB6C2",
        label="CDP",
    )

    ax.bar(
        x + width / 2,
        result["mean_DRAD_EP"],
        width,
        color="#395F83",
        label="DRAD-CF-MPC",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(TOPOLOGIES)
    ax.set_ylabel("Mean expected payoff (lower is better)")
    ax.set_xlabel("Communication-network topology")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)

    figure_png = FIG / f"{base}.png"
    figure_svg = FIG / f"{base}.svg"

    fig.savefig(figure_png, dpi=300)
    fig.savefig(figure_svg)
    plt.close(fig)

    print("\n" + "=" * 70)
    print("TOPOLOGY-LEVEL PAIRED RESULTS")
    print("=" * 70)

    print(result.to_string(index=False))

    print("\n" + "=" * 70)
    print("RESULTS BY INDEPENDENT GRAPH SEED")
    print("=" * 70)

    print(per_seed.to_string(index=False))

    print("\n" + "=" * 70)
    print("NETWORK STRUCTURAL CHARACTERISTICS")
    print("=" * 70)

    print(structure.to_string(index=False))

    print("\nBootstrap intervals are exploratory.")
    print("Only three independently generated graphs per topology.")
    print("Do not interpret these as universal network guarantees.")

    print("\nSummary:", summary_path)
    print("Per-seed results:", seed_path)
    print("Graph structure:", structure_path)
    print("Figure PNG:", figure_png)
    print("Figure SVG:", figure_svg)
    print("EXPERIMENT 07 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
