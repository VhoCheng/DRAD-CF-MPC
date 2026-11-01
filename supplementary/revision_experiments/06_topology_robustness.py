from pathlib import Path
from datetime import datetime, timezone
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import run_experiments_core as core
import drad_mpc as mpc
import dynamic_risk_defense as dynamic

OUT = ROOT / "revision_experiments/results"
WORK_ROOT = ROOT / "revision_experiments/topology_workspaces"

N = 118
M_BA = 2
WS_K = 4
WS_P = 0.10
ER_EDGES = 232

PATTERNS = ["DDAC", "DDDC", "DCAC", "DCDC", "RC"]

BETA = 0.01
ALPHA = 0.15
DELTA = 1e-5

LOOKAHEAD = 5

PILOT = [
    ("DDAC", 0.50, 60, 150),
    ("RC", 0.50, 40, 120),
]

FULL = list(product(
    PATTERNS,
    [0.25, 1.00],
    [40, 80],
    [90, 210],
))

STRATEGIES = ["CDP", "DRAD-CF-MPC"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_source():
    names = [
        "run_experiments_core.py",
        "drad_mpc.py",
        "dynamic_risk_defense.py",
        "seid_malware.py",
        "baseline_defense.py",
    ]

    for name in names:
        current = ROOT / "src" / name
        original = ROOT.parent / "DRAD-CF-MPC/src" / name

        if not current.exists() or not original.exists():
            raise RuntimeError(f"Missing source file: {name}")

        if sha256(current) != sha256(original):
            raise RuntimeError(
                f"Original implementation differs: {name}"
            )

    print("ORIGINAL SOURCE CHECK PASSED", flush=True)


def normalize(values):
    x = np.asarray(values, dtype=float)
    if np.max(x) - np.min(x) < 1e-12:
        return np.zeros_like(x)
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def make_graph(topology, seed):
    if topology == "BA":
        graph = nx.barabasi_albert_graph(
            N, M_BA, seed=seed
        )
        metadata = {"generator": "BA", "effective_seed": seed}

    elif topology == "WS":
        graph = nx.watts_strogatz_graph(
            N, WS_K, WS_P, seed=seed
        )
        metadata = {
            "generator": "WS",
            "effective_seed": seed,
            "k": WS_K,
            "rewiring_probability": WS_P,
        }

    elif topology == "ER":
        graph = None
        metadata = None

        # Connected ER G(n,m), with the same edge count as BA.
        # Record rejection attempts for reproducibility.
        for attempt in range(2000):
            effective_seed = seed + 10007 * attempt

            candidate = nx.gnm_random_graph(
                N, ER_EDGES, seed=effective_seed
            )

            if nx.is_connected(candidate):
                graph = candidate
                metadata = {
                    "generator": "connected_ER_Gnm",
                    "effective_seed": effective_seed,
                    "attempts": attempt + 1,
                    "fixed_edges": ER_EDGES,
                }
                break

        if graph is None:
            raise RuntimeError(
                "Cannot construct connected ER graph."
            )

    else:
        raise ValueError(topology)

    if graph.number_of_nodes() != N:
        raise RuntimeError("Unexpected node count.")

    if not nx.is_connected(graph):
        raise RuntimeError(
            f"{topology} network is disconnected."
        )

    return graph, metadata


def build_comm_features(graph):
    degree = dict(graph.degree())
    between = nx.betweenness_centrality(
        graph, normalized=True
    )
    close = nx.closeness_centrality(graph)

    df = pd.DataFrame({
        "comm_node": list(range(N)),
        "comm_degree": [
            degree.get(i, 0) for i in range(N)
        ],
        "comm_betweenness": [
            between.get(i, 0.0) for i in range(N)
        ],
        "comm_closeness": [
            close.get(i, 0.0) for i in range(N)
        ],
    })

    for col in [
        "comm_degree",
        "comm_betweenness",
        "comm_closeness",
    ]:
        df[col + "_norm"] = normalize(df[col].values)

    return df


def make_coupling(comm, power, pattern):
    rng = np.random.default_rng(42)

    comm_nodes = comm["comm_node"].values.copy()
    power_nodes = power["power_node"].values.copy()

    if pattern == "DDAC":
        c = comm.sort_values(
            "comm_degree", ascending=False
        )["comm_node"].values
        p = power.sort_values(
            "power_degree", ascending=False
        )["power_node"].values

    elif pattern == "DDDC":
        c = comm.sort_values(
            "comm_degree", ascending=False
        )["comm_node"].values
        p = power.sort_values(
            "power_degree", ascending=True
        )["power_node"].values

    elif pattern == "DCAC":
        c = comm.sort_values(
            "comm_degree", ascending=False
        )["comm_node"].values
        p = power.sort_values(
            "capacity_proxy", ascending=False
        )["power_node"].values

    elif pattern == "DCDC":
        c = comm.sort_values(
            "comm_degree", ascending=False
        )["comm_node"].values
        p = power.sort_values(
            "capacity_proxy", ascending=True
        )["power_node"].values

    elif pattern == "RC":
        c = comm_nodes.copy()
        p = power_nodes.copy()
        rng.shuffle(p)

    else:
        raise ValueError(pattern)

    return pd.DataFrame({
        "comm_node": c,
        "power_node": p,
        "pattern": pattern,
    })


def prepare_workspace(topology, seed, run_dir):
    graph, generator_meta = make_graph(topology, seed)

    workspace = run_dir / f"{topology}_seed{seed}"
    tables = workspace / "outputs/tables"
    tables.mkdir(parents=True, exist_ok=False)

    original_tables = ROOT / "outputs/tables"

    power_source = original_tables / "power_features.csv"

    if not power_source.exists():
        raise FileNotFoundError(power_source)

    shutil.copy2(
        power_source,
        tables / "power_features.csv"
    )

    power = pd.read_csv(power_source)

    if len(power) != N:
        raise RuntimeError(
            "Power-layer node count is not 118."
        )

    comm = build_comm_features(graph)

    comm.to_csv(
        tables / "comm_features.csv",
        index=False,
    )

    for pattern in PATTERNS:
        coupling = make_coupling(
            comm, power, pattern
        )

        if (
            coupling["comm_node"].nunique() != N
            or coupling["power_node"].nunique() != N
        ):
            raise RuntimeError(
                f"Invalid one-to-one coupling: {pattern}"
            )

        coupling.to_csv(
            tables / f"coupling_{pattern}.csv",
            index=False,
        )

    # BA seed 42 should regenerate the original feature tables
    # and original five coupling mappings.
    if topology == "BA" and seed == 42:
        original_comm = pd.read_csv(
            original_tables / "comm_features.csv"
        )

        if list(comm.columns) != list(original_comm.columns):
            raise RuntimeError(
                "BA reference feature columns differ."
            )

        if not np.allclose(
            comm.to_numpy(dtype=float),
            original_comm.to_numpy(dtype=float),
            rtol=0,
            atol=1e-10,
        ):
            raise RuntimeError(
                "BA reference features do not match."
            )

        for pattern in PATTERNS:
            old = pd.read_csv(
                original_tables / f"coupling_{pattern}.csv"
            )

            new = pd.read_csv(
                tables / f"coupling_{pattern}.csv"
            )

            if not new.equals(old):
                raise RuntimeError(
                    f"BA reference coupling differs: {pattern}"
                )

        print(
            "BA REFERENCE TABLES CHECK PASSED",
            flush=True,
        )

    edges = sorted([
        sorted([int(a), int(b)])
        for a, b in graph.edges()
    ])

    degrees = np.array([
        graph.degree(i) for i in range(N)
    ])

    graph_record = {
        "topology": topology,
        "requested_seed": seed,
        "generator": generator_meta,
        "nodes": N,
        "edges": len(edges),
        "average_degree": float(degrees.mean()),
        "degree_variance": float(degrees.var()),
        "connected": True,
        "edge_list": edges,
    }

    (workspace / "graph.json").write_text(
        json.dumps(graph_record, indent=2),
        encoding="utf-8",
    )

    return workspace


def worker(task):
    (
        task_id,
        topology,
        graph_seed,
        scenario,
        workspace,
        n_runs,
        base_seed,
    ) = task

    previous_cwd = Path.cwd()

    old_core = core.build_comm_graph
    old_mpc = mpc.build_comm_graph
    old_dynamic = dynamic.build_comm_graph

    try:
        os.chdir(workspace)

        graph_data = json.loads(
            Path("graph.json").read_text(
                encoding="utf-8"
            )
        )

        graph = nx.Graph()
        graph.add_nodes_from(range(N))
        graph.add_edges_from(graph_data["edge_list"])

        if not nx.is_connected(graph):
            raise RuntimeError("Loaded graph is disconnected.")

        if topology != "BA_LEGACY":
            # All model components must use exactly the
            # same communication graph in this experiment.
            def fixed_graph(n_nodes=118, seed=42):
                if n_nodes != N:
                    raise RuntimeError(
                        "Unexpected graph size requested."
                    )
                return graph

            core.build_comm_graph = fixed_graph
            mpc.build_comm_graph = fixed_graph
            dynamic.build_comm_graph = fixed_graph

        pattern, info, budget, onset = scenario

        rows = []

        for r in range(n_runs):
            seed_arg = base_seed + r

            actual_seed = (
                seed_arg
                + onset * 1000
                + budget * 17
            )

            for strategy in STRATEGIES:
                result = core.run_single_setting(
                    pattern=pattern,
                    strategy=strategy,
                    attack_information_ratio=info,
                    defense_budget=budget,
                    onset_time=onset,
                    n_runs=1,
                    lookahead_runs=LOOKAHEAD,
                    infection_rate=BETA,
                    activation_rate=ALPHA,
                    detection_rate=DELTA,
                    seed=seed_arg,
                )

                rows.append({
                    "task_id": task_id,
                    "topology": topology,
                    "graph_seed": graph_seed,
                    "pattern": pattern,
                    "info": info,
                    "budget": budget,
                    "onset": onset,
                    "run_index": r,
                    "actual_seed": actual_seed,
                    "strategy": strategy,
                    "PAS": float(result["PAS"]),
                    "PS": float(result["PS"]),
                    "RT": float(result["RT"]),
                    "RI": float(result["RI"]),
                    "RE": float(result["RE"]),
                    "selected_policy": result[
                        "selected_policy_mode"
                    ],
                    "safe_guard": result[
                        "safe_guard_mode"
                    ],
                })

        df = pd.DataFrame(rows)
        summaries = []

        for strategy in STRATEGIES:
            subset = df[df["strategy"] == strategy]

            mean_pas = float(subset["PAS"].mean())
            mean_ps = float(subset["PS"].mean())

            summaries.append({
                "task_id": task_id,
                "topology": topology,
                "graph_seed": graph_seed,
                "pattern": pattern,
                "info": info,
                "budget": budget,
                "onset": onset,
                "strategy": strategy,
                "n_runs": n_runs,
                "PAS": mean_pas,
                "PS": mean_ps,
                "EP": mean_pas * mean_ps,
                "RT": float(subset["RT"].mean()),
                "RI": float(subset["RI"].mean()),
                "RE": float(subset["RE"].mean()),
                "switch_rate": float(
                    (
                        subset["selected_policy"] != "CDP"
                    ).mean()
                ),
            })

        return summaries, rows

    finally:
        core.build_comm_graph = old_core
        mpc.build_comm_graph = old_mpc
        dynamic.build_comm_graph = old_dynamic

        os.chdir(previous_cwd)


def validate_legacy_pilot(summary):
    files = sorted(
        OUT.glob("03_ablation_pilot_*_summary.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not files:
        raise FileNotFoundError(
            "Original validated ablation pilot not found."
        )

    reference = pd.read_csv(files[0])

    legacy = summary[
        summary["topology"] == "BA_LEGACY"
    ]

    if len(legacy) != 4:
        raise RuntimeError(
            "Expected four BA legacy reference rows."
        )

    for _, row in legacy.iterrows():
        variant = (
            "Full"
            if row["strategy"] == "DRAD-CF-MPC"
            else "CDP"
        )

        matched = reference[
            (reference["pattern"] == row["pattern"])
            & np.isclose(reference["info"], row["info"])
            & (reference["budget"] == row["budget"])
            & (reference["onset"] == row["onset"])
            & (reference["variant"] == variant)
        ]

        if len(matched) != 1:
            raise RuntimeError(
                "Missing BA legacy reference row."
            )

        expected = matched.iloc[0]

        for metric in [
            "EP", "PAS", "PS", "RT", "RI", "RE"
        ]:
            if not np.isclose(
                row[metric],
                expected[metric],
                rtol=0,
                atol=1e-10,
            ):
                raise RuntimeError(
                    f"BA LEGACY VALIDATION FAILED: "
                    f"{row['pattern']} {variant} "
                    f"{metric}: new={row[metric]}, "
                    f"old={expected[metric]}"
                )

    print(
        "BA LEGACY VALIDATION PASSED: "
        "original pilot results reproduced.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["pilot", "full"],
        required=True,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    check_source()

    if args.mode == "pilot":
        topologies = ["BA_LEGACY", "BA", "WS", "ER"]
        graph_seeds = [42]
        scenarios = PILOT
        n_runs = 3
        base_seed = 42
        workers = 2
    else:
        topologies = ["BA", "WS", "ER"]
        graph_seeds = [42, 43, 44]
        scenarios = FULL
        n_runs = 10
        base_seed = 20261020
        workers = max(1, min(args.workers, 6))

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    prefix = f"06_topology_{args.mode}_{stamp}"

    run_dir = WORK_ROOT / prefix
    run_dir.mkdir(parents=True, exist_ok=False)

    workspace_map = {}

    for topology in set(
        "BA" if t == "BA_LEGACY" else t
        for t in topologies
    ):
        for seed in graph_seeds:
            workspace_map[(topology, seed)] = (
                prepare_workspace(
                    topology, seed, run_dir
                )
            )

    tasks = []
    task_id = 0

    for topology in topologies:
        source_topology = (
            "BA" if topology == "BA_LEGACY"
            else topology
        )

        for graph_seed in graph_seeds:
            workspace = workspace_map[
                (source_topology, graph_seed)
            ]

            for scenario in scenarios:
                task_id += 1

                tasks.append((
                    task_id,
                    topology,
                    graph_seed,
                    scenario,
                    str(workspace),
                    n_runs,
                    base_seed,
                ))

    summary_path = OUT / f"{prefix}_summary.csv"
    detail_path = OUT / f"{prefix}_per_run.csv"
    config_path = OUT / f"{prefix}_config.json"

    config = {
        "experiment": "Communication topology robustness",
        "mode": args.mode,
        "topologies": topologies,
        "graph_seeds": graph_seeds,
        "scenarios": scenarios,
        "n_runs": n_runs,
        "lookahead_runs": LOOKAHEAD,
        "base_seed": base_seed,
        "workers": workers,
        "workspace_directory": str(run_dir),
        "graph_protocol": {
            "BA": "n=118, m=2",
            "WS": "n=118, k=4, p=0.10",
            "ER": "connected G(n,232)",
            "coherent_network": (
                "The same graph is supplied to actual "
                "propagation, preview, counterfactual "
                "evaluation and dynamic-risk estimation."
            ),
            "BA_LEGACY": (
                "Original unmodified graph construction; "
                "pilot validation only."
            ),
        },
        "source_hashes": {
            name: sha256(ROOT / "src" / name)
            for name in [
                "run_experiments_core.py",
                "drad_mpc.py",
                "dynamic_risk_defense.py",
                "seid_malware.py",
            ]
        },
    }

    config_path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("=" * 70, flush=True)
    print("TNSE EXPERIMENT 06", flush=True)
    print("COMMUNICATION TOPOLOGY ROBUSTNESS", flush=True)
    print("=" * 70, flush=True)
    print("Mode:", args.mode, flush=True)
    print("Topologies:", topologies, flush=True)
    print("Graph seeds:", graph_seeds, flush=True)
    print("Tasks:", len(tasks), flush=True)
    print("Monte Carlo:", n_runs, flush=True)
    print("Workers:", workers, flush=True)

    start = time.perf_counter()
    summary_rows = []

    with ProcessPoolExecutor(
        max_workers=workers
    ) as pool:

        futures = {
            pool.submit(worker, task): task
            for task in tasks
        }

        for completed, future in enumerate(
            as_completed(futures), start=1
        ):
            task = futures[future]

            try:
                summaries, details = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"TOPOLOGY TASK FAILED: {task}"
                ) from exc

            pd.DataFrame(summaries).to_csv(
                summary_path,
                mode="a",
                index=False,
                header=not summary_path.exists(),
            )

            pd.DataFrame(details).to_csv(
                detail_path,
                mode="a",
                index=False,
                header=not detail_path.exists(),
            )

            summary_rows.extend(summaries)

            print(
                f"[{completed:03d}/{len(tasks)}] "
                f"{task[1]} seed={task[2]} "
                f"scenario={task[3]} "
                f"elapsed={time.perf_counter()-start:.1f}s",
                flush=True,
            )

    summary = pd.DataFrame(summary_rows)

    if len(summary) != 2 * len(tasks):
        raise RuntimeError(
            "Incomplete topology experiment."
        )

    if args.mode == "pilot":
        validate_legacy_pilot(summary)

    aggregate = (
        summary.groupby(
            ["topology", "strategy"],
            as_index=False,
        )
        .agg(
            mean_EP=("EP", "mean"),
            mean_PAS=("PAS", "mean"),
            mean_RT=("RT", "mean"),
            mean_RI=("RI", "mean"),
            mean_RE=("RE", "mean"),
            mean_switch_rate=("switch_rate", "mean"),
        )
    )

    aggregate_path = (
        OUT / f"{prefix}_aggregate.csv"
    )

    aggregate.to_csv(
        aggregate_path, index=False
    )

    print("\n" + "=" * 70, flush=True)
    print("TOPOLOGY ROBUSTNESS RESULTS", flush=True)
    print("=" * 70, flush=True)

    print(
        aggregate.to_string(index=False),
        flush=True,
    )

    print("\nSummary:", summary_path)
    print("Per-run:", detail_path)
    print("Aggregate:", aggregate_path)
    print("Configuration:", config_path)

    print(
        "Total runtime:",
        round(time.perf_counter()-start, 2),
        "seconds",
    )

    print("EXPERIMENT 06 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
