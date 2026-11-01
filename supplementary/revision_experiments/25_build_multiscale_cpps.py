from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd
import networkx as nx

from pypower.api import (
    case14, case57, case118, case300,
    ppoption, runpf,
)
from pypower.idx_bus import (
    BUS_I, PD, VM, VMIN, VMAX,
)
from pypower.idx_gen import (
    GEN_BUS, PG, GEN_STATUS,
)
from pypower.idx_brch import (
    F_BUS, T_BUS, BR_STATUS, PF, PT,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = (
    ROOT / "revision_experiments"
    / "results/25_multiscale_networks"
)

CASES = {
    14: case14,
    57: case57,
    118: case118,
    300: case300,
}

PATTERNS = [
    "DDAC", "DDDC", "DCAC", "DCDC", "RC"
]


def norm(values):
    x = np.asarray(values, dtype=float)

    if not np.all(np.isfinite(x)):
        raise RuntimeError(
            "Non-finite values in physical features"
        )

    gap = np.max(x) - np.min(x)

    if gap < 1e-12:
        return np.zeros_like(x)

    return (x - np.min(x)) / gap


def file_sha256(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def create_one(n_bus):
    started = time.perf_counter()

    ppc = CASES[n_bus]()

    buses = ppc["bus"]
    gens = ppc["gen"]
    branches = ppc["branch"]

    n = len(buses)

    if n != n_bus:
        raise RuntimeError(
            f"Unexpected bus count: {n} != {n_bus}"
        )

    original_bus_ids = [
        int(x) for x in buses[:, BUS_I]
    ]

    if len(set(original_bus_ids)) != n:
        raise RuntimeError(
            "Duplicated physical bus IDs"
        )

    id_to_index = {
        bus_id: i
        for i, bus_id in enumerate(
            original_bus_ids
        )
    }

    # Build physical graph using all in-service
    # AC branches, including transformer-equivalent
    # branches in the original PYPOWER case.
    gp = nx.Graph()
    gp.add_nodes_from(range(n))

    valid_branches = 0

    for branch in branches:
        if int(branch[BR_STATUS]) <= 0:
            continue

        u_id = int(branch[F_BUS])
        v_id = int(branch[T_BUS])

        if (
            u_id not in id_to_index
            or v_id not in id_to_index
        ):
            raise RuntimeError(
                "Branch references unknown bus"
            )

        u = id_to_index[u_id]
        v = id_to_index[v_id]

        if u != v:
            gp.add_edge(u, v)
            valid_branches += 1

    if gp.number_of_edges() == 0:
        raise RuntimeError(
            "Empty physical network"
        )

    degree = dict(gp.degree())

    betweenness = nx.betweenness_centrality(
        gp, normalized=True
    )

    load = np.maximum(
        buses[:, PD].astype(float),
        0.0
    )

    generation = np.zeros(n, dtype=float)

    for gen in gens:
        if int(gen[GEN_STATUS]) <= 0:
            continue

        bus_id = int(gen[GEN_BUS])

        if bus_id not in id_to_index:
            raise RuntimeError(
                "Generator references unknown bus"
            )

        idx = id_to_index[bus_id]

        generation[idx] += max(
            float(gen[PG]), 0.0
        )

    capacity = load + generation

    power = pd.DataFrame({
        "power_node": np.arange(n),
        "original_bus_id": original_bus_ids,
        "power_degree": [
            degree[i] for i in range(n)
        ],
        "power_betweenness": [
            betweenness[i] for i in range(n)
        ],
        "load_mw": load,
        "gen_proxy_mw": generation,
        "capacity_proxy": capacity,
    })

    for col in [
        "power_degree",
        "power_betweenness",
        "load_mw",
        "gen_proxy_mw",
        "capacity_proxy",
    ]:
        power[col + "_norm"] = norm(
            power[col].values
        )

    # Communication layer:
    # same BA family and m=2 used by the
    # original project, resized to each case.
    gc = nx.barabasi_albert_graph(
        n=n, m=2, seed=42
    )

    comm_degree = dict(gc.degree())

    comm_between = (
        nx.betweenness_centrality(
            gc, normalized=True
        )
    )

    comm_close = nx.closeness_centrality(gc)

    comm = pd.DataFrame({
        "comm_node": np.arange(n),
        "comm_degree": [
            comm_degree[i] for i in range(n)
        ],
        "comm_betweenness": [
            comm_between[i] for i in range(n)
        ],
        "comm_closeness": [
            comm_close[i] for i in range(n)
        ],
    })

    for col in [
        "comm_degree",
        "comm_betweenness",
        "comm_closeness",
    ]:
        comm[col + "_norm"] = norm(
            comm[col].values
        )

    case_dir = OUT / f"case{n}"
    table_dir = (
        case_dir / "outputs/tables"
    )

    table_dir.mkdir(
        parents=True, exist_ok=True
    )

    power.to_csv(
        table_dir / "power_features.csv",
        index=False
    )

    comm.to_csv(
        table_dir / "comm_features.csv",
        index=False
    )

    # Same five coupling definitions as
    # the 118-node source generation script.
    for pattern in PATTERNS:
        if pattern == "RC":
            comm_ids = np.arange(n)
            power_ids = np.arange(n)

            rng = np.random.default_rng(42)
            rng.shuffle(power_ids)

        else:
            comm_ids = (
                comm.sort_values(
                    "comm_degree",
                    ascending=False,
                    kind="mergesort",
                )["comm_node"]
                .to_numpy()
            )

            if pattern in ("DDAC", "DDDC"):
                col = "power_degree"
            else:
                col = "capacity_proxy"

            ascending = (
                pattern in ("DDDC", "DCDC")
            )

            power_ids = (
                power.sort_values(
                    col,
                    ascending=ascending,
                    kind="mergesort",
                )["power_node"]
                .to_numpy()
            )

        coupling = pd.DataFrame({
            "comm_node": comm_ids,
            "power_node": power_ids,
            "pattern": pattern,
        })

        if (
            coupling["comm_node"].nunique() != n
            or coupling["power_node"].nunique() != n
        ):
            raise RuntimeError(
                f"Invalid coupling for {pattern}"
            )

        coupling.to_csv(
            table_dir / f"coupling_{pattern}.csv",
            index=False,
        )

    # Actual baseline AC power-flow solution.
    options = ppoption(
        VERBOSE=0,
        OUT_ALL=0,
        PF_ALG=1,
        PF_MAX_IT=50,
    )

    solved, converged = runpf(
        ppc, options
    )

    converged = bool(converged)

    solved_bus = solved["bus"]

    result_by_id = {
        int(row[BUS_I]): row
        for row in solved_bus
    }

    voltage = np.array([
        float(result_by_id[b][VM])
        for b in original_bus_ids
    ])

    lower = np.array([
        float(result_by_id[b][VMIN])
        for b in original_bus_ids
    ])

    upper = np.array([
        float(result_by_id[b][VMAX])
        for b in original_bus_ids
    ])

    voltage_violations = int(np.sum(
        (voltage < lower - 1e-3)
        | (voltage > upper + 1e-3)
    ))

    pf_table = pd.DataFrame({
        "power_node": np.arange(n),
        "original_bus_id": original_bus_ids,
        "vm_pu": voltage,
        "vmin_pu": lower,
        "vmax_pu": upper,
        "voltage_limit_violated": (
            (voltage < lower - 1e-3)
            | (voltage > upper + 1e-3)
        ).astype(int),
    })

    pf_table.to_csv(
        case_dir / "baseline_ac_pf.csv",
        index=False,
    )

    # PF/PT are populated in solved branch data.
    solved_branches = solved["branch"]

    active = (
        solved_branches[:, BR_STATUS] > 0
    )

    branch_loss_mw = float(np.sum(
        solved_branches[active, PF]
        + solved_branches[active, PT]
    ))

    summary = {
        "case": f"IEEE {n}-bus",
        "data_source": "PYPOWER standard case",
        "power_nodes": n,
        "power_graph_edges": int(
            gp.number_of_edges()
        ),
        "in_service_branch_rows": (
            valid_branches
        ),
        "power_graph_components": int(
            nx.number_connected_components(gp)
        ),
        "comm_nodes": n,
        "comm_edges": int(
            gc.number_of_edges()
        ),
        "comm_average_degree": float(
            np.mean([d for _, d in gc.degree()])
        ),
        "comm_clustering": float(
            nx.average_clustering(gc)
        ),
        "total_load_mw": float(load.sum()),
        "ac_pf_converged": converged,
        "base_voltage_min_pu": float(
            voltage.min()
        ),
        "base_voltage_max_pu": float(
            voltage.max()
        ),
        "base_voltage_limit_violations": (
            voltage_violations
        ),
        "baseline_branch_loss_mw": (
            branch_loss_mw
        ),
        "coupling_patterns": PATTERNS,
        "comm_seed": 42,
        "coupling_seed": 42,
        "runtime_seconds": round(
            time.perf_counter() - started, 3
        ),
    }

    (case_dir / "network_summary.json").write_text(
        json.dumps(
            summary, indent=2
        ),
        encoding="utf-8",
    )

    if not converged:
        raise RuntimeError(
            f"AC power flow did not converge: case{n}"
        )

    print(
        f"[case{n}] "
        f"AC PF converged; "
        f"physical edges={gp.number_of_edges()}; "
        f"voltage=[{voltage.min():.4f},"
        f"{voltage.max():.4f}]; "
        f"elapsed={summary['runtime_seconds']}s",
        flush=True,
    )

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["pilot", "full"],
        required=True,
    )

    args = parser.parse_args()

    OUT.mkdir(
        parents=True, exist_ok=True
    )

    sizes = (
        [14, 118]
        if args.mode == "pilot"
        else [14, 57, 118, 300]
    )

    print("=" * 72, flush=True)
    print("TNSE EXPERIMENT 25", flush=True)
    print("IEEE MULTISCALE CPPS NETWORK CONSTRUCTION", flush=True)
    print("=" * 72, flush=True)

    print("Mode:", args.mode, flush=True)
    print("Cases:", sizes, flush=True)

    rows = [
        create_one(n)
        for n in sizes
    ]

    df = pd.DataFrame(rows)

    df.to_csv(
        OUT / "network_comparison.csv",
        index=False
    )

    print("\n" + "=" * 72)
    print("IEEE NETWORK CONSTRUCTION RESULTS")
    print("=" * 72)

    print(
        df[[
            "case",
            "power_nodes",
            "power_graph_edges",
            "comm_nodes",
            "comm_edges",
            "total_load_mw",
            "ac_pf_converged",
            "base_voltage_min_pu",
            "base_voltage_max_pu",
            "base_voltage_limit_violations",
            "runtime_seconds",
        ]].to_string(index=False),
        flush=True,
    )

    print(
        "\nImportant: AC power flow was solved "
        "for the BASE operating point only."
    )

    print(
        "Malware-induced physical consequences "
        "have NOT yet been evaluated by AC power flow."
    )

    print(
        "\nEXPERIMENT 25 COMPLETED SUCCESSFULLY",
        flush=True,
    )


if __name__ == "__main__":
    main()
