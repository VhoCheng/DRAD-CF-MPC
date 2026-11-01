#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Exp54: graph-aware, observation-limited, delayed-action trial."""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "revision_experiments"
OUT = EXP / "results" / "54_graph_observation_delay"
OUT.mkdir(parents=True, exist_ok=True)

ref_path = EXP / "47_final_closed_loop_validation.py"
spec = importlib.util.spec_from_file_location(
    "tnse_exp47_for54", ref_path
)
ref = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ref
spec.loader.exec_module(ref)

os.chdir(ROOT)

N = 118
PATTERNS = ("DDAC", "RC")
BUDGETS = (20, 40)
ONSETS = (90, 210)
TOPOLOGIES = ("BA", "WS", "ER")

OBS_MODES = {
    "oracle": (1.0, 0.0),
    "partial80": (.8, .01),
    "partial50": (.5, .03),
    "blind": (0.0, 0.0)
}

DELAYS = (0, 3)

METHODS = (
    "CDP",
    "STATIC_RANK",
    "LOCAL_RANK",
    "CF_STATIC",
    "CF_LOCAL",
    "CF_LOCAL_NO_GUARD"
)

KTRAIN = 8
KGUARD = 24
HMAX = 32

# Frozen from Exp50's validation, not Exp54 test.
ALPHA = .25

KEYS = [
    "topology", "graph_seed", "pattern",
    "budget", "onset", "seed",
    "mode", "delay"
]


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def seed(*args):
    return ref.stable_seed("exp54", *args)


def norm(x):
    x = np.asarray(x, float).reshape(-1)

    if not np.isfinite(x).all():
        raise ValueError("Nonfinite scores")

    gap = x.max() - x.min()

    if gap < 1e-12:
        return np.zeros_like(x)

    return (x - x.min()) / gap


# --------------------------------------------------
# 1. True communication-network topologies
# --------------------------------------------------

def graph_of(topology, graph_seed):

    if topology == "BA":
        graph = nx.barabasi_albert_graph(
            N, 2, seed=graph_seed
        )

    elif topology == "WS":
        graph = nx.watts_strogatz_graph(
            N, 4, .1, seed=graph_seed
        )

        if not nx.is_connected(graph):
            for j in range(1, 200):
                graph = nx.watts_strogatz_graph(
                    N, 4, .1,
                    seed=graph_seed + 10007*j
                )
                if nx.is_connected(graph):
                    break

    elif topology == "ER":
        graph = None

        for j in range(2000):
            trial = nx.gnm_random_graph(
                N, 232,
                seed=graph_seed + 10007*j
            )

            if nx.is_connected(trial):
                graph = trial
                break

        if graph is None:
            raise RuntimeError(
                "Connected ER graph unavailable"
            )

    else:
        raise ValueError(topology)

    if not nx.is_connected(graph):
        raise RuntimeError(
            f"Disconnected graph: {topology}"
        )

    return graph


# --------------------------------------------------
# 2. Observable cyber state
# --------------------------------------------------

def observe(true_state, mode, casekey, stage):

    recall, false_positive = OBS_MODES[mode]
    truth = np.asarray(
        true_state, dtype=np.int8
    )

    if mode == "oracle":
        return truth.copy()

    if mode == "blind":
        return np.zeros_like(truth)

    rng = np.random.default_rng(
        seed("sensor", *casekey, stage, mode)
    )

    observed = np.zeros_like(truth)

    detected = (
        (truth == 2)
        &
        (rng.random(len(truth)) < recall)
    )

    observed[detected] = 2

    false_alarm = (
        (truth == 0)
        &
        (rng.random(len(truth)) < false_positive)
    )

    observed[false_alarm] = 2
    observed[truth == 3] = 3

    return observed


def top_nodes(scores, budget):

    selected = np.argsort(
        -scores, kind="stable"
    )[:budget]

    nodes = tuple(
        sorted(map(int, selected))
    )

    if (
        len(nodes) != budget
        or len(set(nodes)) != budget
    ):
        raise RuntimeError(
            "Invalid defense budget"
        )

    return nodes


# --------------------------------------------------
# 3. Fair static and state-aware candidates
# --------------------------------------------------

def policies(
    observed, adj, physical_weight,
    degree, budget
):

    basic = (
        .75*norm(physical_weight)
        + .25*norm(degree)
    )

    cdp = top_nodes(
        degree, budget
    )

    static = top_nodes(
        basic, budget
    )

    infected = (
        observed == 2
    ).astype(float)

    pressure = (
        np.asarray(
            adj @ infected
        ).reshape(-1)
        / np.maximum(degree, 1)
    )

    two_hop = (
        np.asarray(
            adj @ pressure
        ).reshape(-1)
        / np.maximum(degree, 1)
    )

    local = norm(
        .8*pressure + .2*two_hop
    )

    dynamic_score = (
        ALPHA*local
        + (1-ALPHA)*basic
    )

    dynamic_score = dynamic_score.copy()
    dynamic_score[observed == 2] = -1

    dynamic = top_nodes(
        dynamic_score, budget
    )

    if (
        not np.any(observed == 2)
        and dynamic != static
    ):
        raise RuntimeError(
            "Blind candidate invariant failed"
        )

    return {
        "CDP": cdp,
        "STATIC": static,
        "LOCAL": dynamic
    }


# --------------------------------------------------
# 4. Two-candidate CF with independent Guard
# --------------------------------------------------

def choose(
    method, observed, candidate_set,
    train, guard, adj, weight, budget
):

    if method == "CDP":
        return candidate_set["CDP"], "CDP", False, 0

    if method == "STATIC_RANK":
        return candidate_set["STATIC"], "STATIC", False, 0

    if method == "LOCAL_RANK":
        return candidate_set["LOCAL"], "LOCAL", False, 0

    alternate = (
        "STATIC"
        if method == "CF_STATIC"
        else "LOCAL"
    )

    if candidate_set[alternate] == candidate_set["CDP"]:
        return candidate_set["CDP"], "CDP", False, 0

    baseline_pred = ref.predict(
        observed,
        candidate_set["CDP"],
        train,
        adj,
        weight,
        budget,
        False
    )

    alternate_pred = ref.predict(
        observed,
        candidate_set[alternate],
        train,
        adj,
        weight,
        budget,
        False
    )

    if (
        alternate_pred["objective"]
        >= baseline_pred["objective"]
    ):
        return (
            candidate_set["CDP"],
            "CDP",
            False,
            2*KTRAIN
        )

    if method == "CF_LOCAL_NO_GUARD":
        return (
            candidate_set[alternate],
            alternate,
            True,
            2*KTRAIN
        )

    baseline_confirm = ref.predict(
        observed,
        candidate_set["CDP"],
        guard,
        adj,
        weight,
        budget,
        False
    )

    alternate_confirm = ref.predict(
        observed,
        candidate_set[alternate],
        guard,
        adj,
        weight,
        budget,
        False
    )

    improvement = (
        np.asarray(
            baseline_confirm["samples"]
        )
        -
        np.asarray(
            alternate_confirm["samples"]
        )
    )

    accept = bool(
        improvement.mean() > 0
    )

    return (
        candidate_set[alternate]
        if accept else candidate_set["CDP"],
        alternate if accept else "CDP",
        accept,
        2*(KTRAIN + KGUARD)
    )


# --------------------------------------------------
# 5. Separate true state from observations
# --------------------------------------------------

def run_case(case, weights, graphs):

    (
        topology, graph_seed,
        pattern, budget,
        onset, repetition,
        mode, delay
    ) = case

    graph, adj, degree = graphs[
        (topology, graph_seed)
    ]

    key = (
        topology, graph_seed, pattern,
        budget, onset, repetition
    )

    start_time = max(
        1, int(onset*.4)
    )

    preview = ref.simulate_seid_once(
        Gc=graph,
        onset_time=start_time,
        infection_rate=.01,
        activation_rate=.15,
        detection_rate=.00001,
        attack_information_ratio=.5,
        protected_nodes=None,
        strict_detection_failure=False,
        seed=seed("preview", *key)
    )

    initial = ref.state_from_preview(
        preview, N
    )

    remaining = onset-start_time

    durations = [
        remaining//3 + (i < remaining % 3)
        for i in range(3)
    ]

    rows = []
    events = []

    for method in METHODS:

        true_state = initial.copy()
        deployed = tuple()

        non_cdp_actions = 0
        action_changes = 0
        total_compute = 0.0

        for stage, duration in enumerate(durations):

            # Controller receives only observed state.
            observed = observe(
                true_state, mode, key, stage
            )

            horizon = min(
                duration, HMAX
            )

            training = ref.tapes(
                [
                    seed(
                        "train", *key, stage, k
                    )
                    for k in range(KTRAIN)
                ],
                horizon,
                N
            )

            confirmation = ref.tapes(
                [
                    seed(
                        "confirm", *key, stage, k
                    )
                    for k in range(KGUARD)
                ],
                horizon,
                N
            )

            clock = time.perf_counter()

            candidates = policies(
                observed,
                adj,
                weights[pattern],
                degree,
                budget
            )

            (
                target,
                policy,
                accepted,
                rollout_count
            ) = choose(
                method,
                observed,
                candidates,
                training,
                confirmation,
                adj,
                weights[pattern],
                budget
            )

            elapsed = (
                time.perf_counter()-clock
            )

            total_compute += elapsed

            non_cdp_actions += int(
                target != candidates["CDP"]
            )

            action_changes += int(
                target != deployed
            )

            # Same external random tape per paired case.
            external = np.random.default_rng(
                seed(
                    "outer", *key, stage
                )
            ).random(
                (duration, 3, N)
            )

            wait = min(
                delay, duration
            )

            # During communication/actuation delay,
            # the previous protection remains in force.
            if wait:
                true_state = ref.propagate(
                    true_state,
                    deployed,
                    external[:wait],
                    adj
                )

            if wait < duration:
                true_state = ref.propagate(
                    true_state,
                    target,
                    external[wait:],
                    adj
                )

            deployed = target

            events.append({
                **dict(zip(KEYS, case)),
                "method": method,
                "stage": stage,
                "policy": policy,
                "observed_I": int(
                    (observed == 2).sum()
                ),
                "unique_candidates": len(
                    set(candidates.values())
                ),
                "rollout_count": rollout_count,
                "decision_time_s": elapsed,
                "non_CDP": int(
                    target != candidates["CDP"]
                ),
                "protected_nodes": json.dumps(
                    target
                )
            })

        # PAS is evaluated on actual simulated state.
        rows.append({
            **dict(zip(KEYS, case)),
            "method": method,
            "PAS": ref.pas(
                true_state, weights[pattern]
            ),
            "attack_failed": int(
                (true_state == 3).any()
            ),
            "terminal_I": int(
                (true_state == 2).sum()
            ),
            "non_CDP_actions": non_cdp_actions,
            "action_changes": action_changes,
            "decision_compute_s": total_compute
        })

    return rows, events


# --------------------------------------------------
# 6. Frozen protocol
# --------------------------------------------------

def make_protocol(args):

    sources = [
        "revision_experiments/"
        "47_final_closed_loop_validation.py",
        "src/drad_mpc.py",
        "src/seid_malware.py",
        Path(__file__).name
    ]

    return {
        "phase": args.phase,
        "graph_seeds": list(
            range(42, 42+args.graphs)
        ),
        "scenario_seeds": list(
            range(
                args.seed_first,
                args.seed_first+args.seeds
            )
        ),
        "topologies": TOPOLOGIES,
        "patterns": PATTERNS,
        "budgets": BUDGETS,
        "onsets": ONSETS,
        "observation_modes": list(OBS_MODES),
        "actuation_delay_steps": DELAYS,
        "alpha": ALPHA,
        "methods": METHODS,
        "KTRAIN": KTRAIN,
        "KGUARD": KGUARD,
        "source_sha256": {
            str(p): sha(ROOT/p)
            for p in sources
        }
    }


# --------------------------------------------------
# 7. Paired analysis and clustered intervals
# --------------------------------------------------

def analyze(df, events, target):

    figdir = target/"figures"
    figdir.mkdir(exist_ok=True)

    wide = df.pivot(
        index=KEYS,
        columns="method",
        values="PAS"
    )

    if wide.isna().any().any():
        raise RuntimeError(
            "Missing matched outcomes"
        )

    comparisons = [
        ("CDP", "CF_LOCAL"),
        ("CF_STATIC", "CF_LOCAL"),
        ("CF_LOCAL_NO_GUARD", "CF_LOCAL"),
        ("STATIC_RANK", "LOCAL_RANK")
    ]

    rng = np.random.default_rng(54123)
    rows = []

    for (
        topology, mode, delay
    ), group in wide.groupby(
        level=["topology", "mode", "delay"]
    ):

        for a, b in comparisons:

            delta = (
                group[a]-group[b]
            ).rename("gain").reset_index()

            # Cluster by graph realization.
            pergraph = (
                delta.groupby("graph_seed")["gain"]
                .mean().to_numpy()
            )

            boots = []

            for _ in range(2500):
                sample = rng.choice(
                    pergraph,
                    size=len(pergraph),
                    replace=True
                )
                boots.append(sample.mean())

            lo, hi = np.quantile(
                boots,
                [.025, .975]
            )

            rows.append({
                "topology": topology,
                "mode": mode,
                "delay": delay,
                "comparison": f"{b} vs {a}",
                "n_scenarios": len(delta),
                "n_graphs": len(pergraph),
                "mean_PAS_gain": float(
                    delta.gain.mean()
                ),
                "graph_cluster_CI_low": float(lo),
                "graph_cluster_CI_high": float(hi),
                "wins": int(
                    (delta.gain > 1e-12).sum()
                ),
                "losses": int(
                    (delta.gain < -1e-12).sum()
                ),
                "ties": int(
                    (abs(delta.gain) <= 1e-12).sum()
                )
            })

    stats = pd.DataFrame(rows)

    stats.to_csv(
        target /
        "paired_by_topology_observation_delay.csv",
        index=False
    )

    summary = (
        df.groupby(
            [
                "topology",
                "mode",
                "delay",
                "method"
            ]
        )
        .agg(
            n=("PAS", "size"),
            mean_PAS=("PAS", "mean"),
            P95_PAS=(
                "PAS",
                lambda x: x.quantile(.95)
            ),
            mean_compute_s=(
                "decision_compute_s",
                "mean"
            ),
            mean_actions=(
                "non_CDP_actions",
                "mean"
            )
        )
        .reset_index()
    )

    summary.to_csv(
        target/"all_method_performance.csv",
        index=False
    )

    timing = events.groupby(
        [
            "topology", "mode",
            "delay", "method"
        ]
    ).agg(
        n=("decision_time_s", "size"),
        mean_s=("decision_time_s", "mean"),
        P95_s=(
            "decision_time_s",
            lambda x: x.quantile(.95)
        ),
        max_s=("decision_time_s", "max")
    )

    timing.to_csv(
        target/"decision_timing.csv"
    )

    # Blind negative control.
    blind = df[
        df["mode"] == "blind"
    ].pivot(
        index=KEYS,
        columns="method",
        values="PAS"
    )

    difference = float(
        abs(
            blind["STATIC_RANK"]
            -
            blind["LOCAL_RANK"]
        ).max()
    )

    checks = {
        "blind_static_vs_local_max_PAS_difference":
            difference,
        "PAS_lower_is_better": True,
        "physical_weight_basis":
            "Fixed IEEE118 coupling proxies",
        "runtime_scope":
            "Controller computation, not end-to-end"
    }

    (
        target/"CHECKS.json"
    ).write_text(
        json.dumps(checks, indent=2)
    )

    if difference > 1e-12:
        raise RuntimeError(
            "Blind negative control failed"
        )

    # Publication-style topology forest panels.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False
    })

    fig, axes = plt.subplots(
        1, 3,
        figsize=(13.8, 4.8),
        layout="constrained"
    )

    order = [
        "oracle", "partial80",
        "partial50", "blind"
    ]

    for ax, topology in zip(
        axes, TOPOLOGIES
    ):

        selected = stats[
            (stats.topology == topology)
            &
            (stats.delay == 0)
            &
            (
                stats.comparison
                == "CF_LOCAL vs CF_STATIC"
            )
        ].set_index("mode").reindex(order)

        ax.axvline(
            0,
            color="#91A6B3",
            linestyle="--",
            linewidth=1
        )

        for i, row in enumerate(
            selected.itertuples()
        ):

            effect = row.mean_PAS_gain

            ax.errorbar(
                effect, i,
                xerr=[
                    [
                        max(
                            0,
                            effect-row.graph_cluster_CI_low
                        )
                    ],
                    [
                        max(
                            0,
                            row.graph_cluster_CI_high-effect
                        )
                    ]
                ],
                fmt="o",
                color="#197F85",
                markersize=5,
                capsize=3
            )

        ax.set_yticks(
            range(len(order)),
            order
        )
        ax.invert_yaxis()

        ax.set_xlabel(
            "PAS(CF_STATIC) - PAS(CF_LOCAL)"
        )

        ax.set_title(
            f"{topology} | matched topology baseline",
            loc="left",
            pad=13
        )

        ax.grid(
            axis="x",
            alpha=.15
        )

    for extension in ("pdf", "svg", "png"):
        fig.savefig(
            figdir /
            f"F54_cross_topology_forest.{extension}",
            dpi=240,
            bbox_inches="tight",
            pad_inches=.16
        )

    plt.close(fig)

    print(
        "\nEXPERIMENT SUMMARY:",
        flush=True
    )

    print(
        stats[[
            "topology", "mode", "delay",
            "comparison", "mean_PAS_gain",
            "graph_cluster_CI_low",
            "graph_cluster_CI_high"
        ]].to_string(index=False),
        flush=True
    )


# --------------------------------------------------
# 8. Execute, checkpoint and package
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        choices=["pilot", "full"],
        required=True
    )

    parser.add_argument(
        "--graphs",
        type=int,
        default=1
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=1
    )

    parser.add_argument(
        "--seed-first",
        type=int,
        default=16000
    )

    args = parser.parse_args()

    if (
        args.graphs < 1
        or args.seeds < 1
    ):
        raise ValueError(
            "Graph and scenario seeds must be positive"
        )

    name = (
        "pilot"
        if args.phase == "pilot"
        else (
            f"full_g{args.graphs}_s{args.seeds}"
            f"_start{args.seed_first}"
        )
    )

    target = OUT/name
    target.mkdir(
        parents=True,
        exist_ok=True
    )

    protocol = make_protocol(args)
    frozen = target/"FROZEN_PROTOCOL.json"

    if frozen.exists():
        if json.loads(
            frozen.read_text()
        ) != protocol:
            raise RuntimeError(
                "Protocol differs from frozen version"
            )
    else:
        frozen.write_text(
            json.dumps(
                protocol,
                sort_keys=True,
                indent=2
            )
        )

    graphs = {}
    metadata = []

    for topology in TOPOLOGIES:
        for graph_seed in protocol["graph_seeds"]:

            graph = graph_of(
                topology,
                graph_seed
            )

            adj = ref.adjacency(graph)

            degree = np.array(
                [
                    graph.degree(i)
                    for i in range(N)
                ],
                dtype=float
            )

            graphs[
                (topology, graph_seed)
            ] = (
                graph, adj, degree
            )

            metadata.append({
                "topology": topology,
                "graph_seed": graph_seed,
                "nodes": N,
                "edges": graph.number_of_edges(),
                "mean_degree": float(
                    degree.mean()
                ),
                "transitivity": nx.transitivity(
                    graph
                ),
                "connected": nx.is_connected(
                    graph
                )
            })

    pd.DataFrame(metadata).to_csv(
        target/"topology_metadata.csv",
        index=False
    )

    # Real project coupling/physical tables.
    weights = {
        pattern: ref.severity_from_pattern(
            pattern, N
        )
        for pattern in PATTERNS
    }

    cases = [
        (
            topology, graph_seed, pattern,
            budget, onset, scenario_seed,
            mode, delay
        )
        for topology in TOPOLOGIES
        for graph_seed in protocol["graph_seeds"]
        for pattern in PATTERNS
        for budget in BUDGETS
        for onset in ONSETS
        for scenario_seed in protocol["scenario_seeds"]
        for mode in OBS_MODES
        for delay in DELAYS
    ]

    checkpoint_dir = target/"checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    clock = time.perf_counter()

    for index, case in enumerate(cases, 1):

        filename = (
            "_".join(map(str, case))
            + ".json"
        )

        path = checkpoint_dir/filename

        if not path.exists():

            outcomes, decisions = run_case(
                case,
                weights,
                graphs
            )

            temp = checkpoint_dir / (
                filename + ".part"
            )

            temp.write_text(
                json.dumps({
                    "outcomes": outcomes,
                    "decisions": decisions
                })
            )

            temp.replace(path)

        if (
            index % 10 == 0
            or index == len(cases)
        ):
            print(
                f"[{args.phase}] "
                f"{index}/{len(cases)} "
                f"elapsed="
                f"{time.perf_counter()-clock:.1f}s",
                flush=True
            )

    all_outcomes = []
    all_decisions = []

    for case in cases:

        filename = (
            "_".join(map(str, case))
            + ".json"
        )

        item = json.loads(
            (checkpoint_dir/filename).read_text()
        )

        all_outcomes.extend(
            item["outcomes"]
        )
        all_decisions.extend(
            item["decisions"]
        )

    df = pd.DataFrame(
        all_outcomes
    )

    decisions = pd.DataFrame(
        all_decisions
    )

    if len(df) != len(cases)*len(METHODS):
        raise RuntimeError(
            "Incomplete experimental records"
        )

    if df.duplicated(
        KEYS+["method"]
    ).any():
        raise RuntimeError(
            "Duplicate case-method pair"
        )

    df.to_csv(
        target/"raw_outcomes.csv",
        index=False
    )

    decisions.to_csv(
        target/"stage_actions.csv",
        index=False
    )

    analyze(
        df, decisions, target
    )

    archive = target/"EVIDENCE.zip"

    with ZipFile(
        archive,
        "w",
        ZIP_DEFLATED,
        compresslevel=6
    ) as z:

        for path in target.rglob("*"):

            if (
                path.is_file()
                and path != archive
                and not path.name.endswith(".part")
            ):
                z.write(
                    path,
                    path.relative_to(ROOT)
                )

        z.write(
            Path(__file__),
            "SOURCE/tnse_exp54_final.py"
        )

    with ZipFile(archive) as z:
        if z.testzip() is not None:
            raise RuntimeError(
                "ZIP integrity check failed"
            )

    print(
        "\nEXP54 FINISHED SUCCESSFULLY",
        "\nOUTPUT:",
        target.resolve(),
        flush=True
    )


if __name__ == "__main__":
    main()
