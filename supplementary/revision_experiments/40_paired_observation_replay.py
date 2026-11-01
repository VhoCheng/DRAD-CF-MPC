from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "revision_experiments"
RES = EXP / "results"
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))
import seid_malware as seid
import drad_mpc as mpc
import run_experiments_core as core

spec = importlib.util.spec_from_file_location(
    "exp38_reference",
    EXP / "38_same_state_validation.py"
)
ref = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ref
spec.loader.exec_module(ref)

INPUT = RES / "39_mechanism_full/per_decision.csv"
HISTORY = RES / "26_multiscale_full/per_run.csv"

METHODS = [
    "CDP", "FULL", "NO_GUARD",
    "NO_DYNAMIC", "NO_CF"
]

CONDITIONS = [
    "clean", "delay_10", "delay_30",
    "loss_20", "false_positive_05",
    "combined_delay30_loss20"
]

def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()

def parse_nodes(value):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return tuple(sorted(int(v) for v in value))

def run_scenario(task):
    sid, records, seed_map, pilot = task
    df = pd.DataFrame(records)
    n = int(df["case_size"].iloc[0])
    pattern = str(df["pattern"].iloc[0])
    info = float(df["info"].iloc[0])
    budget = int(df["budget"].iloc[0])
    onset = int(df["onset"].iloc[0])

    graph = nx.barabasi_albert_graph(
        n=n, m=2, seed=42
    )
    adjacency = nx.to_scipy_sparse_array(
        graph,
        nodelist=list(range(n)),
        format="csr",
        dtype=np.int16
    )

    tables = (
        RES / "25_multiscale_networks"
        / f"case{n}" / "outputs/tables"
    )

    if not tables.exists():
        raise FileNotFoundError(tables)

    impact = mpc.compute_physics_impact_table(
        pattern=pattern,
        table_dir=str(tables)
    )

    weights = np.zeros(n, dtype=float)
    for _, row in impact.iterrows():
        weights[int(row["comm_node"])] = float(
            row["impact_score_raw"]
        )

    denominator = float(weights.sum()) + 1e-9
    if denominator <= 1e-9:
        raise RuntimeError("Invalid impact weights")

    result = []

    for run_index, block in df.groupby("run_index"):
        run_index = int(run_index)

        if pilot and run_index != 0:
            continue

        key = (sid, run_index)
        if key not in seed_map:
            raise RuntimeError(
                f"Missing historical seed {key}"
            )
        seed = seed_map[key]

        if len(block) != 30:
            raise RuntimeError(
                f"Expected 30 actions for {key}"
            )

        preview = core.get_preview_state(
            Gc=graph,
            onset_time=onset,
            attack_information_ratio=info,
            infection_rate=0.01,
            activation_rate=0.15,
            detection_rate=0.00001,
            seed=seed
        )

        true_state = np.zeros(n, dtype=np.int8)

        for field, state_id in [
            ("exposed_nodes", seid.EXPOSED),
            ("active_infectious_nodes", seid.INFECTIOUS),
            ("detected_nodes", seid.DETECTED)
        ]:
            true_state[
                np.asarray(preview[field], dtype=int)
            ] = state_id

        preview_time = max(1, int(onset * 0.4))
        remaining = onset - preview_time

        # The same held-out future random draws
        # are reused for every method and condition.
        test_seeds = [
            seed + 98000000 + j * 7001
            for j in range(3)
        ]

        cache = {}

        for _, action in block.iterrows():
            condition = str(action["condition"])
            method = str(action["method"])
            nodes = parse_nodes(
                action["protected_nodes"]
            )

            if len(nodes) != budget:
                raise RuntimeError(
                    "Protected node count mismatch"
                )

            if condition not in CONDITIONS:
                raise RuntimeError(
                    "Unknown observation condition"
                )
            if method not in METHODS:
                raise RuntimeError(
                    "Unknown method"
                )

            if nodes not in cache:
                losses = []
                failures = []

                for future_seed in test_seeds:
                    final_state, failed = ref.advance(
                        true_state,
                        adjacency,
                        nodes,
                        future_seed,
                        remaining
                    )

                    infected = np.flatnonzero(
                        final_state == seid.INFECTIOUS
                    )

                    pas = (
                        0.0 if failed else
                        float(
                            weights[infected].sum()
                            / denominator
                        )
                    )

                    losses.append(pas)
                    failures.append(int(failed))

                cache[nodes] = (
                    float(np.mean(losses)),
                    float(np.mean(failures))
                )

            pas, failure = cache[nodes]

            result.append({
                "scenario_id": sid,
                "case_size": n,
                "pattern": pattern,
                "run_index": run_index,
                "condition": condition,
                "method": method,
                "PAS": pas,
                "attack_failure_rate": failure,
                "protected_nodes": list(nodes)
            })

    return {"scenario_id": sid, "records": result}

def holm(values):
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(
        1.0,
        np.maximum.accumulate(
            p[order] * (len(p) - np.arange(len(p)))
        )
    )
    return adjusted

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=["pilot", "full"]
    )
    parser.add_argument(
        "--workers", type=int, default=8
    )
    args = parser.parse_args()

    decisions = pd.read_csv(INPUT)
    history = pd.read_csv(HISTORY)

    if len(decisions) != 9600 or len(history) != 960:
        raise RuntimeError(
            "Historical inputs are incomplete"
        )

    seed_map = {}
    for key, group in history.groupby(
        ["scenario_id", "run_index"]
    ):
        seeds = group["seed"].unique()
        if len(seeds) != 1:
            raise RuntimeError(
                f"Historical seed mismatch: {key}"
            )
        seed_map[tuple(map(int, key))] = int(seeds[0])

    all_tasks = [
        (
            int(sid),
            block.to_dict("records"),
            seed_map,
            args.mode == "pilot"
        )
        for sid, block in decisions.groupby(
            "scenario_id", sort=True
        )
    ]

    if len(all_tasks) != 80:
        raise RuntimeError(
            "Expected 80 scenarios"
        )

    if args.mode == "pilot":
        chosen = {}
        for task in all_tasks:
            n = int(task[1][0]["case_size"])
            chosen.setdefault(n, task)
        tasks = list(chosen.values())
    else:
        tasks = all_tasks

    out = RES / f"40_paired_observation_{args.mode}"
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(
        parents=True, exist_ok=True
    )

    manifest = {
        "mode": args.mode,
        "input_sha256": sha(INPUT),
        "history_sha256": sha(HISTORY),
        "driver_sha256": sha(
            Path(__file__).resolve()
        ),
        "reference_sha256": sha(
            EXP / "38_same_state_validation.py"
        ),
        "n_future_seeds": 3,
        "seed_rule": (
            "historical_seed + 98000000 + 7001*j; "
            "identical across observations and methods"
        ),
        "scope": (
            "Paired held-out re-evaluation of actions "
            "already selected in Exp39."
        )
    }

    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )
        if previous != manifest:
            raise RuntimeError(
                "Protocol changed; refusing mixed checkpoints"
            )
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8"
        )

    pending = [
        task for task in tasks
        if not (
            checkpoints /
            f"scenario_{task[0]:03d}.json"
        ).exists()
    ]

    print(
        f"EXPERIMENT 40 {args.mode}: "
        f"{len(tasks)} scenarios, "
        f"{len(pending)} pending",
        flush=True
    )

    with ProcessPoolExecutor(
        max_workers=min(
            args.workers, len(tasks), 8
        )
    ) as executor:
        jobs = {
            executor.submit(
                run_scenario, task
            ): task
            for task in pending
        }

        completed = len(tasks) - len(pending)

        for job in as_completed(jobs):
            task = jobs[job]
            payload = job.result()

            expected = 30 if args.mode == "pilot" else 120

            if len(payload["records"]) != expected:
                raise RuntimeError(
                    f"Unexpected result count: {task[0]}"
                )

            path = checkpoints / (
                f"scenario_{task[0]:03d}.json"
            )
            temporary = path.with_suffix(
                f".{os.getpid()}.tmp"
            )
            temporary.write_text(
                json.dumps(payload),
                encoding="utf-8"
            )
            temporary.replace(path)

            completed += 1
            print(
                f"[{completed}/{len(tasks)}] "
                f"scenario={task[0]}",
                flush=True
            )

    records = []
    for task in tasks:
        path = checkpoints / (
            f"scenario_{task[0]:03d}.json"
        )
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )
        expected = 30 if args.mode == "pilot" else 120

        if len(payload["records"]) != expected:
            raise RuntimeError(
                "Incomplete checkpoint"
            )
        records.extend(payload["records"])

    df = pd.DataFrame(records)
    expected_rows = (
        120 if args.mode == "pilot" else 9600
    )
    if len(df) != expected_rows:
        raise RuntimeError("Wrong final row count")

    keys = [
        "scenario_id", "run_index",
        "condition", "method"
    ]

    if df.duplicated(keys).any():
        raise RuntimeError("Duplicate evaluations")

    # CDP chooses identical actions under every
    # observation condition: paired PAS must match.
    cdp = df[df["method"] == "CDP"].pivot(
        index=["scenario_id", "run_index"],
        columns="condition",
        values="PAS"
    )

    if not np.allclose(
        cdp.to_numpy(),
        cdp["clean"].to_numpy()[:, None],
        atol=1e-12,
        rtol=0
    ):
        raise RuntimeError(
            "Common-future-randomness check failed"
        )

    df.to_csv(
        out / "per_action_paired.csv",
        index=False
    )

    aggregate = df.groupby(
        ["case_size", "condition", "method"],
        as_index=False
    ).agg(
        n=("PAS", "size"),
        mean_PAS=("PAS", "mean"),
        mean_attack_failure=(
            "attack_failure_rate", "mean"
        )
    )

    aggregate.to_csv(
        out / "aggregate.csv",
        index=False
    )

    full = df[df["method"] == "FULL"]
    pivot = full.pivot(
        index=["case_size", "scenario_id", "run_index"],
        columns="condition",
        values="PAS"
    )

    rng = np.random.default_rng(20261020)
    stats = []

    for n in sorted(df["case_size"].unique()):
        frame = pivot.loc[n]

        for condition in CONDITIONS:
            if condition == "clean":
                continue

            differences = (
                frame[condition]
                - frame["clean"]
            )

            scenario_means = differences.groupby(
                level="scenario_id"
            ).mean().to_numpy()

            boots = rng.choice(
                scenario_means,
                size=(3000, len(scenario_means)),
                replace=True
            ).mean(axis=1)

            nonzero = scenario_means[
                np.abs(scenario_means) > 1e-12
            ]

            pvalue = (
                float(wilcoxon(nonzero).pvalue)
                if len(nonzero) else 1.0
            )

            stats.append({
                "case_size": n,
                "condition": condition,
                "delta_PAS_stress_minus_clean": (
                    float(scenario_means.mean())
                ),
                "CI_low": float(
                    np.quantile(boots, 0.025)
                ),
                "CI_high": float(
                    np.quantile(boots, 0.975)
                ),
                "wilcoxon_p": pvalue,
                "scenarios_worse": int(
                    (scenario_means > 1e-12).sum()
                ),
                "scenarios_better": int(
                    (scenario_means < -1e-12).sum()
                )
            })

    stats = pd.DataFrame(stats)
    stats["holm_p_20"] = holm(
        stats["wilcoxon_p"]
    )
    stats.to_csv(
        out / "paired_stress_statistics.csv",
        index=False
    )

    print("\nEXPERIMENT 40 PAIRED STRESS RESULTS")
    print(
        stats.to_string(index=False),
        flush=True
    )
    print("\nEXPERIMENT 40 COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    main()
