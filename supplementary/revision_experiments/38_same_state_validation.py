from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import hashlib
import inspect
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import seid_malware as seid
import drad_mpc as mpc
import run_experiments_core as core
import dynamic_risk_defense as risk

BASE = ROOT / "revision_experiments/results"
HISTORY = BASE / "26_multiscale_full/per_run.csv"
NETWORKS = BASE / "25_multiscale_networks"

METHODS = ["CDP", "Legacy-K5", "Confirm15-Robust"]
COUPLINGS = ["DDAC", "DDDC", "DCAC", "DCDC", "RC"]


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def advance(states, adj, protected, seed, steps):
    """
    New post-observation SEID continuation protocol.

    Common random numbers are indexed by
    time, node and transition type.

    Protection only reduces future S->E transmission.
    It does not retroactively remove E or I nodes.
    """
    rng = np.random.default_rng(seed)
    states = states.copy()
    protected_mask = np.zeros(len(states), dtype=bool)
    protected_mask[list(protected)] = True
    failed = False

    for _ in range(steps):
        infectious = states == seid.INFECTIOUS
        exposed = states == seid.EXPOSED
        susceptible = states == seid.SUSCEPTIBLE

        u_inf = rng.random(len(states))
        u_act = rng.random(len(states))
        u_det = rng.random(len(states))

        neighboring_I = np.asarray(
            adj @ infectious.astype(np.int16)
        ).reshape(-1)

        factor = np.where(protected_mask, 0.2, 1.0)

        probability = (
            1.0 - (1.0 - 0.01 * factor) ** neighboring_I
        )

        new_E = susceptible & (u_inf < probability)
        new_I = exposed & (u_act < 0.15)
        new_D = infectious & (u_det < 0.00001)

        updated = states.copy()
        updated[new_E] = seid.EXPOSED
        updated[new_I] = seid.INFECTIOUS
        updated[new_D] = seid.DETECTED

        states = updated

        if new_D.any():
            failed = True
            break

    return states, failed


def run_scenario(task):
    sid, rows, repetitions = task

    df = pd.DataFrame(rows)

    n = int(df["case_size"].iloc[0])
    pattern = str(df["pattern"].iloc[0])
    info = float(df["info"].iloc[0])
    budget = int(df["budget"].iloc[0])
    onset = int(df["onset"].iloc[0])

    case_dir = NETWORKS / f"case{n}"
    table_dir = case_dir / "outputs/tables"

    if not table_dir.exists():
        raise FileNotFoundError(table_dir)

    previous_cwd = Path.cwd()
    original_graph_builder = risk.build_comm_graph

    graph = nx.barabasi_albert_graph(
        n=n, m=2, seed=42
    )

    adj = nx.to_scipy_sparse_array(
        graph,
        nodelist=list(range(n)),
        format="csr",
        dtype=np.int16,
    )

    records = []
    checks = 0

    try:
        os.chdir(case_dir)

        # Match the topology adaptation from Exp26.
        risk.build_comm_graph = (
            lambda n_nodes=118, seed=42: graph
        )

        impact = mpc.compute_physics_impact_table(
            pattern,
            table_dir="outputs/tables",
        )

        impact_weights = np.zeros(n)

        for _, row in impact.iterrows():
            impact_weights[int(row["comm_node"])] = (
                float(row["impact_score_raw"])
            )

        denominator = impact_weights.sum() + 1e-9

        if denominator <= 0:
            raise RuntimeError("Invalid impact scores")

        for run_index, group in df.groupby("run_index"):
            if set(group["variant"]) != set(METHODS):
                raise RuntimeError(
                    "Incomplete historical methods"
                )

            seeds = group["seed"].unique()

            if len(seeds) != 1:
                raise RuntimeError("Seed mismatch")

            seed = int(seeds[0])

            # This preview is from the original code.
            preview = core.get_preview_state(
                Gc=graph,
                onset_time=onset,
                attack_information_ratio=info,
                infection_rate=0.01,
                activation_rate=0.15,
                detection_rate=0.00001,
                seed=seed,
            )

            t_preview = max(1, int(onset * 0.4))
            remaining_steps = onset - t_preview

            state = np.full(
                n, seid.SUSCEPTIBLE, dtype=np.int8
            )

            for col, code in (
                ("exposed_nodes", seid.EXPOSED),
                ("active_infectious_nodes", seid.INFECTIOUS),
                ("detected_nodes", seid.DETECTED),
            ):
                state[
                    np.asarray(preview[col], dtype=int)
                ] = code

            candidates = mpc.generate_candidate_policies(
                pattern=pattern,
                defense_budget=budget,
                attack_information_ratio=info,
                active_nodes=preview["active_infectious_nodes"],
                exposed_nodes=preview["exposed_nodes"],
                table_dir="outputs/tables",
                seed=seed,
            )

            if "CDP" not in candidates:
                candidates["CDP"] = (
                    mpc.select_protected_nodes(
                        strategy="CDP",
                        defense_budget=budget,
                        pattern=pattern,
                        seed=seed,
                        table_dir="outputs/tables",
                    )
                )

            actions = {}

            for _, row in group.iterrows():
                variant = str(row["variant"])
                policy = str(row["selected_policy"])

                if policy not in candidates:
                    raise RuntimeError(
                        f"Missing historical policy {policy}"
                    )

                nodes = tuple(sorted(
                    int(v) for v in candidates[policy]
                ))

                if len(nodes) != budget:
                    raise RuntimeError(
                        f"Invalid defense budget: {variant}"
                    )

                actions[variant] = (policy, nodes)

            for rep in range(repetitions):
                # Same exogenous post-preview events
                # for each protection policy.
                post_seed = (
                    seed + 9000000 + rep * 1009
                )

                for variant in METHODS:
                    policy, nodes = actions[variant]

                    outcome, failed = advance(
                        state,
                        adj,
                        nodes,
                        post_seed,
                        remaining_steps,
                    )

                    infected = np.flatnonzero(
                        outcome == seid.INFECTIOUS
                    )

                    pas = (
                        0.0 if failed else
                        float(
                            impact_weights[infected].sum()
                            / denominator
                        )
                    )

                    if not 0.0 <= pas <= 1.000001:
                        raise RuntimeError(
                            "PAS outside [0,1]"
                        )

                    records.append({
                        "scenario_id": sid,
                        "case_size": n,
                        "pattern": pattern,
                        "info": info,
                        "budget": budget,
                        "onset": onset,
                        "run_index": int(run_index),
                        "repetition": rep,
                        "variant": variant,
                        "selected_policy": policy,
                        "protected_nodes": list(nodes),
                        "preview_time": t_preview,
                        "preview_I": int(
                            np.sum(
                                state == seid.INFECTIOUS
                            )
                        ),
                        "preview_E": int(
                            np.sum(
                                state == seid.EXPOSED
                            )
                        ),
                        "failed": int(failed),
                        "PAS": pas,
                        "infected_end": len(infected),
                    })

                    checks += 1

    finally:
        risk.build_comm_graph = original_graph_builder
        os.chdir(previous_cwd)

    return {
        "scenario_id": sid,
        "records": records,
        "checks": checks,
    }


def atomic_save(path, payload):
    temp = path.with_suffix(
        f".{os.getpid()}.tmp"
    )
    temp.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["pilot", "full"],
        required=True,
    )
    parser.add_argument(
        "--workers", type=int, default=8
    )
    args = parser.parse_args()

    if not HISTORY.exists():
        raise FileNotFoundError(HISTORY)

    original_signature = inspect.signature(
        seid.simulate_seid_once
    )

    print(
        "Original SEID supports initial_states:",
        "initial_states" in original_signature.parameters,
        flush=True,
    )

    history = pd.read_csv(HISTORY)

    if len(history) != 960:
        raise RuntimeError(
            "Expected 960 historical records"
        )

    tasks = [
        (
            int(sid),
            group.to_dict("records"),
            3 if args.mode == "full" else 2,
        )
        for sid, group in history.groupby(
            "scenario_id", sort=True
        )
    ]

    if args.mode == "pilot":
        first_by_scale = {}
        for task in tasks:
            n = int(task[1][0]["case_size"])
            first_by_scale.setdefault(n, task)

        tasks = []

        for sid, rows, reps in first_by_scale.values():
            tasks.append((
                sid,
                [
                    r for r in rows
                    if int(r["run_index"]) == 0
                ],
                reps,
            ))

    run_dir = BASE / (
        f"38_same_state_{args.mode}"
    )

    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(
        parents=True, exist_ok=True
    )

    manifest = {
        "experiment": "same-state post-preview action evaluation",
        "mode": args.mode,
        "history_sha256": sha(HISTORY),
        "driver_sha256": sha(
            Path(__file__).resolve()
        ),
        "sources": {
            p: sha(ROOT / "src" / p)
            for p in (
                "seid_malware.py",
                "drad_mpc.py",
                "run_experiments_core.py",
                "dynamic_risk_defense.py",
            )
        },
        "scenario_ids": [t[0] for t in tasks],
        "post_preview_repetitions": (
            3 if args.mode == "full" else 2
        ),
        "scope": (
            "Selected historic actions evaluated from "
            "identical original preview node states. "
            "Intervention takes effect after preview. "
            "This is not the original pre-infection protocol."
        ),
    }

    manifest_path = run_dir / "manifest.json"

    if manifest_path.exists():
        previous = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if previous != manifest:
            raise RuntimeError(
                "Protocol changed: cannot reuse checkpoints"
            )
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

    pending = [
        task for task in tasks
        if not (
            checkpoints
            / f"scenario_{task[0]:03d}.json"
        ).exists()
    ]

    print("=" * 70)
    print("TNSE EXPERIMENT 38: SAME-STATE ACTION VALIDATION")
    print("Mode:", args.mode)
    print("Scenarios:", len(tasks))
    print("Pending:", len(pending))
    print("=" * 70, flush=True)

    with ProcessPoolExecutor(
        max_workers=min(
            args.workers, len(tasks), 8
        )
    ) as executor:
        futures = {
            executor.submit(
                run_scenario, task
            ): task
            for task in pending
        }

        for future in as_completed(futures):
            task = futures[future]

            try:
                result = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"Failed scenario: {task[0]}"
                ) from exc

            if int(result["scenario_id"]) != task[0]:
                raise RuntimeError(
                    "Checkpoint identity mismatch"
                )

            expected = (
                len(task[1]) * task[2]
            )

            if (
                result["checks"] != expected
                or len(result["records"]) != expected
            ):
                raise RuntimeError(
                    "Incomplete action evaluation"
                )

            atomic_save(
                checkpoints
                / f"scenario_{task[0]:03d}.json",
                result,
            )

            print(
                f"[saved] scenario={task[0]} "
                f"rows={expected}",
                flush=True,
            )

    all_records = []

    for task in tasks:
        path = checkpoints / (
            f"scenario_{task[0]:03d}.json"
        )
        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        expected = len(task[1]) * task[2]

        if (
            data["scenario_id"] != task[0]
            or len(data["records"]) != expected
            or data["checks"] != expected
        ):
            raise RuntimeError(
                f"Bad checkpoint: {task[0]}"
            )

        all_records.extend(data["records"])

    result = pd.DataFrame(all_records)

    if result.duplicated([
        "scenario_id",
        "run_index",
        "repetition",
        "variant",
    ]).any():
        raise RuntimeError(
            "Duplicate action evaluations"
        )

    result.to_csv(
        run_dir / "per_run.csv",
        index=False,
    )

    aggregate = (
        result.groupby(
            ["case_size", "variant"],
            as_index=False,
        )
        .agg(
            trials=("PAS", "size"),
            mean_PAS=("PAS", "mean"),
            attack_failure_rate=(
                "failed", "mean"
            ),
            mean_infected_end=(
                "infected_end", "mean"
            ),
        )
    )

    aggregate.to_csv(
        run_dir / "aggregate.csv",
        index=False,
    )

    print("\nSAME-STATE PHYSICAL PROXY RESULTS")
    print(
        aggregate.to_string(index=False)
    )

    rows = []
    rng = np.random.default_rng(20261020)

    for n in sorted(
        result["case_size"].unique()
    ):
        sub = result[
            result["case_size"] == n
        ]

        scene = (
            sub.groupby(
                ["scenario_id", "variant"]
            )["PAS"].mean()
            .unstack("variant")
        )

        for method, baseline in (
            ("Legacy-K5", "CDP"),
            ("Confirm15-Robust", "CDP"),
            ("Confirm15-Robust", "Legacy-K5"),
        ):
            difference = (
                scene[baseline]
                - scene[method]
            ).to_numpy()

            sampled = rng.choice(
                difference,
                size=(4000, len(difference)),
                replace=True,
            ).mean(axis=1)

            nonzero = difference[
                np.abs(difference) > 1e-12
            ]

            pvalue = (
                float(wilcoxon(nonzero).pvalue)
                if len(nonzero) else 1.0
            )

            rows.append({
                "case_size": n,
                "comparison": (
                    f"{method} vs {baseline}"
                ),
                "mean_PAS_gain": (
                    float(difference.mean())
                ),
                "CI_low": float(
                    np.quantile(sampled, 0.025)
                ),
                "CI_high": float(
                    np.quantile(sampled, 0.975)
                ),
                "wilcoxon_p": pvalue,
                "better_scenarios": int(
                    (difference > 1e-12).sum()
                ),
                "worse_scenarios": int(
                    (difference < -1e-12).sum()
                ),
            })

    stats = pd.DataFrame(rows)

    order = np.argsort(
        stats["wilcoxon_p"].to_numpy()
    )
    p = stats["wilcoxon_p"].to_numpy()
    corrected = np.empty(len(p))
    corrected[order] = np.minimum(
        1.0,
        np.maximum.accumulate(
            p[order] * (
                len(p) - np.arange(len(p))
            )
        ),
    )
    stats["holm_adjusted_p"] = corrected

    stats.to_csv(
        run_dir / "paired_statistics.csv",
        index=False,
    )

    print("\nSAME-STATE PAIRED STATISTICS")
    print(
        stats.to_string(index=False)
    )

    print(
        "\nEXPERIMENT 38 COMPLETED SUCCESSFULLY",
        flush=True,
    )


if __name__ == "__main__":
    main()
