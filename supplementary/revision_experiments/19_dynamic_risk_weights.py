from pathlib import Path
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "revision_experiments/results"
sys.path.insert(0, str(ROOT / "src"))

import run_experiments_core as core
import drad_mpc as mpc

HELPER = (
    ROOT / "revision_experiments"
    / "14_parallel_replication.py"
)

spec = importlib.util.spec_from_file_location(
    "validated_exp14", HELPER
)
e14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e14)

FEATURES = [
    "comm_degree",
    "comm_betweenness",
    "power_degree",
    "power_capacity",
    "infection_risk",
    "uncertainty",
]

PROFILES = {
    "Original": [0.36, 0.18, 0.08, 0.08, 0.20, 0.10],
    "Balanced": [0.20, 0.15, 0.15, 0.15, 0.20, 0.15],
    "Cyber-Priority": [0.45, 0.25, 0.04, 0.04, 0.14, 0.08],
    "Physical-Priority": [0.20, 0.10, 0.20, 0.20, 0.20, 0.10],
    "Infection-Priority": [0.24, 0.14, 0.06, 0.06, 0.40, 0.10],
    "Uncertainty-Priority": [0.25, 0.15, 0.07, 0.07, 0.20, 0.26],
}

VARIANTS = ["CDP"] + list(PROFILES)

PILOT = [
    ("DDAC", 0.50, 60, 150),
    ("RC", 0.50, 40, 120),
]

FULL = list(product(
    ["DDAC", "DDDC", "DCAC", "DCDC", "RC"],
    [0.50],
    [40, 80],
    [90, 180],
))


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def weighted_drad_nodes(
    pattern, budget, info, active, exposed, seed, values
):
    weights = dict(zip(FEATURES, values))

    risk = mpc.compute_dynamic_risk_score(
        pattern=pattern,
        attack_information_ratio=info,
        active_nodes=active,
        exposed_nodes=exposed,
        protected_nodes=None,
        weights=weights,
        table_dir="outputs/tables",
        seed=seed,
    )

    impact = mpc.compute_physics_impact_table(
        pattern=pattern,
        table_dir="outputs/tables",
    )

    df = risk.merge(
        impact[["comm_node", "impact_score_norm"]],
        on="comm_node",
        how="left",
    )

    df["drad_score"] = (
        0.80 * df["dynamic_risk_score_norm"]
        + 0.20 * df["impact_score_norm"]
    )

    return mpc._top_by_score(
        df, "drad_score", budget
    )


def run_task(task):
    sid, scenario, n_runs, base_seed = task
    os.chdir(ROOT)

    pattern, info, budget, onset = scenario

    graph = core.build_comm_graph(
        n_nodes=118, seed=42
    )

    results = {v: [] for v in VARIANTS}
    decisions = {v: [] for v in VARIANTS}
    candidate_changes = {v: [] for v in VARIANTS}

    detail_rows = []

    original_function = mpc.candidate_drad_score_nodes

    for r in range(n_runs):
        seed = (
            base_seed + r
            + onset * 1000
            + budget * 17
        )

        preview = core.get_preview_state(
            Gc=graph,
            onset_time=onset,
            attack_information_ratio=info,
            infection_rate=0.01,
            activation_rate=0.15,
            detection_rate=1e-5,
            seed=seed,
        )

        active = preview["active_infectious_nodes"]
        exposed = preview["exposed_nodes"]

        original_nodes = tuple(sorted(
            original_function(
                pattern, budget, info,
                active, exposed,
                "outputs/tables", seed
            )
        ))

        # Verify that the independently reconstructed
        # default-weight formula matches the original code.
        reconstructed_nodes = tuple(sorted(
            weighted_drad_nodes(
                pattern, budget, info,
                active, exposed, seed,
                PROFILES["Original"],
            )
        ))

        if original_nodes != reconstructed_nodes:
            raise RuntimeError(
                "DEFAULT RISK WEIGHT RECONSTRUCTION FAILED"
            )

        cf_seeds = [
            seed + 777000 + 10007 * k
            + mpc.stable_int_hash(
                f"{pattern}-{onset}-{budget}",
                100000,
            )
            for k in range(5)
        ]

        # Cache CF evaluations by protected-node set.
        # Every profile uses identical CF random seeds.
        cf_cache = {}
        actual_cache = {}

        selected = {"CDP": "CDP"}
        candidate_maps = {}
        raw_candidate_nodes = {
            "Original": original_nodes
        }

        for name, values in PROFILES.items():
            if name == "Original":
                drad_nodes = original_nodes
            else:
                drad_nodes = tuple(sorted(
                    weighted_drad_nodes(
                        pattern, budget, info,
                        active, exposed, seed, values
                    )
                ))

            raw_candidate_nodes[name] = drad_nodes

            try:
                if name != "Original":
                    # Only DRAD_SCORE is changed.
                    # Other candidate strategies stay original.
                    def replacement(*args, **kwargs):
                        return list(drad_nodes)

                    mpc.candidate_drad_score_nodes = replacement

                candidates = mpc.generate_candidate_policies(
                    pattern=pattern,
                    defense_budget=budget,
                    attack_information_ratio=info,
                    active_nodes=active,
                    exposed_nodes=exposed,
                    table_dir="outputs/tables",
                    seed=seed,
                )

            finally:
                mpc.candidate_drad_score_nodes = (
                    original_function
                )

            if "CDP" not in candidates:
                raise RuntimeError(
                    "CDP candidate is missing"
                )

            metrics = {}

            for policy, nodes in candidates.items():
                key = tuple(sorted(nodes))

                if key not in cf_cache:
                    cf_cache[key] = (
                        mpc.evaluate_candidate_with_common_random_numbers(
                            candidate_nodes=list(key),
                            common_seeds=cf_seeds,
                            Gc=graph,
                            pattern=pattern,
                            attack_information_ratio=info,
                            defense_budget=budget,
                            onset_time=onset,
                            infection_rate=0.01,
                            activation_rate=0.15,
                            detection_rate=1e-5,
                        )
                    )

                metrics[policy] = cf_cache[key]

            policy, _ = e14.e11.helper.choose_policy(
                metrics,
                eta=0.0,
                kappa=0.5,
            )

            selected[name] = policy
            candidate_maps[name] = candidates

        # Determine actual protected-node sets selected
        # under each profile.
        default_policy = selected["Original"]

        for variant in VARIANTS:
            if variant == "CDP":
                policy = "CDP"
                nodes = candidate_maps["Original"]["CDP"]
                change = False
            else:
                policy = selected[variant]
                nodes = candidate_maps[variant][policy]
                change = (
                    raw_candidate_nodes[variant]
                    != original_nodes
                )

            node_key = tuple(sorted(nodes))

            if node_key not in actual_cache:
                actual_cache[node_key] = (
                    e14.e11.helper.actual_outcome(
                        graph,
                        scenario,
                        list(node_key),
                        seed,
                    )
                )

            outcome = actual_cache[node_key]

            results[variant].append(outcome)
            decisions[variant].append(policy)
            candidate_changes[variant].append(change)

            detail_rows.append({
                "scenario_id": sid,
                "pattern": pattern,
                "info": info,
                "budget": budget,
                "onset": onset,
                "run_index": r,
                "seed": seed,
                "variant": variant,
                "selected_policy": policy,
                "default_selected_policy": default_policy,
                "candidate_changed": int(change),
                "policy_changed": int(
                    policy != default_policy
                ),
                "protected_nodes": "|".join(
                    map(str, node_key)
                ),
                "PAS": float(outcome["PAS"]),
                "failed": int(outcome["failed"]),
                "RT": float(outcome["RT"]),
                "RI": float(outcome["RI"]),
                "RE": float(outcome["RE"]),
            })

    baseline = e14.e11.helper.aggregate(
        results["CDP"]
    )

    summary_rows = []

    for name in VARIANTS:
        metrics = e14.e11.helper.aggregate(
            results[name]
        )

        rows = [
            row for row in detail_rows
            if row["variant"] == name
        ]

        summary_rows.append({
            "scenario_id": sid,
            "pattern": pattern,
            "info": info,
            "budget": budget,
            "onset": onset,
            "variant": name,
            "n_runs": n_runs,
            "EP": float(metrics["EP"]),
            "PAS": float(metrics["PAS"]),
            "PS": float(metrics["PS"]),
            "RT": float(metrics["RT"]),
            "RI": float(metrics["RI"]),
            "RE": float(metrics["RE"]),
            "switch_rate": float(np.mean([
                x != "CDP" for x in decisions[name]
            ])),
            "candidate_change_rate": float(
                np.mean(candidate_changes[name])
            ),
            "policy_change_rate": float(
                np.mean([
                    x["policy_changed"]
                    for x in rows
                ])
            ),
            "EP_gain_pct": (
                100.0
                * (baseline["EP"] - metrics["EP"])
                / baseline["EP"]
                if baseline["EP"] > 1e-12
                else None
            ),
        })

    return {
        "scenario_id": sid,
        "summary": summary_rows,
        "details": detail_rows,
    }


def check_pilot(details):
    ref_path = (
        OUT / "14_replication_pilot" / "per_run.csv"
    )

    if not ref_path.exists():
        raise RuntimeError(
            "Experiment 14 reference missing"
        )

    reference = pd.read_csv(ref_path)
    checked = 0

    for row in details:
        if row["variant"] not in ["CDP", "Original"]:
            continue

        expected_variant = (
            "CDP" if row["variant"] == "CDP"
            else "Legacy-K5"
        )

        matched = reference[
            (reference["scenario_id"] == row["scenario_id"])
            & (reference["run_index"] == row["run_index"])
            & (reference["variant"] == expected_variant)
        ]

        if len(matched) != 1:
            raise RuntimeError(
                "Pilot reference row not found"
            )

        original = matched.iloc[0]

        if (
            row["selected_policy"]
            != original["selected_policy"]
        ):
            raise RuntimeError(
                "PILOT POLICY MISMATCH"
            )

        for metric in [
            "PAS", "failed", "RT", "RI", "RE"
        ]:
            if not np.isclose(
                float(row[metric]),
                float(original[metric]),
                atol=1e-10,
                rtol=0,
            ):
                raise RuntimeError(
                    f"PILOT RESULT MISMATCH: {metric}"
                )

        checked += 1

    if checked != 12:
        raise RuntimeError(
            f"Expected 12 checks, got {checked}"
        )

    print(
        "DYNAMIC-RISK PILOT EQUIVALENCE PASSED: "
        "12/12 original outcomes reproduced.",
        flush=True,
    )


def save_json(path, result):
    temporary = path.with_suffix(
        f".{os.getpid()}.tmp"
    )

    temporary.write_text(
        json.dumps(result),
        encoding="utf-8",
    )

    temporary.replace(path)


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
        default=8,
    )

    args = parser.parse_args()

    e14.e11.helper.check_original()

    for name, values in PROFILES.items():
        if abs(sum(values) - 1.0) > 1e-10:
            raise RuntimeError(
                f"Invalid weight sum: {name}"
            )

    if args.mode == "pilot":
        scenarios = PILOT
        n_runs = 3
        base_seed = 42
        workers = 2
    else:
        scenarios = FULL
        n_runs = 8
        base_seed = 20261205
        workers = max(1, min(args.workers, 12))

    run_dir = (
        OUT / f"19_dynamic_risk_{args.mode}"
    )

    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "experiment": "DRAD risk weight sensitivity",
        "mode": args.mode,
        "scenarios": scenarios,
        "weights": PROFILES,
        "features": FEATURES,
        "variants": VARIANTS,
        "n_runs": n_runs,
        "base_seed": base_seed,
        "gamma": 0.10,
        "kappa": 0.50,
        "K": 5,
        "scope": (
            "DRAD_SCORE feature weights only; "
            "RANK_ENSEMBLE remains original"
        ),
        "source_hashes": {
            name: sha(ROOT / "src" / name)
            for name in [
                "drad_mpc.py",
                "dynamic_risk_defense.py",
                "run_experiments_core.py",
                "seid_malware.py",
            ]
        },
        "driver_hash": sha(
            Path(__file__).resolve()
        ),
    }

    manifest_path = run_dir / "manifest.json"

    if manifest_path.exists():
        old = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )
        if old != manifest:
            raise RuntimeError(
                "Experiment protocol changed. "
                "Refusing to mix checkpoints."
            )
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

    tasks = [
        (i, scenario, n_runs, base_seed)
        for i, scenario in enumerate(
            scenarios, start=1
        )
    ]

    pending = [
        task for task in tasks
        if not (
            checkpoints
            / f"scenario_{task[0]:03d}.json"
        ).exists()
    ]

    print("=" * 72, flush=True)
    print("TNSE EXPERIMENT 19", flush=True)
    print("DYNAMIC-RISK FEATURE WEIGHT SENSITIVITY", flush=True)
    print("=" * 72, flush=True)

    print("Mode:", args.mode, flush=True)
    print("Scenarios:", len(scenarios), flush=True)
    print("Profiles:", list(PROFILES), flush=True)
    print("Workers:", workers, flush=True)
    print("Monte Carlo:", n_runs, flush=True)
    print("Pending:", len(pending), flush=True)

    started = time.perf_counter()

    with ProcessPoolExecutor(
        max_workers=workers
    ) as pool:
        futures = {
            pool.submit(run_task, task): task
            for task in pending
        }

        for future in as_completed(futures):
            task = futures[future]
            sid = task[0]

            try:
                result = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"DYNAMIC-RISK TASK FAILED: {task}"
                ) from exc

            checkpoint = (
                checkpoints
                / f"scenario_{sid:03d}.json"
            )

            save_json(checkpoint, result)

            print(
                f"[saved {sid:02d}/{len(tasks)}] "
                f"{task[1]} "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )

    all_summary = []
    all_details = []

    for task in tasks:
        sid = task[0]
        checkpoint = (
            checkpoints
            / f"scenario_{sid:03d}.json"
        )

        data = json.loads(
            checkpoint.read_text(
                encoding="utf-8"
            )
        )

        if data["scenario_id"] != sid:
            raise RuntimeError(
                "Checkpoint identity mismatch"
            )

        all_summary.extend(data["summary"])
        all_details.extend(data["details"])

    summary = pd.DataFrame(all_summary)
    details = pd.DataFrame(all_details)

    if len(summary) != len(tasks) * len(VARIANTS):
        raise RuntimeError(
            "Incomplete summary"
        )

    if len(details) != (
        len(tasks) * n_runs * len(VARIANTS)
    ):
        raise RuntimeError(
            "Incomplete per-run records"
        )

    if summary.duplicated(
        ["scenario_id", "variant"]
    ).any():
        raise RuntimeError(
            "Duplicate summary records"
        )

    if details.duplicated(
        ["scenario_id", "run_index", "variant"]
    ).any():
        raise RuntimeError(
            "Duplicate per-run records"
        )

    if args.mode == "pilot":
        check_pilot(all_details)

    summary_path = run_dir / "summary.csv"
    detail_path = run_dir / "per_run.csv"
    aggregate_path = run_dir / "aggregate.csv"

    summary.to_csv(
        summary_path, index=False
    )

    details.to_csv(
        detail_path, index=False
    )

    aggregate = (
        summary.groupby(
            "variant",
            as_index=False,
        )
        .agg(
            mean_EP=("EP", "mean"),
            mean_PAS=("PAS", "mean"),
            mean_PS=("PS", "mean"),
            mean_switch_rate=("switch_rate", "mean"),
            mean_EP_gain_pct=("EP_gain_pct", "mean"),
            mean_candidate_change=(
                "candidate_change_rate", "mean"
            ),
            mean_policy_change=(
                "policy_change_rate", "mean"
            ),
        )
        .sort_values("mean_EP")
    )

    aggregate.to_csv(
        aggregate_path, index=False
    )

    print("\n" + "=" * 72)
    print("DYNAMIC-RISK WEIGHT RESULTS")
    print("=" * 72)

    print(
        aggregate.to_string(index=False),
        flush=True,
    )

    print("\nSummary:", summary_path)
    print("Per-run:", detail_path)
    print("Aggregate:", aggregate_path)

    print(
        "Runtime:",
        round(time.perf_counter()-started, 2),
        "seconds",
        flush=True,
    )

    print(
        "EXPERIMENT 19 COMPLETED SUCCESSFULLY",
        flush=True,
    )


if __name__ == "__main__":
    main()
