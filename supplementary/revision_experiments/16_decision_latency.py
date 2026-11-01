from pathlib import Path
from itertools import product
from datetime import datetime, timezone
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUT = ROOT / "revision_experiments/results"

os.chdir(ROOT)
sys.path.insert(0, str(SRC))

import run_experiments_core as core
import drad_mpc as mpc

HELPER_PATH = (
    ROOT / "revision_experiments"
    / "14_parallel_replication.py"
)

spec = importlib.util.spec_from_file_location(
    "validated_experiment14",
    HELPER_PATH,
)

e14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e14)

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

METHODS = [
    "CDP",
    "Legacy-K5",
    "Confirm15-Robust",
]


def measure_once(graph, scenario, seed):
    pattern, info, budget, onset = scenario

    # Baseline CDP decision time.
    start = time.perf_counter()

    cdp_nodes = mpc.select_protected_nodes(
        strategy="CDP",
        defense_budget=budget,
        pattern=pattern,
        seed=seed,
        table_dir="outputs/tables",
    )

    cdp_seconds = time.perf_counter() - start

    # Original K5: all candidates and original guard.
    start = time.perf_counter()

    candidates, metrics5 = e14.create_metrics5(
        graph,
        scenario,
        seed,
    )

    legacy, _ = e14.e11.helper.choose_policy(
        metrics5,
        eta=0.0,
        kappa=0.5,
    )

    legacy_seconds = time.perf_counter() - start

    # Additional confirmation phase.
    start = time.perf_counter()

    shortlist, confirmed = (
        e14.e11.confirm_candidates(
            graph,
            scenario,
            seed,
            candidates,
            metrics5,
        )
    )

    robust = e14.e11.select_confirmed(
        confirmed,
        eta=0.5,
        kappa=1.0,
    )

    stage2_seconds = time.perf_counter() - start

    if sorted(cdp_nodes) != sorted(
        candidates["CDP"]
    ):
        raise RuntimeError(
            "CDP candidate node mismatch."
        )

    return {
        "CDP": "CDP",
        "Legacy-K5": legacy,
        "Confirm15-Robust": robust,
        "CDP_seconds": float(cdp_seconds),
        "Legacy_seconds": float(legacy_seconds),
        "Stage2_seconds": float(stage2_seconds),
        "Robust_seconds": float(
            legacy_seconds + stage2_seconds
        ),
        "candidate_count": len(candidates),
        "confirmation_count": len(shortlist),
    }


def validate_pilot(df):
    ref_path = (
        OUT / "14_replication_pilot" / "per_run.csv"
    )

    if not ref_path.exists():
        raise RuntimeError(
            "Experiment 14 pilot reference not found."
        )

    ref = pd.read_csv(ref_path)

    comparisons = 0

    for row in df.itertuples():
        for method in METHODS:
            selected = getattr(
                row,
                method.replace("-", "_"),
            ) if False else row._asdict().get(method)

            # Pandas itertuples renames invalid identifiers,
            # so use the DataFrame directly below instead.

    for _, row in df.iterrows():
        for method in METHODS:
            previous = ref[
                (ref["scenario_id"] == row["scenario_id"])
                & (ref["run_index"] == row["run_index"])
                & (ref["variant"] == method)
            ]

            if len(previous) != 1:
                raise RuntimeError(
                    "Missing historical pilot decision."
                )

            expected = previous.iloc[0][
                "selected_policy"
            ]

            observed = row[method]

            if observed != expected:
                raise RuntimeError(
                    "DECISION EQUIVALENCE FAILED: "
                    f"scenario={row['scenario_id']} "
                    f"run={row['run_index']} "
                    f"method={method} "
                    f"new={observed}, old={expected}"
                )

            comparisons += 1

    if comparisons != 18:
        raise RuntimeError(
            "Expected 18 decision equivalence checks."
        )

    print(
        "DECISION EQUIVALENCE PASSED: "
        "18/18 historical decisions match.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["pilot", "full"],
        required=True,
    )

    args = parser.parse_args()

    e14.e11.helper.check_original()

    if args.mode == "pilot":
        scenarios = PILOT
        n_runs = 3
        base_seed = 42
    else:
        scenarios = FULL
        n_runs = 4
        base_seed = 20261130

    graph = core.build_comm_graph(
        n_nodes=118, seed=42
    )

    print("=" * 72, flush=True)
    print("TNSE EXPERIMENT 16", flush=True)
    print("118-NODE SINGLE-DECISION LATENCY", flush=True)
    print("=" * 72, flush=True)
    print("Mode:", args.mode, flush=True)
    print("Scenarios:", len(scenarios), flush=True)
    print("Repetitions per scenario:", n_runs, flush=True)
    print("Parallel workers: 1", flush=True)

    records = []
    start_all = time.perf_counter()

    for sid, scenario in enumerate(scenarios, 1):
        pattern, info, budget, onset = scenario

        for r in range(n_runs):
            seed = (
                base_seed
                + r
                + onset * 1000
                + budget * 17
            )

            result = measure_once(
                graph, scenario, seed
            )

            records.append({
                "scenario_id": sid,
                "pattern": pattern,
                "info": info,
                "budget": budget,
                "onset": onset,
                "run_index": r,
                "seed": seed,
                **result,
            })

        print(
            f"[{sid}/{len(scenarios)}] "
            f"{scenario} "
            f"elapsed={time.perf_counter()-start_all:.1f}s",
            flush=True,
        )

    df = pd.DataFrame(records)

    if args.mode == "pilot":
        validate_pilot(df)

    rows = []

    for name, column in [
        ("CDP", "CDP_seconds"),
        ("Legacy-K5", "Legacy_seconds"),
        ("Confirm15-Robust", "Robust_seconds"),
        ("Confirmation_Only", "Stage2_seconds"),
    ]:
        x = df[column].to_numpy(dtype=float)

        rows.append({
            "method": name,
            "runs": len(x),
            "mean_seconds": float(x.mean()),
            "median_seconds": float(np.median(x)),
            "p95_seconds": float(np.quantile(x, 0.95)),
            "max_seconds": float(x.max()),
            "min_seconds": float(x.min()),
            "std_seconds": float(x.std(ddof=1)),
        })

    aggregate = pd.DataFrame(rows)

    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    prefix = f"16_latency_{args.mode}_{stamp}"

    raw_path = OUT / f"{prefix}_per_decision.csv"
    agg_path = OUT / f"{prefix}_aggregate.csv"
    config_path = OUT / f"{prefix}_config.json"

    df.to_csv(raw_path, index=False)
    aggregate.to_csv(agg_path, index=False)

    config_path.write_text(
        json.dumps({
            "experiment": "118-node decision latency",
            "mode": args.mode,
            "scenarios": scenarios,
            "n_runs": n_runs,
            "base_seed": base_seed,
            "parallel_workers": 1,
            "decision_scope": (
                "CDP node selection; full K5 candidate "
                "generation and counterfactual evaluation; "
                "additional independent K15 confirmation."
            ),
            "not_included": (
                "Actual post-decision SEID trajectory "
                "simulation or external communication delays."
            ),
            "helper_sha256": {
                filename: e14.digest(
                    ROOT / "revision_experiments" / filename
                )
                for filename in [
                    "10_cf_optimization_pilot.py",
                    "11_crossfit_confirmation.py",
                    "14_parallel_replication.py",
                ]
            },
        }, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("DECISION LATENCY RESULTS")
    print("=" * 72)

    print(aggregate.to_string(index=False))

    print("\nAverage candidate count:",
          df["candidate_count"].mean())
    print("Average confirmation count:",
          df["confirmation_count"].mean())

    print("\nPer-decision CSV:", raw_path)
    print("Aggregate CSV:", agg_path)
    print("Config:", config_path)

    print(
        "Total runtime:",
        round(time.perf_counter()-start_all, 2),
        "seconds",
    )

    print("EXPERIMENT 16 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
