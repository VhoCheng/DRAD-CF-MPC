from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import sys
import time
import inspect
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import run_experiments_core as core
import drad_mpc as mpc


# Same settings as the completed original pilot.
SCENARIOS = [
    dict(
        pattern="DDAC",
        attack_information_ratio=0.50,
        defense_budget=60,
        onset_time=150,
    ),
    dict(
        pattern="RC",
        attack_information_ratio=0.50,
        defense_budget=40,
        onset_time=120,
    ),
]

GAMMAS = [0.00, 0.10, 0.20]
KAPPAS = [0.00, 0.50, 1.00]

N_RUNS = 3
N_CF = 5
BASE_SEED = 42

BETA = 0.01
ALPHA = 0.15
DELTA = 1e-5

OUT = ROOT / "revision_experiments" / "results"
OUT.mkdir(parents=True, exist_ok=True)


def mode(values):
    return pd.Series(values).mode().iloc[0]


def run_seed(scenario, r):
    return (
        BASE_SEED
        + r
        + int(scenario["onset_time"]) * 1000
        + int(scenario["defense_budget"]) * 17
    )


def realized_once(graph, scenario, nodes, seed):
    """Run the original evaluation simulator once."""

    result = core.simulate_seid_once(
        Gc=graph,
        onset_time=scenario["onset_time"],
        infection_rate=BETA,
        activation_rate=ALPHA,
        detection_rate=DELTA,
        attack_information_ratio=(
            scenario["attack_information_ratio"]
        ),
        protected_nodes=nodes,
        strict_detection_failure=True,
        seed=seed,
    )

    pas = (
        0.0
        if result["attack_failed"]
        else core.evaluate_pas_proxy(
            active_infectious_nodes=(
                result["active_infectious_nodes"]
            ),
            pattern=scenario["pattern"],
            table_dir="outputs/tables",
        )
    )

    recovery = core.evaluate_resilience_metrics(
        pas=pas,
        defense_budget=scenario["defense_budget"],
        max_budget=100,
    )

    return {
        "PAS": float(pas),
        "failed": int(result["attack_failed"]),
        "RT": float(recovery["recovery_time"]),
        "RI": float(recovery["resilience_index"]),
        "RE": float(recovery["recovery_efficiency"]),
    }


def summarize(records):
    """Use the exact aggregation in run_single_setting."""

    pas = np.array([x["PAS"] for x in records])
    failed = np.array([x["failed"] for x in records])

    mean_pas = float(pas.mean())
    ps = 1.0 - float(failed.mean())

    return {
        "EP": mean_pas * ps,
        "PAS": mean_pas,
        "PS": ps,
        "RT": float(np.mean([x["RT"] for x in records])),
        "RI": float(np.mean([x["RI"] for x in records])),
        "RE": float(np.mean([x["RE"] for x in records])),
    }


def main():
    print("=" * 72, flush=True)
    print("TNSE REVISION - CACHED SAFE-GUARD REPLAY", flush=True)
    print("=" * 72, flush=True)

    original_evaluator = (
        mpc.evaluate_candidate_with_common_random_numbers
    )

    signature = inspect.signature(original_evaluator)

    cache = {}
    stats = {"hits": 0, "misses": 0}

    # Cache is only used for identical candidate evaluations.
    # The original evaluation function is preserved.
    def cached_evaluator(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments

        graph = a["Gc"]
        edge_signature = tuple(
            sorted(tuple(sorted(edge)) for edge in graph.edges())
        )

        key = (
            tuple(sorted(map(int, a["candidate_nodes"]))),
            tuple(map(int, a["common_seeds"])),
            edge_signature,
            str(a["pattern"]),
            float(a["attack_information_ratio"]),
            int(a["defense_budget"]),
            int(a["onset_time"]),
            float(a["infection_rate"]),
            float(a["activation_rate"]),
            float(a["detection_rate"]),
        )

        if key not in cache:
            stats["misses"] += 1
            cache[key] = original_evaluator(
                *args, **kwargs
            )
        else:
            stats["hits"] += 1

        return cache[key]

    mpc.evaluate_candidate_with_common_random_numbers = (
        cached_evaluator
    )

    summary_rows = []
    detail_rows = []

    start_all = time.perf_counter()

    try:
        for scenario_id, scenario in enumerate(
            SCENARIOS, start=1
        ):
            print(
                f"\nScenario {scenario_id}: {scenario}",
                flush=True,
            )

            # The current project always uses the original
            # fixed IEEE-118 / 118-node BA network.
            graph = core.build_comm_graph(
                n_nodes=118, seed=42
            )

            seeds = [
                run_seed(scenario, r)
                for r in range(N_RUNS)
            ]

            # CDP outcome is evaluated once per seed.
            baseline_records = []
            baseline_by_seed = {}

            for seed in seeds:
                cdp_nodes = core.select_protected_nodes(
                    strategy="CDP",
                    defense_budget=scenario["defense_budget"],
                    pattern=scenario["pattern"],
                    seed=seed,
                    table_dir="outputs/tables",
                )

                actual = realized_once(
                    graph, scenario, cdp_nodes, seed
                )

                baseline_records.append(actual)
                baseline_by_seed[seed] = actual

            baseline = summarize(baseline_records)

            # Reuse the same observed preview and actual outcome.
            preview_cache = {}
            actual_cache = {}

            for seed in seeds:
                preview_cache[seed] = core.get_preview_state(
                    Gc=graph,
                    onset_time=scenario["onset_time"],
                    attack_information_ratio=(
                        scenario["attack_information_ratio"]
                    ),
                    infection_rate=BETA,
                    activation_rate=ALPHA,
                    detection_rate=DELTA,
                    seed=seed,
                )

            for gamma in GAMMAS:
                for kappa in KAPPAS:

                    records = []
                    policies = []
                    guards = []
                    best_policies = []

                    for seed in seeds:
                        preview = preview_cache[seed]

                        selected_nodes, meta = (
                            mpc.select_drad_mpc_protected_nodes(
                                defense_budget=(
                                    scenario["defense_budget"]
                                ),
                                pattern=scenario["pattern"],
                                attack_information_ratio=(
                                    scenario[
                                        "attack_information_ratio"
                                    ]
                                ),
                                active_nodes=(
                                    preview["active_infectious_nodes"]
                                ),
                                exposed_nodes=(
                                    preview["exposed_nodes"]
                                ),
                                onset_time=(
                                    scenario["onset_time"]
                                ),
                                infection_rate=BETA,
                                activation_rate=ALPHA,
                                detection_rate=DELTA,
                                lookahead_runs=N_CF,
                                switch_margin=gamma,
                                confidence_k=kappa,
                                table_dir="outputs/tables",
                                seed=seed,
                            )
                        )

                        selected_policy = meta[
                            "selected_policy"
                        ]

                        guard = meta["safe_guard"]

                        best_policy = meta[
                            "best_policy_before_guard"
                        ]

                        key = (
                            seed,
                            tuple(sorted(selected_nodes)),
                        )

                        if key not in actual_cache:
                            actual_cache[key] = realized_once(
                                graph,
                                scenario,
                                selected_nodes,
                                seed,
                            )

                        actual = actual_cache[key]

                        records.append(actual)
                        policies.append(selected_policy)
                        guards.append(guard)
                        best_policies.append(best_policy)

                        cdp_actual = baseline_by_seed[seed]

                        cdp_score = meta.get(
                            "cdp_score", np.nan
                        )

                        best_score = meta.get(
                            "best_score_before_guard", np.nan
                        )

                        if (
                            np.isfinite(cdp_score)
                            and np.isfinite(best_score)
                        ):
                            predicted_delta_j = (
                                (cdp_score - best_score)
                                / (abs(cdp_score) + 1e-9)
                            )
                        else:
                            predicted_delta_j = np.nan

                        detail_rows.append({
                            "scenario_id": scenario_id,
                            **scenario,
                            "gamma": gamma,
                            "kappa": kappa,
                            "seed": seed,
                            "selected_policy": selected_policy,
                            "best_before_guard": best_policy,
                            "safe_guard": guard,
                            "actual_switch": int(
                                selected_policy != "CDP"
                            ),
                            "predicted_delta_J": predicted_delta_j,
                            "CDP_PAS": cdp_actual["PAS"],
                            "DRAD_PAS": actual["PAS"],
                            "paired_PAS_improvement": (
                                cdp_actual["PAS"] - actual["PAS"]
                            ),
                            "CDP_failed": cdp_actual["failed"],
                            "DRAD_failed": actual["failed"],
                        })

                    summary = summarize(records)

                    switch_rate = float(np.mean([
                        p != "CDP" for p in policies
                    ]))

                    negative_pas_rate = float(np.mean([
                        detail["paired_PAS_improvement"] < -1e-12
                        for detail in detail_rows[-N_RUNS:]
                    ]))

                    if baseline["EP"] > 1e-12:
                        ep_gain = (
                            100.0
                            * (baseline["EP"] - summary["EP"])
                            / baseline["EP"]
                        )
                    else:
                        ep_gain = float("nan")

                    summary_rows.append({
                        "scenario_id": scenario_id,
                        **scenario,
                        "gamma": gamma,
                        "kappa": kappa,
                        "n_runs": N_RUNS,
                        "n_cf": N_CF,
                        "CDP_EP": baseline["EP"],
                        "DRAD_EP": summary["EP"],
                        "CDP_PAS": baseline["PAS"],
                        "DRAD_PAS": summary["PAS"],
                        "EP_gain_pct": ep_gain,
                        "selected_policy_mode": mode(policies),
                        "safe_guard_mode": mode(guards),
                        "switch_rate": switch_rate,
                        "negative_paired_PAS_rate": negative_pas_rate,
                    })

                    print(
                        f"gamma={gamma:.2f}, "
                        f"kappa={kappa:.2f}, "
                        f"gain={ep_gain:+.2f}%, "
                        f"switch={switch_rate:.1%}",
                        flush=True,
                    )

    finally:
        mpc.evaluate_candidate_with_common_random_numbers = (
            original_evaluator
        )

    df = pd.DataFrame(summary_rows)
    details = pd.DataFrame(detail_rows)

    # Validate against the already completed uncached pilot.
    old_path = (
        OUT / "01_safeguard_pilot_20261018_023841.csv"
    )

    if old_path.exists():
        old = pd.read_csv(old_path)

        merged = df.merge(
            old[
                [
                    "scenario_id",
                    "gamma",
                    "kappa",
                    "CDP_EP",
                    "DRAD_EP",
                ]
            ],
            on=["scenario_id", "gamma", "kappa"],
            suffixes=("_new", "_old"),
            validate="one_to_one",
        )

        if len(merged) != len(df):
            raise RuntimeError(
                "REPLAY VALIDATION FAILED: row count differs."
            )

        for metric in ["CDP_EP", "DRAD_EP"]:
            if not np.allclose(
                merged[f"{metric}_new"],
                merged[f"{metric}_old"],
                rtol=0,
                atol=1e-10,
            ):
                difference = np.max(np.abs(
                    merged[f"{metric}_new"]
                    - merged[f"{metric}_old"]
                ))
                raise RuntimeError(
                    f"REPLAY VALIDATION FAILED: "
                    f"{metric}, max difference={difference}"
                )

        print(
            "\nREPLAY VALIDATION PASSED: "
            "all 18 EP results match the original pilot.",
            flush=True,
        )
    else:
        print(
            "\nWARNING: Original pilot CSV not found; "
            "automatic equivalence validation was skipped.",
            flush=True,
        )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    summary_path = (
        OUT / f"01b_safeguard_replay_summary_{timestamp}.csv"
    )

    details_path = (
        OUT / f"01b_safeguard_replay_per_run_{timestamp}.csv"
    )

    metadata_path = (
        OUT / f"01b_safeguard_replay_config_{timestamp}.json"
    )

    df.to_csv(summary_path, index=False)
    details.to_csv(details_path, index=False)

    metadata_path.write_text(
        json.dumps(
            {
                "experiment": "cached pilot replay",
                "status": "pilot_validation_only",
                "scenarios": SCENARIOS,
                "gamma": GAMMAS,
                "kappa": KAPPAS,
                "n_runs": N_RUNS,
                "n_cf": N_CF,
                "seed": BASE_SEED,
                "cache_hits": stats["hits"],
                "cache_misses": stats["misses"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("REPLAY EXPERIMENT SUMMARY")
    print("=" * 72)

    print(
        df[
            [
                "scenario_id",
                "gamma",
                "kappa",
                "EP_gain_pct",
                "switch_rate",
                "negative_paired_PAS_rate",
            ]
        ].to_string(index=False)
    )

    print("\nCache hits:", stats["hits"])
    print("Cache misses:", stats["misses"])
    print("Total runtime:", round(
        time.perf_counter() - start_all, 2
    ), "seconds")

    print("Summary:", summary_path)
    print("Per-run details:", details_path)
    print("Configuration:", metadata_path)
    print("REPLAY EXPERIMENT COMPLETED")


if __name__ == "__main__":
    main()
