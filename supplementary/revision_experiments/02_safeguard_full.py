from pathlib import Path
from datetime import datetime, timezone
from itertools import product
import argparse
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import run_experiments_core as core
import drad_mpc as mpc


PILOT_SCENARIOS = [
    ("DDAC", 0.50, 60, 150),
    ("RC", 0.50, 40, 120),
]

FULL_SCENARIOS = list(product(
    ["DDAC", "DDDC", "DCAC", "DCDC", "RC"],
    [0.25, 0.50, 1.00],
    [40, 80],
    [90, 210],
))

BETA = 0.01
ALPHA = 0.15
DELTA = 1e-5
BASE_SEED = 42
N_CF = 5


def evaluate_once(graph, scenario, nodes, seed):
    pattern, info, budget, onset = scenario

    sim = core.simulate_seid_once(
        Gc=graph,
        onset_time=onset,
        infection_rate=BETA,
        activation_rate=ALPHA,
        detection_rate=DELTA,
        attack_information_ratio=info,
        protected_nodes=nodes,
        strict_detection_failure=True,
        seed=seed,
    )

    pas = (
        0.0 if sim["attack_failed"]
        else core.evaluate_pas_proxy(
            active_infectious_nodes=sim["active_infectious_nodes"],
            pattern=pattern,
            table_dir="outputs/tables",
        )
    )

    recovery = core.evaluate_resilience_metrics(
        pas=pas,
        defense_budget=budget,
        max_budget=100,
    )

    return {
        "PAS": float(pas),
        "failed": int(sim["attack_failed"]),
        "RT": float(recovery["recovery_time"]),
        "RI": float(recovery["resilience_index"]),
        "RE": float(recovery["recovery_efficiency"]),
    }


def evaluate_seed(graph, scenario, run_index):
    pattern, info, budget, onset = scenario

    seed = (
        BASE_SEED
        + run_index
        + onset * 1000
        + budget * 17
    )

    cdp_nodes = core.select_protected_nodes(
        strategy="CDP",
        defense_budget=budget,
        pattern=pattern,
        seed=seed,
        table_dir="outputs/tables",
    )

    cdp_actual = evaluate_once(
        graph, scenario, cdp_nodes, seed
    )

    preview = core.get_preview_state(
        Gc=graph,
        onset_time=onset,
        attack_information_ratio=info,
        infection_rate=BETA,
        activation_rate=ALPHA,
        detection_rate=DELTA,
        seed=seed,
    )

    # Capture the original paired-improvement statistics.
    # No changes are made to the source implementation.
    original_pair = mpc.paired_improvement
    captured = []

    def capture_pair(candidate, baseline):
        result = original_pair(candidate, baseline)
        captured.append(result)
        return result

    mpc.paired_improvement = capture_pair

    try:
        nodes_at_zero, meta = (
            mpc.select_drad_mpc_protected_nodes(
                defense_budget=budget,
                pattern=pattern,
                attack_information_ratio=info,
                active_nodes=preview["active_infectious_nodes"],
                exposed_nodes=preview["exposed_nodes"],
                onset_time=onset,
                infection_rate=BETA,
                activation_rate=ALPHA,
                detection_rate=DELTA,
                lookahead_runs=N_CF,
                switch_margin=0.0,
                confidence_k=0.0,
                table_dir="outputs/tables",
                seed=seed,
            )
        )
    finally:
        mpc.paired_improvement = original_pair

    best = meta["best_policy_before_guard"]
    cdp_score = float(meta["cdp_score"])
    best_score = float(meta["best_score_before_guard"])

    delta_j = (
        (cdp_score - best_score)
        / (abs(cdp_score) + 1e-9)
    )

    if best == "CDP":
        if captured:
            raise RuntimeError(
                "Unexpected paired-improvement call for CDP."
            )
        mean_diff, stderr = 0.0, 0.0
        best_actual = None

    else:
        if len(captured) != 1:
            raise RuntimeError(
                "Unexpected paired-improvement call count."
            )

        mean_diff, stderr = map(float, captured[0])

        # At gamma=0, kappa=0, a policy is accepted
        # exactly when delta_j >= 0 and mean_diff > 0.
        can_switch_at_zero = (
            delta_j >= 0.0 and mean_diff > 0.0
        )

        if can_switch_at_zero:
            if meta["selected_policy"] != best:
                raise RuntimeError(
                    "Zero-threshold selection does not match."
                )

            best_actual = evaluate_once(
                graph, scenario, nodes_at_zero, seed
            )
        else:
            if meta["selected_policy"] != "CDP":
                raise RuntimeError(
                    "Unexpected zero-threshold selection."
                )
            best_actual = None

    return {
        "seed": seed,
        "cdp": cdp_actual,
        "best_actual": best_actual,
        "best": best,
        "delta_j": float(delta_j),
        "mean_diff": float(mean_diff),
        "stderr": float(stderr),
    }


def aggregate(records):
    mean_pas = float(np.mean([
        row["PAS"] for row in records
    ]))

    ps = 1.0 - float(np.mean([
        row["failed"] for row in records
    ]))

    # Preserve the EP definition used by the original code.
    return {
        "EP": mean_pas * ps,
        "PAS": mean_pas,
        "PS": ps,
        "RT": float(np.mean([x["RT"] for x in records])),
        "RI": float(np.mean([x["RI"] for x in records])),
        "RE": float(np.mean([x["RE"] for x in records])),
    }


def replay(scenario_id, scenario, seed_records,
           gammas, kappas):

    pattern, info, budget, onset = scenario

    cdp_summary = aggregate([
        x["cdp"] for x in seed_records
    ])

    summaries = []
    details = []

    for gamma, kappa in product(gammas, kappas):
        outcomes = []
        switches = []
        policies = []
        guards = []
        negative_pas = []

        for record in seed_records:
            best = record["best"]

            should_switch = (
                best != "CDP"
                and record["delta_j"] >= gamma
                and (
                    record["mean_diff"]
                    - kappa * record["stderr"]
                ) > 0.0
            )

            if should_switch:
                if record["best_actual"] is None:
                    raise RuntimeError(
                        "Missing alternative-policy outcome."
                    )

                selected = best
                actual = record["best_actual"]
                guard = "switched_with_counterfactual_gain"

            else:
                selected = "CDP"
                actual = record["cdp"]

                guard = (
                    "cdp_best"
                    if best == "CDP"
                    else "fallback_to_cdp"
                )

            outcomes.append(actual)
            switches.append(int(should_switch))
            policies.append(selected)
            guards.append(guard)

            diff = (
                record["cdp"]["PAS"] - actual["PAS"]
            )

            negative_pas.append(int(diff < -1e-12))

            details.append({
                "scenario_id": scenario_id,
                "pattern": pattern,
                "info": info,
                "budget": budget,
                "onset": onset,
                "gamma": gamma,
                "kappa": kappa,
                "seed": record["seed"],
                "best_before_guard": best,
                "selected_policy": selected,
                "safe_guard": guard,
                "switched": int(should_switch),
                "predicted_delta_J": record["delta_j"],
                "predicted_mean_PAS_gain": record["mean_diff"],
                "predicted_gain_stderr": record["stderr"],
                "CDP_PAS": record["cdp"]["PAS"],
                "DRAD_PAS": actual["PAS"],
                "paired_PAS_gain": diff,
                "CDP_failed": record["cdp"]["failed"],
                "DRAD_failed": actual["failed"],
            })

        drad_summary = aggregate(outcomes)

        cdp_ep = cdp_summary["EP"]
        drad_ep = drad_summary["EP"]

        gain_pct = (
            100.0 * (cdp_ep - drad_ep) / cdp_ep
            if cdp_ep > 1e-12
            else np.nan
        )

        summaries.append({
            "scenario_id": scenario_id,
            "pattern": pattern,
            "info": info,
            "budget": budget,
            "onset": onset,
            "gamma": gamma,
            "kappa": kappa,
            "CDP_EP": cdp_ep,
            "DRAD_EP": drad_ep,
            "CDP_PAS": cdp_summary["PAS"],
            "DRAD_PAS": drad_summary["PAS"],
            "CDP_RT": cdp_summary["RT"],
            "DRAD_RT": drad_summary["RT"],
            "CDP_RI": cdp_summary["RI"],
            "DRAD_RI": drad_summary["RI"],
            "CDP_RE": cdp_summary["RE"],
            "DRAD_RE": drad_summary["RE"],
            "EP_gain_pct": gain_pct,
            "switch_rate": float(np.mean(switches)),
            "negative_paired_PAS_rate": float(
                np.mean(negative_pas)
            ),
            "selected_policy_mode": (
                pd.Series(policies).mode().iloc[0]
            ),
            "safe_guard_mode": (
                pd.Series(guards).mode().iloc[0]
            ),
        })

    return summaries, details


def validate_pilot(summary):
    previous = (
        ROOT / "revision_experiments/results/"
        "01b_safeguard_replay_summary_20261018_030510.csv"
    )

    if not previous.exists():
        raise FileNotFoundError(
            "Previous validated pilot CSV not found."
        )

    old = pd.read_csv(previous)

    merged = summary.merge(
        old[
            [
                "scenario_id", "gamma", "kappa",
                "CDP_EP", "DRAD_EP", "switch_rate",
            ]
        ],
        on=["scenario_id", "gamma", "kappa"],
        suffixes=("_new", "_old"),
        validate="one_to_one",
    )

    if len(merged) != 18:
        raise RuntimeError(
            f"Expected 18 matches, got {len(merged)}"
        )

    for metric in ["CDP_EP", "DRAD_EP", "switch_rate"]:
        a = merged[f"{metric}_new"].to_numpy()
        b = merged[f"{metric}_old"].to_numpy()

        if not np.allclose(
            a, b, rtol=0, atol=1e-10
        ):
            raise RuntimeError(
                f"PILOT EQUIVALENCE FAILED: {metric}, "
                f"max difference={np.max(np.abs(a-b))}"
            )

    print(
        "\nPILOT EQUIVALENCE PASSED: "
        "18 EP and switching-rate results match.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["pilot", "full"],
        required=True,
    )
    args = parser.parse_args()

    if args.mode == "pilot":
        scenarios = PILOT_SCENARIOS
        gammas = [0.0, 0.10, 0.20]
        kappas = [0.0, 0.50, 1.0]
        n_runs = 3
    else:
        scenarios = FULL_SCENARIOS
        gammas = [0.0, 0.05, 0.10, 0.15, 0.20]
        kappas = [0.0, 0.25, 0.50, 0.75, 1.0]
        n_runs = 10

    output_dir = ROOT / "revision_experiments/results"
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    prefix = f"02_safeguard_{args.mode}_{stamp}"

    summary_path = output_dir / f"{prefix}_summary.csv"
    detail_path = output_dir / f"{prefix}_per_run.csv"
    config_path = output_dir / f"{prefix}_config.json"

    config = {
        "experiment": "Safe-Guard sensitivity",
        "mode": args.mode,
        "status": (
            "pilot_equivalence_test"
            if args.mode == "pilot"
            else "representative_scenario_sensitivity"
        ),
        "n_runs": n_runs,
        "n_cf": N_CF,
        "base_seed": BASE_SEED,
        "gammas": gammas,
        "kappas": kappas,
        "scenarios": scenarios,
        "malware": {
            "infection_rate": BETA,
            "activation_rate": ALPHA,
            "detection_rate": DELTA,
        },
        "code_sha256": {
            name: hashlib.sha256(
                (ROOT / name).read_bytes()
            ).hexdigest()
            for name in [
                "src/drad_mpc.py",
                "src/run_experiments_core.py",
                "src/seid_malware.py",
            ]
        },
    }

    config_path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("=" * 70, flush=True)
    print("TNSE - SAFE-GUARD SENSITIVITY", flush=True)
    print("Mode:", args.mode, flush=True)
    print("Scenarios:", len(scenarios), flush=True)
    print("Parameter combinations:",
          len(gammas) * len(kappas), flush=True)
    print("Monte Carlo:", n_runs, flush=True)
    print("Lookahead runs:", N_CF, flush=True)
    print("=" * 70, flush=True)

    start = time.perf_counter()
    all_summaries = []

    for i, scenario in enumerate(scenarios, 1):
        graph = core.build_comm_graph(
            n_nodes=118, seed=42
        )

        seeds = [
            evaluate_seed(graph, scenario, r)
            for r in range(n_runs)
        ]

        summaries, details = replay(
            i, scenario, seeds, gammas, kappas
        )

        scenario_summary = pd.DataFrame(summaries)
        scenario_detail = pd.DataFrame(details)

        # Save checkpoints after every scenario.
        scenario_summary.to_csv(
            summary_path,
            mode="a",
            index=False,
            header=(i == 1),
        )

        scenario_detail.to_csv(
            detail_path,
            mode="a",
            index=False,
            header=(i == 1),
        )

        all_summaries.extend(summaries)

        default = scenario_summary[
            (scenario_summary["gamma"] == 0.10)
            & (scenario_summary["kappa"] == 0.50)
        ].iloc[0]

        print(
            f"[{i:02d}/{len(scenarios)}] "
            f"{scenario} | "
            f"default_gain={default['EP_gain_pct']:+.2f}% | "
            f"default_switch={default['switch_rate']:.1%} | "
            f"elapsed={time.perf_counter()-start:.1f}s",
            flush=True,
        )

    full_summary = pd.DataFrame(all_summaries)

    if args.mode == "pilot":
        validate_pilot(full_summary)

    if args.mode == "full":
        by_params = (
            full_summary.groupby(
                ["gamma", "kappa"], as_index=False
            )
            .agg(
                mean_EP_gain_pct=("EP_gain_pct", "mean"),
                mean_switch_rate=("switch_rate", "mean"),
                mean_negative_PAS_rate=(
                    "negative_paired_PAS_rate", "mean"
                ),
                mean_DRAD_EP=("DRAD_EP", "mean"),
                mean_CDP_EP=("CDP_EP", "mean"),
            )
        )

        aggregate_path = (
            output_dir / f"{prefix}_parameter_aggregate.csv"
        )

        by_params.to_csv(
            aggregate_path, index=False
        )

        print("\nPARAMETER-LEVEL RESULTS", flush=True)
        print(
            by_params.to_string(index=False),
            flush=True,
        )
        print("\nAggregate:", aggregate_path)

    print("\nSummary:", summary_path)
    print("Per-run data:", detail_path)
    print("Configuration:", config_path)
    print("Total runtime:", round(
        time.perf_counter()-start, 2
    ), "seconds")
    print("EXPERIMENT COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
