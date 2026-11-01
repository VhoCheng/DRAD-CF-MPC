from pathlib import Path
from datetime import datetime, timezone
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

OUT = ROOT / "revision_experiments/results"
OUT.mkdir(parents=True, exist_ok=True)

SCENARIOS = [
    ("DDAC", 0.50, 60, 150),
    ("RC", 0.50, 40, 120),
]

VARIANTS = {
    "Legacy-K5": (5, 0.0, 0.5),
    "MoreCF-K20": (20, 0.0, 0.5),
    "Guard-K20": (20, 0.0, 1.0),
    "Robust05-K20": (20, 0.5, 1.0),
    "Robust10-K20": (20, 1.0, 1.0),
}

BETA = 0.01
ALPHA = 0.15
DELTA = 1e-5

GAMMA = 0.10
BASE_SEED = 42
N_RUNS = 3


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_original():
    for name in [
        "drad_mpc.py",
        "run_experiments_core.py",
        "seid_malware.py",
    ]:
        current = ROOT / "src" / name
        original = ROOT.parent / "DRAD-CF-MPC/src" / name

        if not current.exists() or not original.exists():
            raise RuntimeError(f"Missing source: {name}")

        if sha256(current) != sha256(original):
            raise RuntimeError(
                f"Original source changed: {name}"
            )

    print("ORIGINAL SOURCE CHECK PASSED", flush=True)


def actual_outcome(graph, scenario, nodes, seed):
    pattern, info, budget, onset = scenario

    result = core.simulate_seid_once(
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
        0.0
        if result["attack_failed"]
        else core.evaluate_pas_proxy(
            active_infectious_nodes=(
                result["active_infectious_nodes"]
            ),
            pattern=pattern,
            table_dir="outputs/tables",
        )
    )

    resilience = core.evaluate_resilience_metrics(
        pas=pas,
        defense_budget=budget,
        max_budget=100,
    )

    return {
        "PAS": float(pas),
        "failed": int(result["attack_failed"]),
        "RT": float(resilience["recovery_time"]),
        "RI": float(resilience["resilience_index"]),
        "RE": float(resilience["recovery_efficiency"]),
    }


def choose_policy(metrics, eta, kappa):
    scores = {}

    for name, item in metrics.items():
        samples = np.asarray(
            item["pas_samples"], dtype=float
        )

        if len(samples) > 1:
            standard_error = float(
                samples.std(ddof=1)
                / np.sqrt(len(samples))
            )
        else:
            standard_error = 0.0

        scores[name] = (
            float(item["objective"])
            + float(eta) * standard_error
        )

    best = min(scores, key=scores.get)

    if best == "CDP":
        return "CDP", "cdp_best"

    cdp_score = float(scores["CDP"])
    best_score = float(scores[best])

    relative_gain = (
        (cdp_score - best_score)
        / (abs(cdp_score) + 1e-9)
    )

    mean_diff, stderr = mpc.paired_improvement(
        metrics[best],
        metrics["CDP"],
    )

    conservative_gain = (
        mean_diff - float(kappa) * stderr
    )

    if (
        relative_gain >= GAMMA
        and conservative_gain > 0.0
    ):
        return best, "switched_with_counterfactual_gain"

    return "CDP", "fallback_to_cdp"


def predict_candidates(graph, scenario, seed):
    pattern, info, budget, onset = scenario

    preview = core.get_preview_state(
        Gc=graph,
        onset_time=onset,
        attack_information_ratio=info,
        infection_rate=BETA,
        activation_rate=ALPHA,
        detection_rate=DELTA,
        seed=seed,
    )

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
        raise RuntimeError("CDP candidate missing.")

    # Exactly the same counterfactual seed construction
    # as the original DRAD-CF-MPC implementation.
    common_seeds = [
        seed
        + 777_000
        + 10_007 * k
        + mpc.stable_int_hash(
            f"{pattern}-{onset}-{budget}",
            100_000,
        )
        for k in range(20)
    ]

    metrics5 = {}
    metrics20 = {}

    for name, nodes in candidates.items():
        args = dict(
            candidate_nodes=nodes,
            Gc=graph,
            pattern=pattern,
            attack_information_ratio=info,
            defense_budget=budget,
            onset_time=onset,
            infection_rate=BETA,
            activation_rate=ALPHA,
            detection_rate=DELTA,
        )

        metrics5[name] = (
            mpc.evaluate_candidate_with_common_random_numbers(
                common_seeds=common_seeds[:5],
                **args,
            )
        )

        metrics20[name] = (
            mpc.evaluate_candidate_with_common_random_numbers(
                common_seeds=common_seeds,
                **args,
            )
        )

    return candidates, metrics5, metrics20


def aggregate(records):
    pas = float(np.mean([
        x["PAS"] for x in records
    ]))

    ps = 1.0 - float(np.mean([
        x["failed"] for x in records
    ]))

    return {
        "EP": pas * ps,
        "PAS": pas,
        "PS": ps,
        "RT": float(np.mean([x["RT"] for x in records])),
        "RI": float(np.mean([x["RI"] for x in records])),
        "RE": float(np.mean([x["RE"] for x in records])),
    }


def validate_reference(summary):
    files = sorted(
        OUT.glob("03_ablation_pilot_*_summary.csv"),
        key=lambda p: p.stat().st_mtime,
    )

    if not files:
        raise FileNotFoundError(
            "Original ablation pilot is missing."
        )

    reference = pd.read_csv(files[-1])

    checked = 0

    for scenario_id in [1, 2]:
        for variant, reference_variant in [
            ("CDP", "CDP"),
            ("Legacy-K5", "Full"),
        ]:
            actual = summary[
                (summary["scenario_id"] == scenario_id)
                & (summary["variant"] == variant)
            ]

            expected = reference[
                (reference["scenario_id"] == scenario_id)
                & (reference["variant"] == reference_variant)
            ]

            if len(actual) != 1 or len(expected) != 1:
                raise RuntimeError(
                    "Reference row missing."
                )

            actual = actual.iloc[0]
            expected = expected.iloc[0]

            for metric in ["EP", "PAS", "PS", "RT", "RI", "RE"]:
                a = float(actual[metric])
                b = float(expected[metric])

                if not np.isclose(
                    a, b, rtol=0, atol=1e-10
                ):
                    raise RuntimeError(
                        "LEGACY EQUIVALENCE FAILED: "
                        f"scenario={scenario_id}, "
                        f"variant={variant}, "
                        f"metric={metric}, "
                        f"new={a}, old={b}"
                    )

            checked += 1

    if checked != 4:
        raise RuntimeError("Incomplete validation.")

    print(
        "\nLEGACY EQUIVALENCE PASSED",
        flush=True,
    )


def main():
    print("=" * 72, flush=True)
    print("TNSE EXPERIMENT 10", flush=True)
    print("ROBUST COUNTERFACTUAL OPTIMIZATION PILOT", flush=True)
    print("=" * 72, flush=True)

    check_original()

    print("Scenarios:", len(SCENARIOS), flush=True)
    print("Monte Carlo:", N_RUNS, flush=True)
    print("Variants:", list(VARIANTS), flush=True)

    start = time.perf_counter()

    graph = core.build_comm_graph(
        n_nodes=118, seed=42
    )

    summary_rows = []
    detail_rows = []

    for scenario_id, scenario in enumerate(
        SCENARIOS, start=1
    ):
        pattern, info, budget, onset = scenario

        outcomes = {
            name: []
            for name in ["CDP", *VARIANTS]
        }

        selections = {
            name: []
            for name in ["CDP", *VARIANTS]
        }

        for r in range(N_RUNS):
            seed = (
                BASE_SEED + r
                + onset * 1000
                + budget * 17
            )

            candidates, m5, m20 = predict_candidates(
                graph, scenario, seed
            )

            policy_by_variant = {"CDP": "CDP"}

            for variant, params in VARIANTS.items():
                k, eta, kappa = params

                metrics = m5 if k == 5 else m20

                selected, guard = choose_policy(
                    metrics, eta, kappa
                )

                policy_by_variant[variant] = selected

            # Cache the actual outcome of identical
            # protected-node sets under the same seed.
            actual_cache = {}

            for variant, policy in policy_by_variant.items():
                nodes = tuple(
                    sorted(candidates[policy])
                )

                if nodes not in actual_cache:
                    actual_cache[nodes] = actual_outcome(
                        graph,
                        scenario,
                        list(nodes),
                        seed,
                    )

                actual = actual_cache[nodes]

                outcomes[variant].append(actual)
                selections[variant].append(policy)

                detail_rows.append({
                    "scenario_id": scenario_id,
                    "pattern": pattern,
                    "info": info,
                    "budget": budget,
                    "onset": onset,
                    "run_index": r,
                    "seed": seed,
                    "variant": variant,
                    "selected_policy": policy,
                    "PAS": actual["PAS"],
                    "failed": actual["failed"],
                    "RT": actual["RT"],
                    "RI": actual["RI"],
                    "RE": actual["RE"],
                })

        baseline = aggregate(outcomes["CDP"])

        for variant in ["CDP", *VARIANTS]:
            metrics = aggregate(outcomes[variant])

            gain = (
                100.0
                * (baseline["EP"] - metrics["EP"])
                / baseline["EP"]
                if baseline["EP"] > 1e-12
                else float("nan")
            )

            summary_rows.append({
                "scenario_id": scenario_id,
                "pattern": pattern,
                "info": info,
                "budget": budget,
                "onset": onset,
                "variant": variant,
                "EP": metrics["EP"],
                "PAS": metrics["PAS"],
                "PS": metrics["PS"],
                "RT": metrics["RT"],
                "RI": metrics["RI"],
                "RE": metrics["RE"],
                "EP_gain_pct": gain,
                "switch_rate": float(np.mean([
                    x != "CDP"
                    for x in selections[variant]
                ])),
            })

        print(
            f"Completed scenario {scenario_id}/{len(SCENARIOS)}",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    details = pd.DataFrame(detail_rows)

    validate_reference(summary)

    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S_%f")

    prefix = f"10_cf_optimization_pilot_{stamp}"

    summary_path = OUT / f"{prefix}_summary.csv"
    detail_path = OUT / f"{prefix}_per_run.csv"
    config_path = OUT / f"{prefix}_config.json"

    summary.to_csv(summary_path, index=False)
    details.to_csv(detail_path, index=False)

    config_path.write_text(
        json.dumps({
            "experiment": "Robust CF optimization pilot",
            "status": "pilot_not_final_performance",
            "scenarios": SCENARIOS,
            "variants": VARIANTS,
            "n_runs": N_RUNS,
            "gamma": GAMMA,
            "seed": BASE_SEED,
            "source_hashes": {
                name: sha256(ROOT / "src" / name)
                for name in [
                    "drad_mpc.py",
                    "run_experiments_core.py",
                    "seid_malware.py",
                ]
            },
        }, indent=2),
        encoding="utf-8",
    )

    aggregate_df = (
        summary.groupby("variant", as_index=False)
        .agg(
            mean_EP=("EP", "mean"),
            mean_PAS=("PAS", "mean"),
            mean_EP_gain_pct=("EP_gain_pct", "mean"),
            mean_switch_rate=("switch_rate", "mean"),
        )
        .sort_values("mean_EP")
    )

    print("\n" + "=" * 72)
    print("ROBUST COUNTERFACTUAL PILOT RESULTS")
    print("=" * 72)

    print(
        aggregate_df.to_string(index=False),
        flush=True,
    )

    print("\nSummary:", summary_path)
    print("Per-run:", detail_path)
    print("Config:", config_path)

    print(
        "Runtime:",
        round(time.perf_counter()-start, 2),
        "seconds"
    )

    print("EXPERIMENT 10 PILOT COMPLETED")


if __name__ == "__main__":
    main()
