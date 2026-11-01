from pathlib import Path
from datetime import datetime, timezone
from itertools import product
import argparse
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "revision_experiments/results"
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

import run_experiments_core as core
import drad_mpc as mpc

# Reuse the already validated Experiment 10 utilities.
helper_path = (
    ROOT / "revision_experiments"
    / "10_cf_optimization_pilot.py"
)

if not helper_path.exists():
    raise FileNotFoundError(helper_path)

spec = importlib.util.spec_from_file_location(
    "experiment10_helper", helper_path
)

helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)

PILOT_SCENARIOS = [
    ("DDAC", 0.50, 60, 150),
    ("RC", 0.50, 40, 120),
]

TEST_SCENARIOS = list(product(
    ["DDAC", "DDDC", "DCAC", "DCDC", "RC"],
    [0.25, 1.00],
    [20, 80],
    [150, 240],
))

VARIANTS = [
    "CDP",
    "Legacy-K5",
    "MoreCF-K20",
    "Confirm15-K05",
    "Confirm15-K10",
    "Confirm15-Robust",
]

GAMMA = 0.10

BETA = 0.01
ALPHA = 0.15
DELTA = 1e-5


def source_hash(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def confirm_candidates(
    graph,
    scenario,
    seed,
    candidates,
    metrics5,
):
    """
    Stage 1: screen with K=5.

    Stage 2: verify top three non-CDP candidates
    against CDP using fresh, disjoint CF seeds.
    """

    pattern, info, budget, onset = scenario

    ranking = sorted(
        [
            name
            for name in metrics5
            if name != "CDP"
        ],
        key=lambda name: metrics5[name]["objective"],
    )

    shortlist = ["CDP"] + ranking[:3]

    # Disjoint from the legacy CF seed sequence
    # and from the actual evaluation seed.
    confirmation_seeds = [
        int(seed)
        + 9_000_000
        + 10_007 * k
        + mpc.stable_int_hash(
            f"confirm-{pattern}-{onset}-{budget}",
            100_000,
        )
        for k in range(15)
    ]

    confirm_metrics = {}

    for name in shortlist:
        confirm_metrics[name] = (
            mpc.evaluate_candidate_with_common_random_numbers(
                candidate_nodes=candidates[name],
                common_seeds=confirmation_seeds,
                Gc=graph,
                pattern=pattern,
                attack_information_ratio=info,
                defense_budget=budget,
                onset_time=onset,
                infection_rate=BETA,
                activation_rate=ALPHA,
                detection_rate=DELTA,
            )
        )

    return shortlist, confirm_metrics


def select_confirmed(
    confirm_metrics,
    eta,
    kappa,
):
    scores = {}

    for name, metrics in confirm_metrics.items():
        samples = np.asarray(
            metrics["pas_samples"],
            dtype=float,
        )

        se = (
            float(samples.std(ddof=1) / np.sqrt(len(samples)))
            if len(samples) > 1
            else 0.0
        )

        scores[name] = (
            float(metrics["objective"])
            + eta * se
        )

    cdp_score = scores["CDP"]

    eligible = []

    for name in confirm_metrics:
        if name == "CDP":
            continue

        improvement = (
            (cdp_score - scores[name])
            / (abs(cdp_score) + 1e-9)
        )

        mean_diff, stderr = mpc.paired_improvement(
            confirm_metrics[name],
            confirm_metrics["CDP"],
        )

        if (
            improvement >= GAMMA
            and mean_diff - kappa * stderr > 0.0
        ):
            eligible.append(name)

    if not eligible:
        return "CDP"

    return min(
        eligible,
        key=lambda name: scores[name],
    )


def validate_original_pilot(summary):
    files = sorted(
        OUT.glob(
            "10_cf_optimization_pilot_*_summary.csv"
        ),
        key=lambda p: p.stat().st_mtime,
    )

    if not files:
        raise FileNotFoundError(
            "Experiment 10 reference CSV not found."
        )

    previous = pd.read_csv(files[-1])

    for scenario_id in [1, 2]:
        for variant in [
            "CDP",
            "Legacy-K5",
            "MoreCF-K20",
        ]:
            current = summary[
                (summary["scenario_id"] == scenario_id)
                & (summary["variant"] == variant)
            ]

            original = previous[
                (previous["scenario_id"] == scenario_id)
                & (previous["variant"] == variant)
            ]

            if len(current) != 1 or len(original) != 1:
                raise RuntimeError(
                    "Reference row missing."
                )

            for metric in [
                "EP", "PAS", "PS",
                "RT", "RI", "RE",
                "switch_rate",
            ]:
                a = float(current.iloc[0][metric])
                b = float(original.iloc[0][metric])

                if not np.isclose(
                    a, b,
                    atol=1e-10,
                    rtol=0,
                ):
                    raise RuntimeError(
                        "PILOT REFERENCE FAILED: "
                        f"{scenario_id}, {variant}, "
                        f"{metric}: new={a}, old={b}"
                    )

    print(
        "\nPILOT REFERENCE PASSED: "
        "CDP, Legacy-K5 and MoreCF-K20 reproduced.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["pilot", "test"],
        required=True,
    )

    args = parser.parse_args()

    helper.check_original()

    if args.mode == "pilot":
        scenarios = PILOT_SCENARIOS
        n_runs = 3
        base_seed = 42
    else:
        scenarios = TEST_SCENARIOS
        n_runs = 10
        base_seed = 20261103

    print("=" * 72, flush=True)
    print("TNSE EXPERIMENT 12 - FROZEN HOLDOUT", flush=True)
    print("FROZEN COUNTERFACTUAL CONFIRMATION TEST", flush=True)
    print("=" * 72, flush=True)
    print("Mode:", args.mode, flush=True)
    print("Scenarios:", len(scenarios), flush=True)
    print("Monte Carlo:", n_runs, flush=True)
    print("Base seed:", base_seed, flush=True)
    print("Variants:", VARIANTS, flush=True)

    graph = core.build_comm_graph(
        n_nodes=118,
        seed=42,
    )

    summary_rows = []
    detail_rows = []

    start = time.perf_counter()

    for scenario_id, scenario in enumerate(
        scenarios, start=1
    ):
        pattern, info, budget, onset = scenario

        records = {
            name: [] for name in VARIANTS
        }

        selected_policies = {
            name: [] for name in VARIANTS
        }

        for r in range(n_runs):
            seed = (
                base_seed + r
                + onset * 1000
                + budget * 17
            )

            candidates, metrics5, metrics20 = (
                helper.predict_candidates(
                    graph,
                    scenario,
                    seed,
                )
            )

            legacy, _ = helper.choose_policy(
                metrics5, 0.0, 0.5
            )

            more_cf, _ = helper.choose_policy(
                metrics20, 0.0, 0.5
            )

            shortlist, confirmed = confirm_candidates(
                graph,
                scenario,
                seed,
                candidates,
                metrics5,
            )

            decisions = {
                "CDP": "CDP",
                "Legacy-K5": legacy,
                "MoreCF-K20": more_cf,
                "Confirm15-K05": select_confirmed(
                    confirmed, eta=0.0, kappa=0.5
                ),
                "Confirm15-K10": select_confirmed(
                    confirmed, eta=0.0, kappa=1.0
                ),
                "Confirm15-Robust": select_confirmed(
                    confirmed, eta=0.5, kappa=1.0
                ),
            }

            actual_cache = {}

            for variant, policy in decisions.items():
                nodes = tuple(
                    sorted(candidates[policy])
                )

                if nodes not in actual_cache:
                    actual_cache[nodes] = (
                        helper.actual_outcome(
                            graph,
                            scenario,
                            list(nodes),
                            seed,
                        )
                    )

                outcome = actual_cache[nodes]

                records[variant].append(outcome)
                selected_policies[variant].append(policy)

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
                    "shortlist": "|".join(shortlist),
                    "PAS": outcome["PAS"],
                    "failed": outcome["failed"],
                    "RT": outcome["RT"],
                    "RI": outcome["RI"],
                    "RE": outcome["RE"],
                })

        baseline = helper.aggregate(
            records["CDP"]
        )

        for variant in VARIANTS:
            metrics = helper.aggregate(
                records[variant]
            )

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
                "n_runs": n_runs,
                "EP": metrics["EP"],
                "PAS": metrics["PAS"],
                "PS": metrics["PS"],
                "RT": metrics["RT"],
                "RI": metrics["RI"],
                "RE": metrics["RE"],
                "EP_gain_pct": gain,
                "switch_rate": float(np.mean([
                    p != "CDP"
                    for p in selected_policies[variant]
                ])),
            })

        print(
            f"[{scenario_id}/{len(scenarios)}] "
            f"{scenario} "
            f"elapsed={time.perf_counter()-start:.1f}s",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)

    if args.mode == "pilot":
        validate_original_pilot(summary)

    details = pd.DataFrame(detail_rows)

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S_%f"
    )

    prefix = (
        f"12_holdout_{args.mode}_{timestamp}"
    )

    summary_path = OUT / f"{prefix}_summary.csv"
    detail_path = OUT / f"{prefix}_per_run.csv"
    config_path = OUT / f"{prefix}_config.json"

    summary.to_csv(
        summary_path, index=False
    )

    details.to_csv(
        detail_path, index=False
    )

    config_path.write_text(
        json.dumps({
            "experiment": (
                "Two-stage counterfactual confirmation"
            ),
            "status": args.mode,
            "scenarios": scenarios,
            "n_runs": n_runs,
            "base_seed": base_seed,
            "first_stage_rollouts": 5,
            "shortlist_non_cdp": 3,
            "second_stage_rollouts": 15,
            "second_stage_seed_offset": 9_000_000,
            "gamma": GAMMA,
            "variants": VARIANTS,
            "source_hashes": {
                name: source_hash(SRC / name)
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
            mean_gain_pct=("EP_gain_pct", "mean"),
            mean_switch_rate=("switch_rate", "mean"),
        )
        .sort_values("mean_EP")
    )

    print("\n" + "=" * 72)
    print("EXPERIMENT 11 RESULTS")
    print("=" * 72)

    print(
        aggregate_df.to_string(index=False),
        flush=True,
    )

    print("\nSummary:", summary_path)
    print("Per-run details:", detail_path)
    print("Configuration:", config_path)

    print(
        "Total runtime:",
        round(time.perf_counter()-start, 2),
        "seconds",
    )

    print("EXPERIMENT 11 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
