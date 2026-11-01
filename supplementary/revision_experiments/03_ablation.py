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

OUT = ROOT / "revision_experiments" / "results"
OUT.mkdir(parents=True, exist_ok=True)

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

VARIANTS = [
    "CDP",
    "Full",
    "NoSafeGuard",
    "NoDynamicRisk",
    "RiskHeuristicNoCF",
]

GAMMA = 0.10
KAPPA = 0.50

N_CF = 5
BASE_SEED = 42

BETA = 0.01
ALPHA = 0.15
DELTA = 1e-5


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_original_implementation():
    names = [
        "drad_mpc.py",
        "run_experiments_core.py",
        "seid_malware.py",
    ]

    for name in names:
        original = ROOT.parent / "DRAD-CF-MPC" / "src" / name
        current = ROOT / "src" / name

        if not original.exists() or not current.exists():
            raise RuntimeError(
                f"Missing source file: {name}"
            )

        if file_hash(original) != file_hash(current):
            raise RuntimeError(
                f"Source differs from original: {name}. "
                "Do not continue until reviewed."
            )

    print("ORIGINAL SOURCE CHECK PASSED", flush=True)


def simulate_actual(graph, scenario, nodes, seed):
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

    resilience = core.evaluate_resilience_metrics(
        pas=pas,
        defense_budget=budget,
        max_budget=100,
    )

    return {
        "PAS": float(pas),
        "failed": int(sim["attack_failed"]),
        "RT": float(resilience["recovery_time"]),
        "RI": float(resilience["resilience_index"]),
        "RE": float(resilience["recovery_efficiency"]),
    }


def collect_full_decision(graph, scenario, seed):
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

    captured_candidates = {}
    captured_metrics = {}

    original_generator = mpc.generate_candidate_policies
    original_evaluator = (
        mpc.evaluate_candidate_with_common_random_numbers
    )

    def capture_generator(*args, **kwargs):
        candidates = original_generator(*args, **kwargs)
        captured_candidates.update(candidates)
        return candidates

    def capture_evaluator(*args, **kwargs):
        if "candidate_nodes" in kwargs:
            nodes = kwargs["candidate_nodes"]
        else:
            nodes = args[0]

        result = original_evaluator(*args, **kwargs)

        key = tuple(sorted(int(x) for x in nodes))
        captured_metrics[key] = result

        return result

    mpc.generate_candidate_policies = capture_generator
    mpc.evaluate_candidate_with_common_random_numbers = (
        capture_evaluator
    )

    try:
        full_nodes, full_meta = (
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
                switch_margin=GAMMA,
                confidence_k=KAPPA,
                table_dir="outputs/tables",
                seed=seed,
            )
        )
    finally:
        mpc.generate_candidate_policies = original_generator
        mpc.evaluate_candidate_with_common_random_numbers = (
            original_evaluator
        )

    candidates = captured_candidates
    scores = full_meta["candidate_scores"]

    if "CDP" not in candidates:
        raise RuntimeError("CDP candidate missing.")

    full_policy = full_meta["selected_policy"]

    if full_policy not in candidates:
        raise RuntimeError("Full selected policy missing.")

    if sorted(full_nodes) != sorted(candidates[full_policy]):
        raise RuntimeError(
            "Full policy and candidate nodes do not match."
        )

    # Ablation 1: remove Safe-Guard.
    # Always execute the candidate with minimum CF objective.
    best_all = full_meta["best_policy_before_guard"]
    no_guard_nodes = candidates[best_all]

    # Ablation 2: remove dynamic-risk-based candidates.
    # Preserve all other candidates and the original Safe-Guard.
    excluded = {"DRAD_SCORE", "RANK_ENSEMBLE"}

    reduced_scores = {
        name: float(scores[name])
        for name in candidates
        if name not in excluded
    }

    if "CDP" not in reduced_scores:
        raise RuntimeError("CDP missing after risk ablation.")

    best_reduced = min(
        reduced_scores, key=reduced_scores.get
    )

    if best_reduced == "CDP":
        reduced_policy = "CDP"
    else:
        cdp_nodes_key = tuple(sorted(candidates["CDP"]))
        best_nodes_key = tuple(sorted(candidates[best_reduced]))

        if (
            cdp_nodes_key not in captured_metrics
            or best_nodes_key not in captured_metrics
        ):
            raise RuntimeError(
                "Missing captured counterfactual samples."
            )

        mean_diff, stderr = mpc.paired_improvement(
            captured_metrics[best_nodes_key],
            captured_metrics[cdp_nodes_key],
        )

        cdp_score = reduced_scores["CDP"]
        best_score = reduced_scores[best_reduced]

        objective_gain = (
            (cdp_score - best_score)
            / (abs(cdp_score) + 1e-9)
        )

        allowed = (
            objective_gain >= GAMMA
            and mean_diff - KAPPA * stderr > 0.0
        )

        reduced_policy = (
            best_reduced if allowed else "CDP"
        )

    no_risk_nodes = candidates[reduced_policy]

    # Non-counterfactual heuristic comparator.
    # Uses the original DRAD_SCORE candidate implementation.
    no_cf_nodes = mpc.candidate_drad_score_nodes(
        pattern=pattern,
        defense_budget=budget,
        attack_information_ratio=info,
        active_nodes=preview["active_infectious_nodes"],
        exposed_nodes=preview["exposed_nodes"],
        table_dir="outputs/tables",
        seed=seed,
    )

    cdp_nodes = core.select_protected_nodes(
        strategy="CDP",
        defense_budget=budget,
        pattern=pattern,
        seed=seed,
        table_dir="outputs/tables",
    )

    return {
        "nodes": {
            "CDP": cdp_nodes,
            "Full": full_nodes,
            "NoSafeGuard": no_guard_nodes,
            "NoDynamicRisk": no_risk_nodes,
            "RiskHeuristicNoCF": no_cf_nodes,
        },
        "policies": {
            "CDP": "CDP",
            "Full": full_policy,
            "NoSafeGuard": best_all,
            "NoDynamicRisk": reduced_policy,
            "RiskHeuristicNoCF": "DRAD_SCORE",
        },
        "full_guard": full_meta["safe_guard"],
    }


def aggregate(records):
    mean_pas = float(np.mean([
        x["PAS"] for x in records
    ]))

    ps = 1.0 - float(np.mean([
        x["failed"] for x in records
    ]))

    return {
        "EP": mean_pas * ps,
        "PAS": mean_pas,
        "PS": ps,
        "RT": float(np.mean([x["RT"] for x in records])),
        "RI": float(np.mean([x["RI"] for x in records])),
        "RE": float(np.mean([x["RE"] for x in records])),
    }


def run_scenario(scenario_id, scenario, n_runs):
    pattern, info, budget, onset = scenario

    graph = core.build_comm_graph(
        n_nodes=118, seed=42
    )

    actual_by_variant = {
        variant: [] for variant in VARIANTS
    }

    detail_rows = []

    for r in range(n_runs):
        seed = BASE_SEED + r + onset * 1000 + budget * 17

        decision = collect_full_decision(
            graph, scenario, seed
        )

        nodes_by_variant = decision["nodes"]
        policies = decision["policies"]

        # The same node selection is simulated only once
        # under the same scenario and random seed.
        actual_cache = {}

        cdp_nodes = tuple(
            sorted(nodes_by_variant["CDP"])
        )

        cdp_actual = simulate_actual(
            graph, scenario, cdp_nodes, seed
        )

        actual_cache[cdp_nodes] = cdp_actual

        for variant in VARIANTS:
            selected_nodes = tuple(
                sorted(nodes_by_variant[variant])
            )

            if selected_nodes not in actual_cache:
                actual_cache[selected_nodes] = (
                    simulate_actual(
                        graph,
                        scenario,
                        selected_nodes,
                        seed,
                    )
                )

            actual = actual_cache[selected_nodes]

            actual_by_variant[variant].append(actual)

            detail_rows.append({
                "scenario_id": scenario_id,
                "pattern": pattern,
                "info": info,
                "budget": budget,
                "onset": onset,
                "run_index": r,
                "seed": seed,
                "variant": variant,
                "selected_policy": policies[variant],
                "actual_switch": int(
                    selected_nodes != cdp_nodes
                ),
                "full_safe_guard": decision["full_guard"],
                "PAS": actual["PAS"],
                "failed": actual["failed"],
                "RT": actual["RT"],
                "RI": actual["RI"],
                "RE": actual["RE"],
                "CDP_PAS": cdp_actual["PAS"],
                "paired_PAS_gain": (
                    cdp_actual["PAS"] - actual["PAS"]
                ),
            })

    baseline = aggregate(actual_by_variant["CDP"])
    summary_rows = []

    for variant in VARIANTS:
        metrics = aggregate(actual_by_variant[variant])

        group = [
            row for row in detail_rows
            if row["variant"] == variant
        ]

        gain_pct = (
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
            "lookahead_runs": N_CF,
            "EP": metrics["EP"],
            "PAS": metrics["PAS"],
            "PS": metrics["PS"],
            "RT": metrics["RT"],
            "RI": metrics["RI"],
            "RE": metrics["RE"],
            "CDP_EP": baseline["EP"],
            "EP_gain_pct": gain_pct,
            "switch_rate": float(np.mean([
                x["actual_switch"] for x in group
            ])),
            "negative_paired_PAS_rate": float(
                np.mean([
                    x["paired_PAS_gain"] < -1e-12
                    for x in group
                ])
            ),
        })

    return summary_rows, detail_rows


def check_reference(mode, scenario_id, scenario, rows):
    if mode == "pilot":
        filename = (
            "02_safeguard_pilot_"
            "20261018_031803_578077_summary.csv"
        )
    else:
        filename = (
            "02_safeguard_full_"
            "20261018_031835_602501_summary.csv"
        )

    path = OUT / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Reference results missing: {path}"
        )

    reference = pd.read_csv(path)

    original = reference[
        (reference["scenario_id"] == scenario_id)
        & np.isclose(reference["gamma"], GAMMA)
        & np.isclose(reference["kappa"], KAPPA)
    ]

    if len(original) != 1:
        raise RuntimeError(
            "Expected exactly one reference result."
        )

    original = original.iloc[0]

    full = next(
        x for x in rows if x["variant"] == "Full"
    )

    cdp = next(
        x for x in rows if x["variant"] == "CDP"
    )

    checks = [
        (full["EP"], float(original["DRAD_EP"])),
        (cdp["EP"], float(original["CDP_EP"])),
        (full["switch_rate"],
         float(original["switch_rate"])),
    ]

    if not all(
        np.isclose(a, b, atol=1e-10, rtol=0)
        for a, b in checks
    ):
        raise RuntimeError(
            f"REFERENCE VALIDATION FAILED: "
            f"scenario={scenario_id}, "
            f"Full_EP={full['EP']}, "
            f"Reference_EP={original['DRAD_EP']}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["pilot", "full"],
        required=True,
    )

    args = parser.parse_args()

    check_original_implementation()

    if args.mode == "pilot":
        scenarios = PILOT_SCENARIOS
        n_runs = 3
    else:
        scenarios = FULL_SCENARIOS
        n_runs = 10

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S_%f")

    prefix = f"03_ablation_{args.mode}_{timestamp}"

    summary_path = OUT / f"{prefix}_summary.csv"
    detail_path = OUT / f"{prefix}_per_run.csv"
    config_path = OUT / f"{prefix}_config.json"

    config = {
        "experiment": "TNSE component ablation",
        "mode": args.mode,
        "variants": VARIANTS,
        "scenarios": scenarios,
        "n_runs": n_runs,
        "lookahead_runs": N_CF,
        "gamma": GAMMA,
        "kappa": KAPPA,
        "base_seed": BASE_SEED,
        "beta": BETA,
        "alpha": ALPHA,
        "delta": DELTA,
        "source_hashes": {
            name: file_hash(ROOT / "src" / name)
            for name in [
                "drad_mpc.py",
                "run_experiments_core.py",
                "seid_malware.py",
            ]
        },
    }

    config_path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("=" * 70, flush=True)
    print("TNSE COMPONENT ABLATION", flush=True)
    print("Mode:", args.mode, flush=True)
    print("Scenarios:", len(scenarios), flush=True)
    print("Runs:", n_runs, flush=True)
    print("Lookahead:", N_CF, flush=True)
    print("Variants:", VARIANTS, flush=True)
    print("=" * 70, flush=True)

    all_rows = []
    start = time.perf_counter()

    for i, scenario in enumerate(scenarios, 1):
        rows, details = run_scenario(
            i, scenario, n_runs
        )

        # Compare Full and CDP against already validated
        # previous Safe-Guard sensitivity results.
        check_reference(
            args.mode, i, scenario, rows
        )

        pd.DataFrame(rows).to_csv(
            summary_path,
            mode="a",
            header=(i == 1),
            index=False,
        )

        pd.DataFrame(details).to_csv(
            detail_path,
            mode="a",
            header=(i == 1),
            index=False,
        )

        all_rows.extend(rows)

        full = next(
            x for x in rows
            if x["variant"] == "Full"
        )

        print(
            f"[{i:02d}/{len(scenarios)}] "
            f"{scenario} | "
            f"Full_gain={full['EP_gain_pct']:+.2f}% | "
            f"switch={full['switch_rate']:.1%} | "
            f"REFERENCE OK | "
            f"elapsed={time.perf_counter()-start:.1f}s",
            flush=True,
        )

    result = pd.DataFrame(all_rows)

    aggregate = (
        result.groupby("variant", as_index=False)
        .agg(
            mean_EP=("EP", "mean"),
            mean_PAS=("PAS", "mean"),
            mean_RT=("RT", "mean"),
            mean_RI=("RI", "mean"),
            mean_RE=("RE", "mean"),
            mean_gain_pct=("EP_gain_pct", "mean"),
            mean_switch_rate=("switch_rate", "mean"),
            mean_negative_PAS_rate=(
                "negative_paired_PAS_rate", "mean"
            ),
        )
    )

    aggregate_path = (
        OUT / f"{prefix}_aggregate.csv"
    )

    aggregate.to_csv(
        aggregate_path, index=False
    )

    print("\n" + "=" * 70, flush=True)
    print("ABLATION AGGREGATE RESULTS", flush=True)
    print("=" * 70, flush=True)
    print(
        aggregate.to_string(index=False),
        flush=True,
    )

    print("\nSummary:", summary_path)
    print("Per-run:", detail_path)
    print("Aggregate:", aggregate_path)
    print("Config:", config_path)
    print("Runtime:", round(
        time.perf_counter()-start, 2
    ), "seconds")

    print("ABLATION EXPERIMENT COMPLETED")


if __name__ == "__main__":
    main()
