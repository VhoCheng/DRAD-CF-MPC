from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd

from pypower.api import (
    case14, case57, case118, case300,
    runpf, ppoption,
)
from pypower.idx_bus import (
    BUS_I, PD, QD, VM, VA,
)
from pypower.idx_brch import (
    BR_STATUS, RATE_A, PF, QF, PT, QT,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "revision_experiments/results"
SOURCE = ROOT / "revision_experiments"

OUT = RESULTS / "30_31_physical_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

CASES = {
    14: case14,
    57: case57,
    118: case118,
    300: case300,
}

PROFILES = [
    "Morning",
    "Solar-Peak",
    "Evening-Peak",
]

JOINT_SCRIPT = SOURCE / "27_29_joint_suite.py"

spec = importlib.util.spec_from_file_location(
    "tnse_joint_reference",
    JOINT_SCRIPT,
)

joint = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = joint
spec.loader.exec_module(joint)


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def holm(p):
    p = np.asarray(p, dtype=float)

    order = np.argsort(p)
    n = len(p)

    ordered = p[order] * (
        n - np.arange(n)
    )

    ordered = np.minimum(
        1.0,
        np.maximum.accumulate(ordered),
    )

    result = np.empty(n)
    result[order] = ordered

    return result


def solve(ppc, algorithm=1, warm=None):
    case = copy.deepcopy(ppc)

    if warm is not None:
        old = warm["bus"]
        new = case["bus"]

        if not np.array_equal(
            old[:, BUS_I],
            new[:, BUS_I],
        ):
            raise RuntimeError(
                "Warm-start bus ID mismatch"
            )

        case["bus"][:, VM] = old[:, VM]
        case["bus"][:, VA] = old[:, VA]

    options = ppoption(
        VERBOSE=0,
        OUT_ALL=0,
        PF_ALG=algorithm,
        PF_MAX_IT=80,
        PF_MAX_IT_FD=150,
        PF_TOL=1e-7,
    )

    start = time.perf_counter()

    with warnings.catch_warnings(
        record=True
    ) as captured:

        warnings.simplefilter("always")

        try:
            solved, success = runpf(
                case,
                options,
            )

            finite_voltage = np.all(
                np.isfinite(
                    solved["bus"][:, [VM, VA]]
                )
            )

            finite_branch = np.all(
                np.isfinite(
                    solved["branch"][
                        :, [PF, QF, PT, QT]
                    ]
                )
            )

            valid = bool(
                success
                and finite_voltage
                and finite_branch
            )

            error = ""

            if not valid:
                error = (
                    "Unconverged or nonfinite solution"
                )

        except Exception as exc:
            solved = None
            valid = False
            error = (
                type(exc).__name__
                + ": "
                + str(exc)[:180]
            )

    warning_names = sorted(set(
        type(item.message).__name__
        for item in captured
    ))

    record = {
        "success": int(valid),
        "algorithm": algorithm,
        "seconds": round(
            time.perf_counter() - start,
            4,
        ),
        "warnings": "|".join(
            warning_names
        ),
        "error": error,
    }

    return record, (
        solved if valid else None
    )


def continuation(base, target, warm, steps):
    if warm is None:
        return {
            "success": 0,
            "steps": steps,
            "last_completed_fraction": 0.0,
            "failed_fraction": 0.0,
            "error": "Baseline PF unavailable",
        }

    previous = warm

    base_pd = base["bus"][:, PD].copy()
    base_qd = base["bus"][:, QD].copy()

    target_pd = target["bus"][:, PD].copy()
    target_qd = target["bus"][:, QD].copy()

    for i in range(1, steps + 1):
        fraction = i / steps

        intermediate = copy.deepcopy(base)

        intermediate["bus"][:, PD] = (
            base_pd
            + fraction * (target_pd - base_pd)
        )

        intermediate["bus"][:, QD] = (
            base_qd
            + fraction * (target_qd - base_qd)
        )

        record, solved = solve(
            intermediate,
            algorithm=1,
            warm=previous,
        )

        if solved is None:
            # Alternative numerical algorithms at
            # exactly the same intermediate condition.
            for alg in (2, 3):
                record_alt, solved_alt = solve(
                    intermediate,
                    algorithm=alg,
                    warm=previous,
                )

                if solved_alt is not None:
                    solved = solved_alt
                    record = record_alt
                    break

        if solved is None:
            return {
                "success": 0,
                "steps": steps,
                "last_completed_fraction": (
                    (i - 1) / steps
                ),
                "failed_fraction": fraction,
                "error": record["error"],
            }

        previous = solved

    return {
        "success": 1,
        "steps": steps,
        "last_completed_fraction": 1.0,
        "failed_fraction": None,
        "error": "",
    }


def network_ratings(n):
    ppc = CASES[n]()

    branch = ppc["branch"]

    active = branch[:, BR_STATUS] > 0

    rated = (
        active
        & np.isfinite(branch[:, RATE_A])
        & (branch[:, RATE_A] > 0)
    )

    return {
        "case_size": n,
        "total_branch_records": int(len(branch)),
        "active_branch_records": int(
            np.sum(active)
        ),
        "rated_active_branches": int(
            np.sum(rated)
        ),
        "rated_fraction": float(
            np.sum(rated)
            / max(1, np.sum(active))
        ),
        "unrated_active_branches": int(
            np.sum(active & ~rated)
        ),
    }


def diagnose_case(n):
    base = CASES[n]()

    base_result, base_solved = solve(
        base,
        algorithm=1,
    )

    records = []

    for profile in PROFILES:
        target, *_ = joint.operating_case(
            n,
            profile,
        )

        # Direct Newton-Raphson.
        direct, direct_solved = solve(
            target,
            algorithm=1,
        )

        attempts = [{
            "method": "direct_newton",
            **direct,
        }]

        successful_method = (
            "direct_newton"
            if direct_solved is not None
            else ""
        )

        # Try same target with different initialization
        # and independent power-flow algorithms.
        if not successful_method:
            trials = [
                ("base_warm_newton", 1, base_solved),
                ("fast_decoupled_XB", 2, base_solved),
                ("fast_decoupled_BX", 3, base_solved),
            ]

            for name, alg, warm in trials:
                result, solved = solve(
                    target,
                    algorithm=alg,
                    warm=warm,
                )

                attempts.append({
                    "method": name,
                    **result,
                })

                if solved is not None:
                    successful_method = name
                    break

        continuation_records = []

        if not successful_method:
            for steps in (8, 24):
                result = continuation(
                    base,
                    target,
                    base_solved,
                    steps,
                )

                continuation_records.append(
                    result
                )

                if result["success"]:
                    successful_method = (
                        f"continuation_{steps}"
                    )
                    break

        records.append({
            "case_size": n,
            "profile": profile,
            "original_case_pf_success": (
                base_result["success"]
            ),
            "direct_target_pf_success": (
                direct["success"]
            ),
            "recovered_pf_success": int(
                bool(successful_method)
            ),
            "successful_method": (
                successful_method
            ),
            "load_factor": (
                joint.OPERATING_PROFILES[
                    profile
                ][0]
            ),
            "solar_fraction": (
                joint.OPERATING_PROFILES[
                    profile
                ][1]
            ),
            "negative_pd_buses": int(
                np.sum(
                    target["bus"][:, PD] < 0
                )
            ),
            "total_target_net_pd_mw": float(
                target["bus"][:, PD].sum()
            ),
            "attempts": attempts,
            "continuation": (
                continuation_records
            ),
        })

        print(
            f"[IEEE{n} / {profile}] "
            f"original={base_result['success']} "
            f"direct={direct['success']} "
            f"recovered={int(bool(successful_method))} "
            f"method={successful_method or 'NONE'}",
            flush=True,
        )

    return records


def audit_existing_results():
    physical_path = (
        RESULTS
        / "27_29_joint_full"
        / "28_29_physical_per_run.csv"
    )

    paired_path = (
        RESULTS
        / "27_29_joint_full"
        / "27_multiscale_paired_statistics.csv"
    )

    if not physical_path.exists():
        raise FileNotFoundError(
            physical_path
        )

    if not paired_path.exists():
        raise FileNotFoundError(
            paired_path
        )

    df = pd.read_csv(physical_path)
    paired = pd.read_csv(paired_path)

    if len(df) != 5760:
        raise RuntimeError(
            "Expected 5760 physical records"
        )

    if len(paired) != 12:
        raise RuntimeError(
            "Expected 12 paired comparisons"
        )

    paired["holm_adjusted_p"] = holm(
        paired[
            "exploratory_wilcoxon_p"
        ].to_numpy()
    )

    grouped = (
        df.groupby(
            ["case_size", "profile"],
            as_index=False,
        )
        .agg(
            baseline_pf_success=(
                "baseline_pf_converged",
                "mean",
            ),
            attack_pf_success=(
                "attack_pf_converged",
                "mean",
            ),
            n_observations=(
                "variant",
                "size",
            ),
        )
    )

    return paired, grouped, df


def main():
    print("=" * 75)
    print("TNSE EXPERIMENT 30-31")
    print("POWER-FLOW DIAGNOSTICS AND PHYSICAL METRIC AUDIT")
    print("=" * 75)

    ratings = pd.DataFrame([
        network_ratings(n)
        for n in CASES
    ])

    print("\n=== BRANCH RATING COVERAGE ===")
    print(
        ratings.to_string(index=False),
        flush=True,
    )

    paired, grouped, physical = (
        audit_existing_results()
    )

    print(
        "\n=== PREVIOUS OPERATING-PROFILE PF SUCCESS ==="
    )

    print(
        grouped.to_string(index=False),
        flush=True,
    )

    print("\n=== HOLM-CORRECTED EXPLORATORY TESTS ===")

    print(
        paired[[
            "case_size",
            "comparison",
            "gain_pct_of_means",
            "exploratory_wilcoxon_p",
            "holm_adjusted_p",
        ]].to_string(index=False),
        flush=True,
    )

    # Repeat original target operation points
    # with independent convergence diagnostics.
    diagnostics = []

    for n in CASES:
        diagnostics.extend(
            diagnose_case(n)
        )

    simple_diagnostics = pd.DataFrame([
        {
            k: v
            for k, v in record.items()
            if k not in (
                "attempts",
                "continuation",
            )
        }
        for record in diagnostics
    ])

    print("\n=== SOLVER RECOVERY DIAGNOSTICS ===")

    print(
        simple_diagnostics.to_string(index=False),
        flush=True,
    )

    ratings.to_csv(
        OUT / "31_branch_rating_coverage.csv",
        index=False,
    )

    paired.to_csv(
        OUT / "31_holm_adjusted_tests.csv",
        index=False,
    )

    grouped.to_csv(
        OUT / "31_original_pf_success.csv",
        index=False,
    )

    simple_diagnostics.to_csv(
        OUT / "30_solver_summary.csv",
        index=False,
    )

    (OUT / "30_solver_details.json").write_text(
        json.dumps(
            diagnostics,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = {
        "experiment": "30-31 physical diagnostics",
        "reference_physical_sha256": sha(
            RESULTS
            / "27_29_joint_full"
            / "28_29_physical_per_run.csv"
        ),
        "reference_driver_sha256": sha(
            JOINT_SCRIPT
        ),
        "diagnostic_driver_sha256": sha(
            Path(__file__).resolve()
        ),
        "note": (
            "No original results overwritten; "
            "solver methods and starting points "
            "tested without altering target demands."
        ),
    }

    (OUT / "manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nResults saved to:",
        OUT,
        flush=True,
    )

    print(
        "EXPERIMENTS 30-31 COMPLETED SUCCESSFULLY",
        flush=True,
    )


if __name__ == "__main__":
    main()
