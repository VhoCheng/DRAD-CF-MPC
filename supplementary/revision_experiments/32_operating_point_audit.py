from pathlib import Path
import argparse
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd

from pypower.api import (
    case14, case57, case118, case300,
    ppoption, runpf,
)
from pypower.idx_bus import (
    PD, QD, VM, VA, BUS_I,
)
from pypower.idx_brch import (
    BR_STATUS, RATE_A, PF, PT, QF, QT,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = (
    ROOT / "revision_experiments/results"
    / "32_operating_point_audit"
)
OUT.mkdir(parents=True, exist_ok=True)

CASES = {
    14: case14,
    57: case57,
    118: case118,
    300: case300,
}

PROFILES = {
    "Morning": (0.85, 0.05),
    "Solar-Peak": (1.05, 0.30),
    "Evening-Peak": (1.20, 0.02),
}

VARIANTS = [
    "previous_clipped_top20",
    "signed_load_top20",
    "signed_load_distributed_solar",
]

# Import the completed historical implementation
# without executing its main() function.
reference = (
    ROOT / "revision_experiments"
    / "27_29_joint_suite.py"
)

spec = importlib.util.spec_from_file_location(
    "exp27_29_reference_audit",
    reference,
)
joint = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = joint
spec.loader.exec_module(joint)


def digest(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def solve(case, alg=1, warm=None):
    ppc = copy.deepcopy(case)

    if warm is not None:
        if not np.array_equal(
            warm["bus"][:, BUS_I],
            ppc["bus"][:, BUS_I],
        ):
            raise RuntimeError(
                "Warm-start bus IDs differ"
            )

        ppc["bus"][:, VM] = (
            warm["bus"][:, VM]
        )
        ppc["bus"][:, VA] = (
            warm["bus"][:, VA]
        )

    opts = ppoption(
        VERBOSE=0,
        OUT_ALL=0,
        PF_ALG=alg,
        PF_MAX_IT=80,
        PF_MAX_IT_FD=150,
        PF_TOL=1e-7,
    )

    began = time.perf_counter()

    with warnings.catch_warnings(
        record=True
    ) as captured:
        warnings.simplefilter("always")

        try:
            with contextlib.redirect_stdout(
                io.StringIO()
            ):
                result, success = runpf(
                    ppc, opts
                )

            finite = (
                np.all(np.isfinite(
                    result["bus"][:, [VM, VA]]
                ))
                and np.all(np.isfinite(
                    result["branch"][
                        :, [PF, PT, QF, QT]
                    ]
                ))
            )

            ok = bool(success and finite)
            error = (
                ""
                if ok else
                "No convergence or nonfinite solution"
            )

        except Exception as exc:
            result = None
            ok = False
            error = (
                type(exc).__name__
                + ": "
                + str(exc)[:200]
            )

    rec = {
        "success": int(ok),
        "algorithm": alg,
        "elapsed_seconds": round(
            time.perf_counter() - began, 3
        ),
        "warnings": "|".join(sorted(set(
            type(w.message).__name__
            for w in captured
        ))),
        "error": error,
    }

    return rec, (result if ok else None)


def make_profile(n, profile, variant):
    if variant == "previous_clipped_top20":
        return joint.operating_case(
            n, profile
        )[0]

    load_factor, solar_fraction = (
        PROFILES[profile]
    )

    case = CASES[n]()

    # Preserve original negative PD/QD entries.
    original_p = (
        case["bus"][:, PD].copy()
    )
    original_q = (
        case["bus"][:, QD].copy()
    )

    positive_p = np.maximum(
        original_p, 0.0
    )
    positive_q = np.maximum(
        original_q, 0.0
    )

    # Scale positive gross demand only.
    # Original negative injections remain intact.
    case["bus"][:, PD] = (
        original_p
        + (load_factor - 1.0) * positive_p
    )
    case["bus"][:, QD] = (
        original_q
        + (load_factor - 1.0) * positive_q
    )

    eligible = np.where(
        positive_p > 1e-12
    )[0]

    if len(eligible) == 0:
        return case

    if variant == "signed_load_top20":
        ordered = sorted(
            eligible,
            key=lambda i: (
                -positive_p[i], int(i)
            )
        )

        count = max(
            1,
            int(np.ceil(
                0.20 * len(ordered)
            ))
        )

        solar_buses = np.asarray(
            ordered[:count], dtype=int
        )

    elif variant == "signed_load_distributed_solar":
        solar_buses = eligible

    else:
        raise ValueError(variant)

    weights = positive_p[solar_buses]
    weights = weights / weights.sum()

    solar_power = (
        positive_p.sum()
        * solar_fraction
    )

    # Synthetic active-power PV injection.
    # This is a scenario assumption,
    # not measured distributed PV output.
    case["bus"][solar_buses, PD] -= (
        solar_power * weights
    )

    return case


def continuation(base, target, base_solution):
    if base_solution is None:
        return {
            "success": 0,
            "last_fraction": 0.0,
            "failed_fraction": 0.0,
        }

    current = base_solution
    last_fraction = 0.0

    for step in range(1, 17):
        fraction = step / 16.0

        middle = copy.deepcopy(base)

        for col in (PD, QD):
            middle["bus"][:, col] = (
                base["bus"][:, col]
                + fraction * (
                    target["bus"][:, col]
                    - base["bus"][:, col]
                )
            )

        solution = None

        for alg in (1, 2, 3):
            _, trial = solve(
                middle,
                alg=alg,
                warm=current,
            )

            if trial is not None:
                solution = trial
                break

        if solution is None:
            return {
                "success": 0,
                "last_fraction": last_fraction,
                "failed_fraction": fraction,
            }

        current = solution
        last_fraction = fraction

    return {
        "success": 1,
        "last_fraction": 1.0,
        "failed_fraction": None,
    }


def rating_audit(n, baseline_solution):
    branch = CASES[n]()["branch"]

    active = (
        branch[:, BR_STATUS] > 0
    )

    rated = (
        active
        & np.isfinite(branch[:, RATE_A])
        & (branch[:, RATE_A] > 0)
    )

    rates = branch[rated, RATE_A]

    placeholder_candidates = (
        np.isclose(rates, 9900)
        | np.isclose(rates, 9999)
        | (rates >= 5000)
    )

    out = {
        "case_size": n,
        "active_branches": int(active.sum()),
        "rated_branches": int(rated.sum()),
        "rated_fraction": float(
            rated.sum() / max(1, active.sum())
        ),
        "rate_min_mva": float(
            rates.min()
        ),
        "rate_median_mva": float(
            np.median(rates)
        ),
        "rate_max_mva": float(
            rates.max()
        ),
        "rate_ge_5000_fraction": float(
            placeholder_candidates.mean()
        ),
    }

    if baseline_solution is not None:
        solved = baseline_solution["branch"]

        apparent = np.maximum(
            np.hypot(
                solved[:, PF],
                solved[:, QF],
            ),
            np.hypot(
                solved[:, PT],
                solved[:, QT],
            ),
        )

        loading = (
            apparent[rated] / rates
        )

        out["baseline_loading_p95"] = float(
            np.quantile(loading, 0.95)
        )

        out["baseline_loading_max"] = float(
            loading.max()
        )

    else:
        out["baseline_loading_p95"] = np.nan
        out["baseline_loading_max"] = np.nan

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["pilot", "full"],
        required=True,
    )

    args = parser.parse_args()

    sizes = (
        [118, 300]
        if args.mode == "pilot"
        else [14, 57, 118, 300]
    )

    rows = []
    rating_rows = []
    details = []

    print("=" * 76, flush=True)
    print("TNSE EXPERIMENT 32", flush=True)
    print("OPERATING-POINT FEASIBILITY AND RATING AUDIT")
    print("=" * 76, flush=True)

    for n in sizes:
        base = CASES[n]()
        base_record, base_solved = solve(base)

        if base_solved is None:
            raise RuntimeError(
                f"Original IEEE{n} AC PF did not converge"
            )

        p = base["bus"][:, PD]

        print(
            f"\nIEEE{n}: "
            f"original negative-PD buses="
            f"{int(np.sum(p < 0))}, "
            f"negative-PD sum="
            f"{float(p[p < 0].sum()):.3f} MW, "
            f"baseline PF=OK",
            flush=True,
        )

        rating_rows.append(
            rating_audit(
                n, base_solved
            )
        )

        for profile in PROFILES:
            for variant in VARIANTS:
                target = make_profile(
                    n, profile, variant
                )

                direct, solution = solve(
                    target, alg=1
                )

                method = (
                    "direct_newton"
                    if solution is not None
                    else ""
                )

                attempt_log = [{
                    "name": "direct_newton",
                    **direct,
                }]

                if solution is None:
                    for name, alg in [
                        ("warm_newton", 1),
                        ("warm_fast_XB", 2),
                        ("warm_fast_BX", 3),
                    ]:
                        attempt, trial = solve(
                            target,
                            alg=alg,
                            warm=base_solved,
                        )

                        attempt_log.append({
                            "name": name,
                            **attempt,
                        })

                        if trial is not None:
                            solution = trial
                            method = name
                            break

                cont = None

                if (
                    solution is None
                    and n == 300
                ):
                    cont = continuation(
                        base,
                        target,
                        base_solved,
                    )

                    if cont["success"]:
                        method = "continuation_16"

                recovered = bool(
                    method
                )

                row = {
                    "case_size": n,
                    "profile": profile,
                    "variant": variant,
                    "direct_success": (
                        direct["success"]
                    ),
                    "recovered_success": int(
                        recovered
                    ),
                    "successful_method": (
                        method or "NONE"
                    ),
                    "original_negative_PD_count": int(
                        np.sum(p < 0)
                    ),
                    "original_negative_PD_sum_MW": float(
                        p[p < 0].sum()
                    ),
                    "target_net_PD_MW": float(
                        target["bus"][:, PD].sum()
                    ),
                    "target_negative_PD_count": int(
                        np.sum(
                            target["bus"][:, PD] < 0
                        )
                    ),
                    "continuation_last_fraction": (
                        cont["last_fraction"]
                        if cont is not None
                        else np.nan
                    ),
                }

                rows.append(row)

                details.append({
                    **row,
                    "attempts": attempt_log,
                    "continuation": cont,
                })

                print(
                    f"[IEEE{n} {profile} "
                    f"{variant}] "
                    f"direct={direct['success']} "
                    f"recovered={int(recovered)} "
                    f"method={method or 'NONE'}",
                    flush=True,
                )

    table = pd.DataFrame(rows)
    ratings = pd.DataFrame(rating_rows)

    table.to_csv(
        OUT / f"feasibility_{args.mode}.csv",
        index=False,
    )

    ratings.to_csv(
        OUT / f"branch_ratings_{args.mode}.csv",
        index=False,
    )

    (OUT / f"details_{args.mode}.json").write_text(
        json.dumps(
            details, indent=2
        ),
        encoding="utf-8",
    )

    manifest = {
        "mode": args.mode,
        "source_joint_sha256": digest(
            reference
        ),
        "diagnostic_driver_sha256": digest(
            Path(__file__).resolve()
        ),
        "note": (
            "Three explicitly different scenario "
            "constructions; no original results changed."
        ),
    }

    (OUT / f"manifest_{args.mode}.json").write_text(
        json.dumps(
            manifest, indent=2
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 76)
    print("EXPERIMENT 32 - BRANCH RATING AUDIT")
    print("=" * 76)
    print(
        ratings.to_string(index=False),
        flush=True,
    )

    print("\n" + "=" * 76)
    print("EXPERIMENT 32 - OPERATING POINT FEASIBILITY")
    print("=" * 76)
    print(
        table.to_string(index=False),
        flush=True,
    )

    print(
        "\nEXPERIMENT 32 COMPLETED SUCCESSFULLY",
        flush=True,
    )


if __name__ == "__main__":
    main()
