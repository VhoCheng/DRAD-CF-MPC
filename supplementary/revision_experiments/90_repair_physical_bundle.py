from pathlib import Path
import importlib.util
import json
import re
import sys
import ast

import numpy as np
import pandas as pd

from pypower.idx_bus import VM, VA

ROOT = Path(__file__).resolve().parents[1]
EXPS = ROOT / "revision_experiments"
RESULTS = EXPS / "results"

ORIGINAL_33 = EXPS / "33_dispatch_pf.py"
ORIGINAL_34 = EXPS / "34_redispatched_physical_replay.py"

STAGED = RESULTS / "34_staged_dispatch"

TARGET_34 = EXPS / (
    "34_redispatched_physical_replay_v2.py"
)

for p in (ORIGINAL_33, ORIGINAL_34):
    if not p.exists():
        raise FileNotFoundError(p)

def import_file(name, path):
    spec = importlib.util.spec_from_file_location(
        name, path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

e33 = import_file(
    "exp33_repair_reference",
    ORIGINAL_33
)

history = pd.read_csv(
    RESULTS / "33_dispatch_pf" / "summary_full.csv"
)

expected_profiles = [
    "Morning",
    "Solar-Peak",
    "Evening-Peak",
]

print("=" * 72)
print("STAGE A: REBUILD DISPATCH SNAPSHOTS")
print("=" * 72)

records = []

for n in (14, 57, 118, 300):
    original = e33.CASES[n]()

    _, solved_base = e33.solve(original)

    if solved_base is None:
        raise RuntimeError(
            f"IEEE{n}: original PF failed"
        )

    for profile in expected_profiles:
        found = history[
            (history["case_size"] == n)
            & (history["profile"] == profile)
        ]

        if len(found) != 1:
            raise RuntimeError(
                f"Historical row missing: {n}/{profile}"
            )

        old = found.iloc[0]

        if int(old["redispatch_converged"]) != 1:
            raise RuntimeError(
                f"Historical PF was not converged: "
                f"{n}/{profile}"
            )

        if int(old["Pg_limits_satisfied"]) != 1:
            raise RuntimeError(
                f"Historical Pg check failed: "
                f"{n}/{profile}"
            )

        target = e33.make_profile(n, profile)

        target, report = e33.allocate_generation(
            original,
            target,
            solved_base
        )

        # Same solver order as Experiment 33.
        solution = None
        method = "NONE"

        for name, algorithm, warm in [
            ("redispatch_newton", 1, None),
            ("redispatch_warm", 1, solved_base),
            ("redispatch_fast_XB", 2, solved_base),
            ("redispatch_fast_BX", 3, solved_base),
        ]:
            _, trial = e33.solve(
                target,
                warm=warm,
                algorithm=algorithm,
            )

            if trial is not None:
                solution = trial
                method = name
                break

        if solution is None:
            e33.TARGET = target

            solution = e33.continuation(
                original,
                solved_base,
                steps=20,
            )

            if solution is not None:
                method = "redispatch_continuation"

        if solution is None:
            raise RuntimeError(
                f"Rebuilt PF did not converge: "
                f"IEEE{n} {profile}"
            )

        constraints = e33.check_constraints(
            solution,
            original
        )

        if constraints["Pg_violations"] != 0:
            raise RuntimeError(
                f"Rebuilt Pg limit violation: "
                f"IEEE{n} {profile}"
            )

        # Reproduce previously reported operating point.
        for field in (
            "voltage_min",
            "voltage_max",
            "slack_Pg_mw",
        ):
            if not np.isclose(
                float(constraints[field]),
                float(old[field]),
                atol=1e-3,
                rtol=0,
            ):
                raise RuntimeError(
                    f"Exp33 mismatch {n}/{profile}: "
                    f"{field}: rebuilt="
                    f"{constraints[field]}, "
                    f"history={old[field]}"
                )

        for field in (
            "voltage_violations",
            "Pg_violations",
            "Qg_violations",
        ):
            if int(constraints[field]) != int(old[field]):
                raise RuntimeError(
                    f"Exp33 mismatch: "
                    f"{n}/{profile}/{field}"
                )

        # Previously saved inputs must also agree.
        old_snapshot = (
            RESULTS / "33_dispatch_pf"
            / f"case{n}"
            / f"{profile}.npz"
        )

        if old_snapshot.exists():
            with np.load(
                old_snapshot,
                allow_pickle=False,
            ) as data:
                for key, rebuilt in (
                    ("bus_pd", target["bus"][:, e33.PD]),
                    ("bus_qd", target["bus"][:, e33.QD]),
                    ("gen_pg", target["gen"][:, e33.PG]),
                ):
                    if not np.allclose(
                        data[key],
                        rebuilt,
                        atol=1e-8,
                        rtol=0,
                    ):
                        raise RuntimeError(
                            f"Old snapshot differs: "
                            f"{n}/{profile}/{key}"
                        )

        output_dir = STAGED / f"case{n}"
        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output = output_dir / f"{profile}.npz"

        staged_data = {
            "bus_pd": target["bus"][:, e33.PD],
            "bus_qd": target["bus"][:, e33.QD],
            "gen_pg": target["gen"][:, e33.PG],
            "bus_vm": solution["bus"][:, VM],
            "bus_va": solution["bus"][:, VA],
        }

        if output.exists():
            with np.load(
                output,
                allow_pickle=False,
            ) as previous:
                for key, value in staged_data.items():
                    if not np.allclose(
                        previous[key],
                        value,
                        atol=1e-8,
                        rtol=0,
                    ):
                        raise RuntimeError(
                            f"Staged snapshot differs: "
                            f"{output}/{key}"
                        )
        else:
            np.savez_compressed(
                output,
                **staged_data
            )

        records.append({
            "case_size": n,
            "profile": profile,
            "solver_method": method,
            "converged": 1,
            "Pg_violations": (
                constraints["Pg_violations"]
            ),
            "Qg_violations": (
                constraints["Qg_violations"]
            ),
            "voltage_violations": (
                constraints["voltage_violations"]
            ),
            "non_slack_residual_MW": report[
                "slack_residual_target_mw"
            ],
            "original_snapshot_exists": int(
                old_snapshot.exists()
            ),
            "staged_path": str(output),
        })

        print(
            f"[READY] IEEE{n} {profile} "
            f"method={method} "
            f"Pg_bad={constraints['Pg_violations']} "
            f"Qg_bad={constraints['Qg_violations']} "
            f"V_bad={constraints['voltage_violations']}",
            flush=True,
        )

if len(records) != 12:
    raise RuntimeError(
        "Expected exactly 12 dispatch snapshots"
    )

STAGED.mkdir(
    parents=True,
    exist_ok=True
)

pd.DataFrame(records).to_csv(
    STAGED / "validated_snapshots.csv",
    index=False,
)

print("\n12/12 DISPATCH SNAPSHOTS VALIDATED")

print("\n" + "=" * 72)
print("STAGE B: PATCH EXPERIMENT 34 INTO V2")
print("=" * 72)

code = ORIGINAL_34.read_text(
    encoding="utf-8"
)

old_start = code.index(
    "def dispatch_file(n, profile):"
)

old_end = code.index(
    "\n\ndef load_redispatched_case(",
    old_start,
)

replacement = '''
def dispatch_file(n, profile):
    return (
        RESULTS
        / "34_staged_dispatch"
        / f"case{n}"
        / f"{profile}.npz"
    )
'''

code = (
    code[:old_start]
    + replacement
    + code[old_end:]
)

old_import = (
    "from pypower.idx_bus import PD, QD"
)

new_import = (
    "from pypower.idx_bus import PD, QD, VM, VA"
)

if code.count(old_import) != 1:
    raise RuntimeError(
        "Cannot safely update PYPOWER import"
    )

code = code.replace(
    old_import,
    new_import,
    1,
)

old_read = '''    with np.load(path, allow_pickle=False) as data:
        pd_target = data["bus_pd"].copy()
        qd_target = data["bus_qd"].copy()
        pg_target = data["gen_pg"].copy()'''

new_read = '''    with np.load(path, allow_pickle=False) as data:
        pd_target = data["bus_pd"].copy()
        qd_target = data["bus_qd"].copy()
        pg_target = data["gen_pg"].copy()
        vm_target = data["bus_vm"].copy()
        va_target = data["bus_va"].copy()'''

if code.count(old_read) != 1:
    raise RuntimeError(
        "Cannot safely patch snapshot loader"
    )

code = code.replace(
    old_read,
    new_read,
    1,
)

old_assign = (
    '    case["gen"][:, PG] = pg_target'
)

new_assign = '''    case["gen"][:, PG] = pg_target
    case["bus"][:, VM] = vm_target
    case["bus"][:, VA] = va_target'''

if code.count(old_assign) != 1:
    raise RuntimeError(
        "Cannot safely patch voltage initialization"
    )

code = code.replace(
    old_assign,
    new_assign,
    1,
)

# Remaining demand assigned to the slack is NOT
# by itself proof of a failed AC power-flow solution.
# Keep convergence and actual Pg-limit checks.

needle = '                or int(x["small_slack_residual"]) != 1\\n'
count = code.count(needle)

if count == 1:
    code = code.replace(needle, "", 1)

if count != 1:
    raise RuntimeError(
        "Cannot safely patch residual check"
    )

old_folder = (
    'f"34_redispatched_physical_{args.mode}"'
)

new_folder = (
    'f"34_redispatched_physical_v2_{args.mode}"'
)

if code.count(old_folder) != 1:
    raise RuntimeError(
        "Cannot safely separate v2 output directory"
    )

code = code.replace(
    old_folder,
    new_folder,
    1,
)

code = code.replace(
    '"TNSE EXPERIMENT 34"',
    '"TNSE EXPERIMENT 34 V2"',
    1,
)

ast.parse(code)

if TARGET_34.exists():
    if TARGET_34.read_text(
        encoding="utf-8"
    ) != code:
        raise RuntimeError(
            "V2 file already exists with different content"
        )
else:
    TARGET_34.write_text(
        code,
        encoding="utf-8"
    )

print("Experiment 34 V2 syntax: OK")
print("Original 33 and 34 files unchanged")
print("Output:", TARGET_34)
print("PHYSICAL EXPERIMENT REPAIR READY")
