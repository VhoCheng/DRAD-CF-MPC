from pathlib import Path
import ast
import hashlib
import json
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "revision_experiments"
S = ROOT / "src"
R = E / "results"
OUT = R / "44_terminal_gate"
OUT.mkdir(parents=True, exist_ok=True)

REQ = {
    "02 gamma-kappa": (
        "02_safeguard_full*_parameter_aggregate.csv", 25
    ),
    "03 ablation": (
        "03_ablation_full*_per_run.csv", 3000
    ),
    "08 observation": (
        "08_observation_full*_per_run.csv", 3600
    ),
    "09 observation audit": (
        "09_observation_audit*_per_run.csv", 3600
    ),
    "14 independent test": (
        "14_replication_test/per_run.csv", 900
    ),
    "18 objective weights": (
        "18_objective_weights_full/aggregate.csv", 6
    ),
    "19 risk weights": (
        "19_dynamic_risk_full/aggregate.csv", 7
    ),
    "23 calibration": (
        "23_calibration_full/per_decision.csv", 320
    ),
    "26 multiscale": (
        "26_multiscale_full/per_run.csv", 960
    ),
    "34 physical PF": (
        "34_redispatched_physical_v2_full/"
        "physical_per_run.csv", 5760
    ),
    "35 paired statistics": (
        "35_37_final_evidence/"
        "35_physical_paired_statistics.csv", 12
    ),
    "37 timing": (
        "35_37_final_evidence/"
        "37_serial_timing.csv", 32
    ),
    "38 same-state test": (
        "38_same_state_full/per_run.csv", 2880
    ),
    "39 ablation": (
        "39_mechanism_full/per_decision.csv", 9600
    ),
    "40 observation paired": (
        "40_paired_observation_full/"
        "per_action_paired.csv", 9600
    ),
    "42 guard": (
        "42_final_guard_audit/"
        "42_guard_tradeoff_by_condition.csv", 24
    ),
    "42 tail": (
        "42_final_guard_audit/"
        "42_tail_risk_and_regret.csv", 12
    ),
}

def path(pattern):
    matches = [
        x for x in sorted(R.glob(pattern))
        if x.is_file()
    ]
    return matches[-1] if matches else None

def nodes(x):
    obj = ast.literal_eval(x) if isinstance(x, str) else x
    if not isinstance(obj, (list, tuple, set)):
        raise ValueError("Unexpected protected-node structure")
    return tuple(sorted(int(y) for y in obj))

print("\n=== A. OFFICIAL EXPERIMENT FILES ===")

rows = []

for name, (pattern, expected) in REQ.items():
    p = path(pattern)
    n = len(pd.read_csv(p)) if p else -1
    ok = n == expected

    print(
        f"{'PASS' if ok else 'CHECK'} | "
        f"{name}: {n}/{expected}"
    )

    rows.append({
        "name": name,
        "n": n,
        "expected": expected,
        "file": str(p) if p else "",
        "file_pass": ok
    })

pd.DataFrame(rows).to_csv(
    OUT / "experiment_inventory.csv",
    index=False
)

print("\n=== B. SAFEGUARD GAMMA/KAPPA ===")

p = path(REQ["02 gamma-kappa"][0])

if p:
    t = pd.read_csv(p)
    pairs = t[["gamma", "kappa"]].drop_duplicates()

    print("Distinct pairs:", len(pairs))
    print("Gamma:", sorted(t.gamma.unique()))
    print("Kappa:", sorted(t.kappa.unique()))

print("\n=== C. SOURCE CODE / PAPER CLAIMS ===")

seid = (S / "seid_malware.py").read_text(
    encoding="utf8"
)
mpc = (S / "drad_mpc.py").read_text(
    encoding="utf8"
)
core = (S / "run_experiments_core.py").read_text(
    encoding="utf8"
)

fn = next(
    x for x in ast.parse(seid).body
    if isinstance(x, ast.FunctionDef)
    and x.name == "simulate_seid_once"
)

args = [x.arg for x in fn.args.args]

checks = {
    "original_accepts_initial_state": any(
        x in args for x in (
            "initial_states",
            "initial_state",
            "state_vector"
        )
    ),
    "original_PAS_zero_if_failed": bool(
        re.search(
            r'pas\s*=\s*0(?:\.0)?\s+if\s+'
            r'result\[.attack_failed.\]',
            mpc
        )
    ),
    "original_EP_multiplies_meanPAS_by_PS": bool(
        re.search(
            r'\bep\s*=\s*mean_pas\s*\*\s*ps\b',
            mpc
        )
    ),
    "separate_preview_simulation": (
        "preview = simulate_seid_once(" in core
    ),
    "original_source_sha256": {
        f: hashlib.sha256(
            (S / f).read_bytes()
        ).hexdigest()
        for f in [
            "seid_malware.py",
            "drad_mpc.py",
            "run_experiments_core.py"
        ]
    }
}

checks["EP_double_count_possible"] = (
    checks["original_PAS_zero_if_failed"]
    and checks["original_EP_multiplies_meanPAS_by_PS"]
)

print(json.dumps(
    {
        k: v for k, v in checks.items()
        if k != "original_source_sha256"
    },
    indent=2
))

(OUT / "source_gate.json").write_text(
    json.dumps(checks, indent=2),
    encoding="utf8"
)

print("\n=== D. ABLATION ACTION-DIVERGENCE ===")

p = R / "39_mechanism_full/per_decision.csv"
df = pd.read_csv(p)

clean = df[df.condition == "clean"].copy()
clean["node_set"] = clean.protected_nodes.map(nodes)

key = [
    "case_size",
    "scenario_id",
    "run_index"
]

w = clean.set_index(
    key + ["method"]
)["node_set"].unstack("method")

a = []

for n, part in w.groupby(level="case_size"):
    for method in [
        "NO_DYNAMIC",
        "NO_CF",
        "NO_GUARD",
        "CDP"
    ]:
        mismatch = float(
            (part.FULL != part[method]).mean()
        )

        a.append({
            "case_size": n,
            "versus": method,
            "action_mismatch_fraction": mismatch,
            "n_pairs": len(part)
        })

        print(
            f"IEEE{n}: FULL vs {method}, "
            f"differing actions {mismatch:.2%}"
        )

pd.DataFrame(a).to_csv(
    OUT / "action_divergence.csv",
    index=False
)

print("\n=== E. EP METRIC ACCOUNTING ===")

p = R / "26_multiscale_full/per_run.csv"
z = pd.read_csv(p)

required = {
    "PAS", "failed", "variant",
    "scenario_id", "case_size"
}

if not required.issubset(z.columns):
    raise RuntimeError(
        "Missing EP audit columns: "
        + str(required - set(z.columns))
    )

z["failed"] = pd.to_numeric(z["failed"])
z["PAS"] = pd.to_numeric(z["PAS"])

bad = z[
    (z.failed == 1)
    & (z.PAS.abs() > 1e-10)
]

g = (
    z.groupby(
        ["case_size", "scenario_id", "variant"],
        as_index=False
    )
    .agg(
        mean_PAS=("PAS", "mean"),
        fail_rate=("failed", "mean")
    )
)

g["PS"] = 1 - g.fail_rate

g["EP_if_unconditionalPAS"] = g.mean_PAS
g["EP_if_multiplied_again"] = (
    g.mean_PAS * g.PS
)

g["double_discount"] = (
    g.EP_if_unconditionalPAS
    - g.EP_if_multiplied_again
)

g.to_csv(
    OUT / "ep_accounting_diagnostic.csv",
    index=False
)

print(
    "Nonzero PAS on failed runs:",
    len(bad),
    "of",
    int((z.failed == 1).sum())
)

print(
    "Mean extra EP discount:",
    round(float(g.double_discount.mean()), 8)
)

print("\n=== F. REMAINING METHODOLOGICAL GAPS ===")

items = [
    (
        "CONFIRMED",
        "gamma/kappa grid",
        "02 Full: 5x5; 15000 runs"
    ),
    (
        "NEEDS_METHOD_CHECK",
        "dynamic-risk ablation",
        "Frozen risk features may not remove all state dependence"
    ),
    (
        "NEEDS_METHOD_CHECK",
        "counterfactual ablation",
        "NO_CF also bypasses guard; compound removal"
    ),
    (
        "NEEDS_IMPL_CHANGE",
        "strict same-state MPC",
        "Original simulator lacks state-continuation input"
    ),
    (
        "NEEDS_IMPL_CHANGE",
        "EP formula",
        "Check success-probability double discount"
    ),
    (
        "NEEDS_PAPER_ALIGNMENT",
        "risk weighting",
        "Table II and executed candidate weights differ"
    ),
    (
        "NEEDS_PHYSICAL_CHECK",
        "physical operation",
        "AC PF alone does not prove P/Q/V/thermal feasibility"
    ),
    (
        "NEEDS_ASSUMPTIONS",
        "deployment latency",
        "End-to-end transport and actuation are not measured"
    )
]

pd.DataFrame(
    items,
    columns=["status", "issue", "interpretation"]
).to_csv(
    OUT / "remaining_gaps.csv",
    index=False
)

for status, issue, note in items:
    print(f"{status}: {issue} — {note}")

print("\nEXP44 AUDIT COMPLETED")
print("OLD EXPERIMENTS UNCHANGED")
print("Outputs:", OUT)
