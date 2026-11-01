from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "revision_experiments/results"
OUT = R / "42_final_guard_audit"
OUT.mkdir(parents=True, exist_ok=True)

P39 = R / "39_mechanism_full/per_decision.csv"
P40 = R / "40_paired_observation_full/per_action_paired.csv"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

a = pd.read_csv(P39)
b = pd.read_csv(P40)

assert len(a) == 9600, "Exp39 incomplete"
assert len(b) == 9600, "Exp40 incomplete"

keys = [
    "case_size", "scenario_id",
    "run_index", "condition"
]

methods = [
    "CDP", "FULL", "NO_GUARD",
    "NO_DYNAMIC", "NO_CF"
]

for name, data in [("Exp39", a), ("Exp40", b)]:
    assert not data.duplicated(
        keys + ["method"]
    ).any(), f"{name}: duplicate records"
    assert set(data["method"]) == set(methods)

wide = b.pivot(
    index=keys,
    columns="method",
    values="PAS"
).reset_index()

assert len(wide) == 1920
assert wide[methods].notna().all().all()

rng = np.random.default_rng(20261020)

rows = []

for (n, condition), block in wide.groupby(
    ["case_size", "condition"]
):
    scene = block.groupby(
        "scenario_id"
    )[methods].mean()

    assert len(scene) == 20
    assert block["run_index"].nunique() == 4

    # Positive values mean Safe-Guard is better
    # than the unguarded candidate selection.
    difference = (
        scene["NO_GUARD"] - scene["FULL"]
    ).to_numpy()

    boots = rng.choice(
        difference,
        size=(3000, len(difference)),
        replace=True
    ).mean(axis=1)

    nonzero = difference[
        abs(difference) > 1e-12
    ]

    if len(nonzero) == 0:
        pvalue = 1.0
    else:
        pvalue = float(
            wilcoxon(nonzero).pvalue
        )

    rows.append({
        "case_size": int(n),
        "condition": condition,
        "n_independent_scenarios": len(scene),
        "mean_CDP": scene["CDP"].mean(),
        "mean_FULL": scene["FULL"].mean(),
        "mean_NO_GUARD": scene["NO_GUARD"].mean(),
        "guard_benefit": difference.mean(),
        "guard_CI_low": np.quantile(
            boots, 0.025
        ),
        "guard_CI_high": np.quantile(
            boots, 0.975
        ),
        "guard_helps_scenarios": int(
            (difference > 1e-12).sum()
        ),
        "guard_harms_scenarios": int(
            (difference < -1e-12).sum()
        ),
        "guard_ties_scenarios": int(
            (abs(difference) <= 1e-12).sum()
        ),
        "FULL_worse_than_CDP_scenarios": int(
            (
                scene["FULL"] >
                scene["CDP"] + 1e-12
            ).sum()
        ),
        "NO_GUARD_worse_than_CDP_scenarios": int(
            (
                scene["NO_GUARD"] >
                scene["CDP"] + 1e-12
            ).sum()
        ),
        "wilcoxon_p": pvalue
    })

stats = pd.DataFrame(rows)

# One multiplicity family: four scales x six conditions.
p = stats["wilcoxon_p"].to_numpy()
order = np.argsort(p)
adjusted = np.empty(len(p))

adjusted[order] = np.minimum(
    1.0,
    np.maximum.accumulate(
        p[order] * (
            len(p) - np.arange(len(p))
        )
    )
)

stats["holm_p_24"] = adjusted

stats.to_csv(
    OUT / "42_guard_tradeoff_by_condition.csv",
    index=False
)

clean = b[b["condition"] == "clean"]

clean_summary = (
    clean.groupby(
        ["case_size", "method"],
        as_index=False
    )
    .agg(
        mean_PAS=("PAS", "mean"),
        p95_PAS=(
            "PAS",
            lambda x: np.quantile(x, 0.95)
        )
    )
)

clean_summary.to_csv(
    OUT / "42_clean_method_summary.csv",
    index=False
)

switches = (
    a.groupby(
        ["case_size", "condition", "method"],
        as_index=False
    )
    .agg(
        switch_rate=(
            "switched_from_CDP", "mean"
        )
    )
)

switches.to_csv(
    OUT / "42_switch_rates.csv",
    index=False
)

# Analytical communication-size examples.
# Assumptions:
# 1 byte for every observed node state
# 2 bytes for each selected node identifier
# 24 bytes per message for protocol metadata
# No encryption, retransmission, topology updates,
# propagation delay or queueing included.
payload_rows = []

for n in [14, 57, 118, 300]:
    subset = a[a["case_size"] == n]
    budgets = sorted(
        subset["budget"].unique()
    )

    for budget in budgets:
        state_bytes = int(n + 24)
        action_bytes = int(2 * budget + 24)
        total_bytes = state_bytes + action_bytes

        payload_rows.append({
            "case_size": n,
            "budget": int(budget),
            "state_bytes_assumed": state_bytes,
            "action_bytes_assumed": action_bytes,
            "total_bytes_assumed": total_bytes,
            "serialization_ms_10Mbps": (
                total_bytes * 8 / 1e7 * 1000
            ),
            "serialization_ms_100Mbps": (
                total_bytes * 8 / 1e8 * 1000
            )
        })

pd.DataFrame(payload_rows).to_csv(
    OUT / "42_communication_size_assumptions.csv",
    index=False
)

manifest = {
    "experiment": "42 final guard trade-off analysis",
    "exp39_sha256": sha(P39),
    "exp40_sha256": sha(P40),
    "scenario_unit": (
        "20 scenarios per network; four MC runs "
        "averaged within each scenario"
    ),
    "comparison": (
        "NO_GUARD PAS minus FULL PAS, "
        "positive means guard is beneficial"
    ),
    "communication": (
        "Analytical payload assumption, "
        "NOT measured network delay"
    )
}

(OUT / "manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8"
)

print("\n===== EXP42 CLEAN GUARD TRADE-OFF =====")
print(
    stats[
        stats["condition"] == "clean"
    ].to_string(index=False)
)

print("\n===== EXP42 SWITCH RATES =====")
print(
    switches[
        switches["condition"] == "clean"
    ].to_string(index=False)
)

print("\n===== EXP42 COMPLETENESS =====")
print("Paired cases:", len(stats), "/24")
print("Independent scenarios per case:",
      stats["n_independent_scenarios"].unique())
print("Saved:", OUT)
print("\nEXPERIMENT 42 COMPLETED SUCCESSFULLY")
