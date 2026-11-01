#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TNSE Exp50 V2: state-local candidate study.

Reuses the Exp49 split-confirmation engine.

pilot:
  Test candidate generation and simulation.

validate:
  Compare three predeclared local-risk weights
  using validation seeds only.

test:
  Evaluate one frozen configuration on new seeds.

All results are retained, including negative findings.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "revision_experiments"

OUT = (
    EXP / "results" /
    "50_state_local_v2"
)
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

import tnse_exp49 as e49

ref = e49.ref

# Send Exp49's reusable checkpoint engine
# to a completely new output directory.
e49.SAVED = OUT

ORIGINAL_CHOOSE = ref.choose
ORIGINAL_CANDIDATES = ref.candidate_nodes

PATTERNS = ["DDAC", "RC"]
BUDGETS = [20, 40, 60]
ONSETS = [90, 210]

ALPHAS = [0.25, 0.50, 0.75]

VAL_SEED = 5000
TEST_SEED = 6000

METHODS = [
    "CDP",
    "FULL",
    "NO_DYNAMIC",
    "NO_GUARD",
    "STATE_LOCAL",
    "STATIC_EXTRA"
]

CURRENT_METHOD = None
ALPHA = .50

ADJ_DENSE = None
PHYSICAL = None
DEGREE = None


def normalize(x):
    x = np.asarray(x, dtype=float)

    finite = np.isfinite(x)

    if not finite.all():
        raise ValueError(
            "Nonfinite candidate score"
        )

    lo = float(x.min())
    hi = float(x.max())

    if hi-lo < 1e-12:
        return np.zeros_like(x)

    return (x-lo)/(hi-lo)


def score_candidate(
    pattern, state, method
):
    """
    Static and dynamic candidates share the
    same structural/physical baseline.

    Dynamic mode adds current local infection
    and exposure information.
    """

    state = np.asarray(state).reshape(-1)
    n = len(state)

    if len(DEGREE) != n:
        raise ValueError(
            "State / adjacency mismatch"
        )

    importance = normalize(
        PHYSICAL[pattern]
    )

    structural = normalize(DEGREE)

    common = (
        .75*importance
        + .25*structural
    )

    if method == "STATIC_EXTRA":
        return common

    infected = (
        state == 2
    ).astype(float)

    exposed = (
        state == 1
    ).astype(float)

    pressure = (
        ADJ_DENSE @ (
            infected + .25*exposed
        )
    )

    pressure = (
        pressure /
        np.maximum(DEGREE, 1)
    )

    # Two-hop exposure context.
    two_hop = (
        ADJ_DENSE @ pressure
    )

    two_hop = (
        two_hop /
        np.maximum(DEGREE, 1)
    )

    local = normalize(
        .8*pressure + .2*two_hop
    )

    score = (
        ALPHA*local
        + (1-ALPHA)*common
    )

    # Infected nodes are not preferred as
    # new preventive-protection targets.
    eligible = (
        state != 2
    )

    if int(eligible.sum()) > 0:
        score = score.copy()
        score[~eligible] = -1.0

    return score


def candidate_wrapper(
    pattern, budget, state,
    info, seed, dynamic
):
    """
    All methods use the original candidate
    generator.

    STATE_LOCAL and STATIC_EXTRA deliberately
    share identical original candidate settings.

    Only their additional candidate differs.
    """

    if CURRENT_METHOD not in (
        "STATE_LOCAL",
        "STATIC_EXTRA"
    ):
        return ORIGINAL_CANDIDATES(
            pattern, budget, state,
            info, seed, dynamic
        )

    # The same original candidate set for both
    # new methods: avoid confounded ablation.
    candidates = dict(
        ORIGINAL_CANDIDATES(
            pattern,
            budget,
            state,
            info,
            seed,
            True
        )
    )

    scores = score_candidate(
        pattern,
        state,
        CURRENT_METHOD
    )

    n = len(scores)

    if budget > n:
        raise ValueError(
            "Budget exceeds network size"
        )

    # Deterministic tie-breaking.
    order = np.argsort(
        -scores,
        kind="stable"
    )

    selected = tuple(
        sorted(
            int(x)
            for x in order[:budget]
        )
    )

    assert len(selected) == budget
    assert len(set(selected)) == budget

    existing = {
        tuple(
            sorted(int(z) for z in nodes)
        )
        for nodes in candidates.values()
    }

    if selected not in existing:
        candidates["LOCAL_EXTRA"] = selected

    return candidates


def choose_wrapper(method, *args, **kwargs):
    global CURRENT_METHOD

    CURRENT_METHOD = method

    try:
        return ORIGINAL_CHOOSE(
            method,
            *args,
            **kwargs
        )
    finally:
        CURRENT_METHOD = None


ref.candidate_nodes = candidate_wrapper
ref.choose = choose_wrapper


def make_cases(first_seed, count):
    return [
        (pattern, budget, onset, seed)
        for pattern in PATTERNS
        for budget in BUDGETS
        for onset in ONSETS
        for seed in range(
            first_seed,
            first_seed+count
        )
    ]


def fingerprint():
    hashes = e49.source_hashes()

    hashes[
        "tnse_exp50_v2.py"
    ] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()

    return hashes


def build_environment():
    global ADJ_DENSE
    global PHYSICAL
    global DEGREE

    ref.os.chdir(ROOT)

    graph = ref.mpc.build_comm_graph(
        n_nodes=118,
        seed=42
    )

    adj = ref.adjacency(graph)

    if hasattr(adj, "toarray"):
        ADJ_DENSE = np.asarray(
            adj.toarray(), dtype=float
        )
    else:
        ADJ_DENSE = np.asarray(
            adj, dtype=float
        )

    DEGREE = np.asarray(
        (ADJ_DENSE > 0).sum(axis=1),
        dtype=float
    ).reshape(-1)

    PHYSICAL = {
        pattern: np.asarray(
            ref.severity_from_pattern(
                pattern, 118
            ),
            dtype=float
        ).reshape(-1)
        for pattern in PATTERNS
    }

    for pattern, weights in PHYSICAL.items():
        if len(weights) != 118:
            raise ValueError(
                f"Invalid physical weights: {pattern}"
            )

    return graph, adj, PHYSICAL


def run(
    label, cases, alpha,
    methods, graph, adj,
    weights, hashes
):
    global ALPHA

    ALPHA = float(alpha)

    return e49.run_block(
        label,
        cases,
        0.0,
        0.0,
        list(methods),
        graph,
        adj,
        weights,
        hashes
    )


def cvar95(x):
    x = np.asarray(x, float)

    k = max(
        1,
        int(np.ceil(.05*len(x)))
    )

    return float(
        np.sort(x)[-k:].mean()
    )


def objective(x):
    x = np.asarray(x, float)

    return float(
        x.mean() + .20*cvar95(x)
    )


def freeze(path, record):
    content = json.dumps(
        record,
        indent=2,
        sort_keys=True
    )

    if path.exists():
        if json.loads(
            path.read_text()
        ) != record:
            raise RuntimeError(
                "Frozen protocol changed: "
                + str(path)
            )
    else:
        path.write_text(content)


def paired_statistics(df, events, dest):
    """
    Positive gain = B lower PAS than A.

    Bootstrap resamples complete paired
    cases within each predefined stratum.
    """

    keys = [
        "pattern",
        "budget",
        "onset",
        "seed"
    ]

    if df.duplicated(
        keys+["method"]
    ).any():
        raise RuntimeError(
            "Duplicate case-method records"
        )

    wide = df.pivot(
        index=keys,
        columns="method",
        values="PAS"
    )

    if wide.isna().any().any():
        raise RuntimeError(
            "Missing paired PAS values"
        )

    comparisons = [
        ("CDP", "FULL"),
        ("CDP", "STATE_LOCAL"),
        ("FULL", "STATE_LOCAL"),
        ("STATIC_EXTRA", "STATE_LOCAL"),
        ("NO_DYNAMIC", "FULL"),
        ("NO_GUARD", "FULL")
    ]

    rng = np.random.default_rng(
        20261023
    )

    rows = []
    boots_n = 2500

    for A, B in comparisons:
        delta = wide[A]-wide[B]

        blocks = [
            g.to_numpy(dtype=float)
            for _, g in delta.groupby(
                level=[
                    "pattern",
                    "budget",
                    "onset"
                ]
            )
        ]

        boot = np.zeros(boots_n)

        total = len(delta)

        for values in blocks:
            n = len(values)

            ix = rng.integers(
                0, n,
                size=(boots_n, n)
            )

            boot += (
                values[ix].mean(axis=1)
                * n / total
            )

        x = delta.to_numpy(dtype=float)

        changed = x[
            np.abs(x) > 1e-12
        ]

        if len(changed) >= 2:
            p = float(
                wilcoxon(changed).pvalue
            )
        else:
            p = 1.0

        rows.append({
            "comparison": f"{B} vs {A}",
            "n": len(x),
            "mean_gain": float(x.mean()),
            "CI_low": float(
                np.quantile(boot, .025)
            ),
            "CI_high": float(
                np.quantile(boot, .975)
            ),
            "p_unadjusted": p,
            "better_rate": float(
                (x > 1e-12).mean()
            ),
            "worse_rate": float(
                (x < -1e-12).mean()
            ),
            "tie_rate": float(
                (np.abs(x) <= 1e-12).mean()
            )
        })

    stats = pd.DataFrame(rows)

    # Holm adjustment across predeclared
    # pairwise comparisons.
    ps = stats[
        "p_unadjusted"
    ].to_numpy()

    rank = np.argsort(ps)
    m = len(ps)

    adj_sorted = np.maximum.accumulate(
        ps[rank] * (
            m-np.arange(m)
        )
    )

    adjusted = np.empty(m)

    adjusted[rank] = np.minimum(
        adj_sorted, 1.0
    )

    stats["holm_p"] = adjusted

    stats.to_csv(
        dest/"paired_statistics.csv",
        index=False
    )

    # Summary for every method.
    summary = []

    for method, g in df.groupby("method"):
        x = g["PAS"].to_numpy(float)

        summary.append({
            "method": method,
            "n": len(x),
            "mean_PAS": float(x.mean()),
            "P95_PAS": float(
                np.quantile(x, .95)
            ),
            "CVaR95_PAS": cvar95(x),
            "mean_switches": float(
                g["switch_count"].mean()
            )
        })

    pd.DataFrame(summary).to_csv(
        dest/"method_summary.csv",
        index=False
    )

    # Paired results by predefined conditions.
    subgroup = []

    for key, group in wide.groupby(
        level=[
            "pattern",
            "budget",
            "onset"
        ]
    ):
        for A, B in comparisons:

            x = (
                group[A]-group[B]
            ).to_numpy(float)

            subgroup.append({
                "pattern": key[0],
                "budget": key[1],
                "onset": key[2],
                "comparison": f"{B} vs {A}",
                "n": len(x),
                "mean_gain": float(x.mean()),
                "better": int(
                    (x > 1e-12).sum()
                ),
                "worse": int(
                    (x < -1e-12).sum()
                ),
                "tie": int(
                    (np.abs(x) <= 1e-12).sum()
                )
            })

    pd.DataFrame(subgroup).to_csv(
        dest/"all_predefined_subgroups.csv",
        index=False
    )

    # Action audit.
    if {
        "stage", "method", "protected_nodes"
    }.issubset(events.columns):

        first = events.loc[
            events.stage == 0
        ].pivot(
            index=keys,
            columns="method",
            values="protected_nodes"
        )

        action_rows = []

        for m in first.columns:

            if m == "FULL":
                continue

            action_rows.append({
                "method": m,
                "difference_from_FULL": float(
                    (
                        first[m].astype(str)
                        != first["FULL"].astype(str)
                    ).mean()
                )
            })

        pd.DataFrame(action_rows).to_csv(
            dest/"first_action_differences.csv",
            index=False
        )

    print("\n=== PAIRED RESULTS ===")
    print(
        stats.to_string(index=False),
        flush=True
    )

    print("\n=== ALL METHOD SUMMARY ===")
    print(
        pd.DataFrame(summary).to_string(
            index=False
        ),
        flush=True
    )

    return stats


def plot(stats, dest):
    fig, ax = plt.subplots(
        figsize=(10.5, 5.0)
    )

    colors = [
        "#2A7886" if v > 0
        else "#C26A61"
        for v in stats.mean_gain
    ]

    ys = np.arange(len(stats))

    for i, row in enumerate(stats.itertuples()):
        v = row.mean_gain

        ax.errorbar(
            v, i,
            xerr=[
                [max(0, v-row.CI_low)],
                [max(0, row.CI_high-v)]
            ],
            fmt="o",
            color=colors[i],
            ecolor=colors[i],
            markersize=5,
            capsize=3,
            linewidth=1.6
        )

    ax.set_yticks(
        ys,
        stats.comparison
    )

    ax.invert_yaxis()

    ax.axvline(
        0,
        color="#8394A1",
        linestyle="--",
        linewidth=.9
    )

    ax.set_xlabel(
        "Paired PAS reduction "
        "(positive favors first-named method)"
    )

    ax.set_title(
        "Frozen holdout | "
        "paired component effects",
        loc="left",
        pad=16
    )

    ax.grid(
        axis="x",
        alpha=.16
    )

    ax.set_axisbelow(True)

    fig.subplots_adjust(
        left=.29,
        right=.96,
        top=.86,
        bottom=.18
    )

    for ext in ["pdf", "svg", "png"]:
        fig.savefig(
            dest/f"Exp50_effect_forest.{ext}",
            dpi=260,
            bbox_inches="tight",
            pad_inches=.18
        )

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        required=True,
        choices=[
            "pilot",
            "validate",
            "test"
        ]
    )

    parser.add_argument(
        "--n-val",
        type=int,
        default=6
    )

    parser.add_argument(
        "--n-test",
        type=int,
        default=25
    )

    args = parser.parse_args()

    # Ensure Exp49 is available and unchanged.
    locked49 = (
        EXP/"results"/
        "49_split_confirm_revision"/
        "LOCKED_HYPERPARAMETERS.json"
    )

    if not locked49.exists():
        raise RuntimeError(
            "Exp49 locked parameter file missing"
        )

    original = json.loads(
        locked49.read_text()
    )

    if not (
        float(original["gamma"]) == 0
        and
        float(original["kappa"]) == 0
    ):
        raise RuntimeError(
            "Exp49 configuration differs"
        )

    if args.n_val < 3 or args.n_test < 5:
        raise ValueError(
            "Too few seeds per condition"
        )

    graph, adj, weights = build_environment()
    hashes = fingerprint()

    if args.phase == "pilot":
        cases = [
            ("DDAC", 20, 90, 7901),
            ("RC", 40, 210, 7902),
            ("DDAC", 60, 210, 7903)
        ]

        df, ev = run(
            "pilot",
            cases,
            .50,
            METHODS,
            graph, adj,
            weights, hashes
        )

        print(
            "\nPILOT PASSED",
            flush=True
        )
        print(
            df.groupby("method")["PAS"]
            .mean().to_string(),
            flush=True
        )
        return

    if args.phase == "validate":
        cases = make_cases(
            VAL_SEED,
            args.n_val
        )

        scores = []

        for alpha in ALPHAS:
            df, ev = run(
                f"validation_alpha_{alpha:.2f}",
                cases,
                alpha,
                [
                    "CDP",
                    "FULL",
                    "STATE_LOCAL",
                    "STATIC_EXTRA"
                ],
                graph, adj,
                weights, hashes
            )

            result = {
                m: objective(
                    df.loc[
                        df.method == m,
                        "PAS"
                    ].to_numpy()
                )
                for m in [
                    "FULL",
                    "STATE_LOCAL",
                    "STATIC_EXTRA"
                ]
            }

            scores.append({
                "alpha": alpha,
                "state_local_score": result[
                    "STATE_LOCAL"
                ],
                "full_score": result["FULL"],
                "static_score": result[
                    "STATIC_EXTRA"
                ]
            })

        table = pd.DataFrame(
            scores
        ).sort_values(
            ["state_local_score", "alpha"]
        )

        table.to_csv(
            OUT/"validation_scores.csv",
            index=False
        )

        winner = table.iloc[0]

        record = {
            "alpha": float(winner.alpha),
            "n_val": args.n_val,
            "n_test": args.n_test,
            "source_hashes": hashes,
            "criterion": (
                "validation mean PAS "
                "+ 0.20 CVaR95"
            ),
            "methods": METHODS,
            "test_seed_first": TEST_SEED
        }

        freeze(
            OUT/"LOCKED_ALPHA.json",
            record
        )

        print(
            "\nVALIDATION RESULTS",
            flush=True
        )
        print(
            table.to_string(index=False),
            flush=True
        )
        print(
            "\nLOCKED:",
            json.dumps(
                record, indent=2
            ),
            flush=True
        )
        return

    # test
    lock = OUT/"LOCKED_ALPHA.json"

    if not lock.exists():
        raise RuntimeError(
            "Run validation before test"
        )

    record = json.loads(
        lock.read_text()
    )

    if (
        record["source_hashes"] != hashes
        or record["n_val"] != args.n_val
        or record["n_test"] != args.n_test
    ):
        raise RuntimeError(
            "Code or split changed after tuning"
        )

    cases = make_cases(
        TEST_SEED,
        args.n_test
    )

    df, ev = run(
        "test",
        cases,
        record["alpha"],
        METHODS,
        graph, adj,
        weights, hashes
    )

    dest = OUT/"test"

    stats = paired_statistics(
        df, ev, dest
    )

    plot(stats, dest)

    print(
        "\nEXP50 V2 COMPLETED",
        flush=True
    )
    print(
        "RESULT DIRECTORY:",
        dest,
        flush=True
    )


if __name__ == "__main__":
    main()
