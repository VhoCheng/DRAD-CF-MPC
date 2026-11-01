from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "revision_experiments"
OUT = EXP / "results" / "47_independent_closed_loop"
OUT.mkdir(parents=True, exist_ok=True)

os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

import drad_mpc as mpc
from seid_malware import simulate_seid_once

METHODS = [
    "CDP",
    "LEGACY_EP",
    "FIXED_EP",
    "FIXED_EP_NO_DYNAMIC",
    "FIXED_EP_NO_GUARD",
    "FIXED_EP_HEURISTIC_GUARD",
    "FIXED_EP_HEURISTIC_NO_GUARD",
    "FIXED_EP_SPLIT_GUARD",
]

PATTERNS = ["DDAC", "RC"]
BUDGETS = [20, 40]
ONSETS = [90, 210]

K_TRAIN = 8
K_GUARD = 24
GAMMA = 0.10
KAPPA = 0.50
STAGES = 3
H_PREVIEW_MAX = 32


def stable_seed(*items):
    s = "|".join(str(x) for x in items)
    digest = hashlib.blake2b(
        s.encode(), digest_size=8
    ).digest()
    return int.from_bytes(
        digest, "little"
    ) % (2**32 - 1)


def adjacency(graph):
    n = graph.number_of_nodes()
    edges = list(graph.edges())

    ri = np.array(
        [int(i) for i, j in edges]
        + [int(j) for i, j in edges],
        dtype=int
    )

    ci = np.array(
        [int(j) for i, j in edges]
        + [int(i) for i, j in edges],
        dtype=int
    )

    return csr_matrix(
        (
            np.ones(len(ri), dtype=np.int16),
            (ri, ci)
        ),
        shape=(n, n)
    )


def propagate(state, protected, tape, adj):
    state = state.copy()
    n = len(state)

    factor = np.ones(n)
    factor[np.asarray(
        protected, dtype=int
    )] = 0.2

    for u in tape:
        if np.any(state == 3):
            break

        neighbors = adj @ (
            state == 2
        ).astype(np.int16)

        p = 1.0 - np.power(
            1.0 - 0.01 * factor,
            neighbors
        )

        nxt = state.copy()

        nxt[
            (state == 0) & (u[0] < p)
        ] = 1

        nxt[
            (state == 1) & (u[1] < 0.15)
        ] = 2

        nxt[
            (state == 2) & (u[2] < 0.00001)
        ] = 3

        state = nxt

    return state


def severity_from_pattern(pattern, n):
    table = mpc.compute_physics_impact_table(
        pattern=pattern,
        table_dir="outputs/tables"
    )

    weight = np.zeros(n)

    for node, group in table.groupby(
        "comm_node"
    ):
        j = int(node)

        if 0 <= j < n:
            weight[j] += float(
                group["impact_score_raw"].sum()
            )

    if weight.sum() <= 0:
        raise ValueError(
            f"No valid coupling weights: {pattern}"
        )

    return weight / weight.sum()


def pas(state, weight):
    if np.any(state == 3):
        return 0.0

    return float(
        weight[state == 2].sum()
    )


def tapes(seeds, steps, n):
    return [
        np.random.default_rng(s).random(
            (steps, 3, n)
        )
        for s in seeds
    ]


def predict(
    state, nodes, ts,
    adj, weight, budget, old_ep
):
    losses = []
    successes = []

    for tape in ts:
        result = propagate(
            state, nodes, tape, adj
        )

        losses.append(
            pas(result, weight)
        )

        successes.append(
            not np.any(result == 3)
        )

    a = np.asarray(
        losses, dtype=float
    )

    mean = float(a.mean())
    ps = float(np.mean(successes))

    # Legacy code double-discounts success.
    # Corrected EP uses unconditional PAS.
    ep = mean * ps if old_ep else mean

    rt = (
        1.0 + 20.0 * mean
        * (1.0 - 0.35 * budget / 100.0)
    )

    ri = np.clip(
        1.0 - 0.65 * mean
        + 0.1 * budget / 100.0,
        0, 1
    )

    objective = (
        0.68 * ep
        + 0.24 * mean
        + 0.06 * rt / 21.0
        - 0.02 * ri
    )

    return {
        "objective": float(objective),
        "samples": a,
        "mean": mean,
        "ps": ps,
        "ep": ep
    }


def candidate_nodes(
    pattern, budget, state,
    info, seed, dynamic
):
    active = (
        np.where(state == 2)[0]
        .astype(int).tolist()
        if dynamic else []
    )

    exposed = (
        np.where(state == 1)[0]
        .astype(int).tolist()
        if dynamic else []
    )

    candidates = mpc.generate_candidate_policies(
        pattern=pattern,
        defense_budget=budget,
        attack_information_ratio=info,
        active_nodes=active,
        exposed_nodes=exposed,
        table_dir="outputs/tables",
        seed=seed
    )

    if "CDP" not in candidates:
        raise RuntimeError(
            "CDP missing from candidate pool"
        )

    out = {
        k: tuple(sorted(
            set(map(int, v))
        ))
        for k, v in candidates.items()
    }

    if any(
        len(v) != budget
        for v in out.values()
    ):
        raise ValueError(
            "Candidate budget mismatch"
        )

    return out


def choose(
    method, state, pattern,
    budget, info, seed,
    train, guard, adj, weight
):
    dynamic = (
        method != "FIXED_EP_NO_DYNAMIC"
    )

    candidates = candidate_nodes(
        pattern, budget, state,
        info, seed, dynamic
    )

    if method == "CDP":
        return (
            candidates["CDP"],
            "CDP", "CDP",
            False, len(candidates)
        )

    old_ep = method == "LEGACY_EP"

    heuristic = method in (
        "FIXED_EP_HEURISTIC_GUARD",
        "FIXED_EP_HEURISTIC_NO_GUARD"
    )

    if heuristic:
        active = (
            np.where(state == 2)[0]
            .astype(int).tolist()
        )

        exposed = (
            np.where(state == 1)[0]
            .astype(int).tolist()
        )

        heur = tuple(sorted(
            mpc.candidate_drad_score_nodes(
                pattern=pattern,
                defense_budget=budget,
                attack_information_ratio=info,
                active_nodes=active,
                exposed_nodes=exposed,
                table_dir="outputs/tables",
                seed=seed
            )
        ))

        if len(heur) != budget:
            raise ValueError(
                "Heuristic budget mismatch"
            )

        candidates["HEURISTIC"] = heur
        chosen = "HEURISTIC"
        needed = ["CDP", "HEURISTIC"]

    else:
        needed = list(candidates)
        chosen = None

    metrics = {
        k: predict(
            state, candidates[k],
            train, adj, weight,
            budget, old_ep
        )
        for k in needed
    }

    if chosen is None:
        chosen = min(
            metrics,
            key=lambda k:
                metrics[k]["objective"]
        )

    if method in (
        "FIXED_EP_NO_GUARD",
        "FIXED_EP_HEURISTIC_NO_GUARD"
    ):
        accepted = chosen != "CDP"
        return (
            candidates[chosen],
            chosen, chosen,
            accepted, len(candidates)
        )

    if chosen == "CDP":
        return (
            candidates["CDP"],
            "CDP", chosen,
            False, len(candidates)
        )

    m = metrics

    # Independent samples for guard verification.
    if method == "FIXED_EP_SPLIT_GUARD":
        m = {
            k: predict(
                state, candidates[k],
                guard, adj, weight,
                budget, False
            )
            for k in ("CDP", chosen)
        }

    cdp_m = m["CDP"]
    cand_m = m[chosen]

    relative = (
        cdp_m["objective"]
        - cand_m["objective"]
    ) / (
        abs(cdp_m["objective"]) + 1e-9
    )

    dif = (
        cdp_m["samples"]
        - cand_m["samples"]
    )

    se = (
        float(
            np.std(dif, ddof=1)
            / np.sqrt(len(dif))
        )
        if len(dif) > 1
        else 0.0
    )

    accept = (
        relative >= GAMMA
        and float(dif.mean()) - KAPPA * se > 0
    )

    return (
        candidates[chosen]
        if accept else candidates["CDP"],

        chosen if accept else "CDP",
        chosen,
        bool(accept),
        len(candidates)
    )


def state_from_preview(preview, n):
    state = np.zeros(
        n, dtype=np.int8
    )

    for key, code in [
        ("exposed_nodes", 1),
        ("active_infectious_nodes", 2),
        ("detected_nodes", 3)
    ]:
        state[
            np.asarray(
                preview[key], dtype=int
            )
        ] = code

    return state


def run_one(case, graph, adj, weights):
    pattern, budget, onset, seed = case

    n = graph.number_of_nodes()
    info = 0.5

    start_time = max(
        1, int(onset * 0.4)
    )

    preview = simulate_seid_once(
        Gc=graph,
        onset_time=start_time,
        infection_rate=0.01,
        activation_rate=0.15,
        detection_rate=0.00001,
        attack_information_ratio=info,
        protected_nodes=None,
        strict_detection_failure=False,
        seed=stable_seed(
            "preview", *case
        )
    )

    initial = state_from_preview(
        preview, n
    )

    remaining = onset - start_time

    durations = [
        remaining // STAGES
        + (j < remaining % STAGES)
        for j in range(STAGES)
    ]

    entry = dict(
        pattern=pattern,
        budget=budget,
        onset=onset,
        seed=seed,
        initial_S=int(np.sum(initial == 0)),
        initial_E=int(np.sum(initial == 1)),
        initial_I=int(np.sum(initial == 2)),
        initial_D=int(np.sum(initial == 3))
    )

    results = []
    events = []

    for method in METHODS:
        state = initial.copy()
        switches = 0
        first_action = []

        for stage, duration in enumerate(
            durations
        ):
            steps = min(
                H_PREVIEW_MAX, duration
            )

            tr = tapes(
                [
                    stable_seed(
                        "train", *case, stage, k
                    )
                    for k in range(K_TRAIN)
                ],
                steps, n
            )

            gd = tapes(
                [
                    stable_seed(
                        "guard", *case, stage, k
                    )
                    for k in range(K_GUARD)
                ],
                steps, n
            )

            chosen, policy, before, switched, nc = choose(
                method,
                state,
                pattern,
                budget,
                info,
                stable_seed(
                    "candidate", *case, stage
                ),
                tr,
                gd,
                adj,
                weights[pattern]
            )

            if stage == 0:
                first_action = list(chosen)

            switches += int(switched)

            events.append({
                **entry,
                "method": method,
                "stage": stage,
                "active_before": int(
                    np.sum(state == 2)
                ),
                "exposed_before": int(
                    np.sum(state == 1)
                ),
                "chosen": policy,
                "best_before_guard": before,
                "switched": int(switched),
                "candidate_count": nc,
                "protected_nodes": json.dumps(
                    chosen
                ),
                "first_state_sha": hashlib.sha256(
                    initial.tobytes()
                ).hexdigest()
            })

            outer = np.random.default_rng(
                stable_seed(
                    "outer", *case, stage
                )
            ).random(
                (duration, 3, n)
            )

            state = propagate(
                state, chosen, outer, adj
            )

        results.append({
            **entry,
            "method": method,
            "PAS": pas(
                state, weights[pattern]
            ),
            "attack_failed": int(
                np.any(state == 3)
            ),
            "switch_count": switches,
            "terminal_I": int(
                np.sum(state == 2)
            ),
            "first_protected_nodes": json.dumps(
                first_action
            )
        })

    return results, events


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)

    out = np.empty(len(p))

    out[order] = np.minimum(
        1,
        np.maximum.accumulate(
            p[order] * (
                len(p) - np.arange(len(p))
            )
        )
    )

    return out


def summarize(df):
    keys = [
        "pattern", "budget",
        "onset", "seed"
    ]

    wide = df.pivot(
        index=keys,
        columns="method",
        values="PAS"
    )

    if wide[METHODS].isna().any().any():
        raise RuntimeError(
            "Missing paired methods"
        )

    rng = np.random.default_rng(
        20261022
    )

    pairs = [
        ("FIXED_EP", "CDP"),
        ("FIXED_EP", "LEGACY_EP"),
        ("FIXED_EP", "FIXED_EP_NO_DYNAMIC"),
        ("FIXED_EP", "FIXED_EP_NO_GUARD"),
        ("FIXED_EP", "FIXED_EP_HEURISTIC_GUARD"),
        ("FIXED_EP_SPLIT_GUARD", "FIXED_EP"),
        (
            "FIXED_EP_NO_GUARD",
            "FIXED_EP_HEURISTIC_NO_GUARD"
        ),
        (
            "FIXED_EP_HEURISTIC_GUARD",
            "FIXED_EP_HEURISTIC_NO_GUARD"
        )
    ]

    rows = []

    for a, b in pairs:
        # Positive gain means A has lower PAS.
        x = (
            wide[b] - wide[a]
        ).to_numpy(dtype=float)

        boots = rng.choice(
            x,
            (3000, len(x)),
            replace=True
        ).mean(axis=1)

        nonzero = x[
            np.abs(x) > 1e-12
        ]

        p = (
            1.0 if len(nonzero) == 0
            else float(
                wilcoxon(nonzero).pvalue
            )
        )

        rows.append({
            "A": a,
            "B": b,
            "n_scenarios": len(x),
            "mean_PAS_A": float(
                wide[a].mean()
            ),
            "mean_PAS_B": float(
                wide[b].mean()
            ),
            "gain_B_minus_A": float(
                x.mean()
            ),
            "CI_low": float(
                np.quantile(boots, 0.025)
            ),
            "CI_high": float(
                np.quantile(boots, 0.975)
            ),
            "A_worse_than_B_rate": float(
                np.mean(x < 0)
            ),
            "wilcoxon_p": p
        })

    stats = pd.DataFrame(rows)

    stats["holm_p_8"] = holm(
        stats["wilcoxon_p"]
    )

    return stats


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["pilot", "full"],
        required=True
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=50
    )

    args = parser.parse_args()

    src_hashes = {
        name: hashlib.sha256(
            (ROOT / "src" / name).read_bytes()
        ).hexdigest()
        for name in [
            "drad_mpc.py",
            "seid_malware.py",
            "run_experiments_core.py"
        ]
    }

    protocol = {
        "methods": METHODS,
        "patterns": PATTERNS,
        "budgets": BUDGETS,
        "onsets": ONSETS,
        "seeds": args.seeds,
        "K_train": K_TRAIN,
        "K_guard": K_GUARD,
        "stages": STAGES,
        "max_preview_steps": H_PREVIEW_MAX,
        "gamma": GAMMA,
        "kappa": KAPPA,
        "source_hashes": src_hashes,
        "note": (
            "New common-state rolling-decision "
            "validation, not the original controller."
        )
    }

    manifest = OUT / (
        args.mode + "_protocol.json"
    )

    if manifest.exists():
        old = json.loads(
            manifest.read_text()
        )

        if old != protocol:
            raise RuntimeError(
                "Existing protocol differs; "
                "do not mix experimental configurations."
            )

    manifest.write_text(
        json.dumps(
            protocol, indent=2
        )
    )

    graph = mpc.build_comm_graph(
        n_nodes=118,
        seed=42
    )

    if sorted(graph.nodes()) != list(range(118)):
        raise ValueError(
            "Unexpected communication node IDs"
        )

    adj = adjacency(graph)

    weights = {
        p: severity_from_pattern(p, 118)
        for p in PATTERNS
    }

    cases = [
        (pat, b, onset, s)
        for pat in PATTERNS
        for b in BUDGETS
        for onset in ONSETS
        for s in range(args.seeds)
    ]

    if args.mode == "pilot":
        cases = [
            cases[0],
            cases[-1]
        ]

    dest = OUT / args.mode
    dest.mkdir(exist_ok=True)

    checkpoints = dest / "checkpoints"
    checkpoints.mkdir(exist_ok=True)

    t0 = time.time()

    for i, case in enumerate(cases, 1):
        name = "%s_b%s_t%s_s%s.json" % case
        path = checkpoints / name

        if not path.exists():
            results, events = run_one(
                case, graph, adj, weights
            )

            path.write_text(
                json.dumps({
                    "results": results,
                    "events": events
                })
            )

        print(
            f"[{i}/{len(cases)}] {case} "
            f"completed; elapsed={time.time()-t0:.1f}s",
            flush=True
        )

    all_results = []
    all_events = []

    for case in cases:
        name = "%s_b%s_t%s_s%s.json" % case

        data = json.loads(
            (checkpoints / name).read_text()
        )

        all_results.extend(
            data["results"]
        )

        all_events.extend(
            data["events"]
        )

    df = pd.DataFrame(all_results)
    ev = pd.DataFrame(all_events)

    assert len(df) == len(cases) * len(METHODS)
    assert len(ev) == len(df) * STAGES

    assert not df.duplicated(
        [
            "pattern", "budget",
            "onset", "seed", "method"
        ]
    ).any()

    df.to_csv(
        dest / "paired_outcomes.csv",
        index=False
    )

    ev.to_csv(
        dest / "policy_decisions.csv",
        index=False
    )

    stats = summarize(df)

    stats.to_csv(
        dest / "paired_statistics.csv",
        index=False
    )

    df.groupby(
        [
            "pattern", "budget",
            "onset", "method"
        ],
        as_index=False
    ).agg(
        mean_PAS=("PAS", "mean"),
        n_scenarios=("PAS", "size"),
        failed_fraction=(
            "attack_failed", "mean"
        ),
        average_switches=(
            "switch_count", "mean"
        )
    ).to_csv(
        dest / "by_setting_summary.csv",
        index=False
    )

    first = ev[
        ev["stage"] == 0
    ]

    action = first.pivot(
        index=[
            "pattern", "budget",
            "onset", "seed"
        ],
        columns="method",
        values="protected_nodes"
    )

    divergence = []

    for method in METHODS:
        if method == "FIXED_EP":
            continue

        divergence.append({
            "method": method,
            "first_action_difference_from_FIXED_EP":
                float(
                    (
                        action[method]
                        != action["FIXED_EP"]
                    ).mean()
                ),
            "n_scenarios": len(action)
        })

    pd.DataFrame(divergence).to_csv(
        dest / "first_action_divergence.csv",
        index=False
    )

    print(
        "\n=== CLOSED-LOOP RESULTS ==="
    )

    print(
        df.groupby("method").agg(
            mean_PAS=("PAS", "mean"),
            failure_rate=(
                "attack_failed", "mean"
            ),
            mean_switches=(
                "switch_count", "mean"
            )
        ).to_string()
    )

    print(
        "\n=== EIGHT PLANNED PAIRED COMPARISONS ==="
    )

    print(stats.to_string(index=False))

    print(
        "\nEXP47",
        args.mode.upper(),
        "COMPLETED"
    )

    print("OUTPUT:", dest)


if __name__ == "__main__":
    main()
