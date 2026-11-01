from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "revision_experiments"
RESULTS = EXP / "results"
HISTORY = RESULTS / "26_multiscale_full/per_run.csv"
NETWORKS = RESULTS / "25_multiscale_networks"

sys.path.insert(0, str(ROOT / "src"))
import seid_malware as seid
import drad_mpc as mpc
import run_experiments_core as core
import dynamic_risk_defense as risk

spec = importlib.util.spec_from_file_location(
    "tnse_exp38_reference",
    EXP / "38_same_state_validation.py"
)
ref = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ref
spec.loader.exec_module(ref)

METHODS = [
    "CDP", "FULL", "NO_GUARD",
    "NO_DYNAMIC", "NO_CF"
]
CONDITIONS = [
    "clean", "delay_10", "delay_30",
    "loss_20", "false_positive_05",
    "combined_delay30_loss20"
]
COMPARISONS = [
    ("FULL", "CDP"),
    ("FULL", "NO_GUARD"),
    ("FULL", "NO_DYNAMIC"),
    ("FULL", "NO_CF")
]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def to_state(preview, n):
    state = np.zeros(n, dtype=np.int8)
    for key, code in (
        ("exposed_nodes", seid.EXPOSED),
        ("active_infectious_nodes", seid.INFECTIOUS),
        ("detected_nodes", seid.DETECTED)
    ):
        state[np.asarray(preview[key], dtype=int)] = code
    return state

def make_observation(
    true_state, graph, info, seed, t_preview, condition
):
    if condition in ("delay_10", "delay_30",
                     "combined_delay30_loss20"):
        delay = 10 if condition == "delay_10" else 30
        old_t = max(1, t_preview - delay)
        old = seid.simulate_seid_once(
            Gc=graph,
            onset_time=old_t,
            infection_rate=0.01,
            activation_rate=0.15,
            detection_rate=0.00001,
            attack_information_ratio=info,
            protected_nodes=None,
            strict_detection_failure=False,
            seed=seed
        )
        observed = to_state(old, len(true_state))
    else:
        observed = true_state.copy()

    rng = np.random.default_rng(
        seed + 40000000 + CONDITIONS.index(condition) * 10007
    )

    if condition in ("loss_20", "combined_delay30_loss20"):
        detected = np.flatnonzero(
            (observed == seid.EXPOSED) |
            (observed == seid.INFECTIOUS)
        )
        missing = rng.random(len(detected)) < 0.20
        observed[detected[missing]] = seid.SUSCEPTIBLE

    if condition == "false_positive_05":
        susceptible = np.flatnonzero(
            observed == seid.SUSCEPTIBLE
        )
        fake = rng.random(len(susceptible)) < 0.05
        observed[susceptible[fake]] = seid.INFECTIOUS

    return observed

def worker(task):
    sid, data, n_runs, k, eval_runs = task
    frame = pd.DataFrame(data)

    n = int(frame["case_size"].iloc[0])
    pattern = str(frame["pattern"].iloc[0])
    info = float(frame["info"].iloc[0])
    budget = int(frame["budget"].iloc[0])
    onset = int(frame["onset"].iloc[0])

    case_dir = NETWORKS / f"case{n}"
    table_dir = case_dir / "outputs/tables"
    if not table_dir.exists():
        raise FileNotFoundError(table_dir)

    graph = nx.barabasi_albert_graph(
        n=n, m=2, seed=42
    )
    adj = nx.to_scipy_sparse_array(
        graph,
        nodelist=list(range(n)),
        format="csr",
        dtype=np.int16
    )

    old_cwd = Path.cwd()
    old_builder = risk.build_comm_graph
    rows = []

    try:
        os.chdir(case_dir)
        risk.build_comm_graph = (
            lambda n_nodes=118, seed=42: graph
        )

        impact = mpc.compute_physics_impact_table(
            pattern, table_dir="outputs/tables"
        )
        weights = np.zeros(n, dtype=float)
        for _, r in impact.iterrows():
            weights[int(r["comm_node"])] = float(
                r["impact_score_raw"]
            )
        denominator = float(weights.sum()) + 1e-9
        if denominator <= 1e-9:
            raise RuntimeError("Nonpositive physical-impact weights")

        for run_index, group in frame.groupby("run_index"):
            if int(run_index) >= n_runs:
                continue
            seeds = group["seed"].unique()
            if len(seeds) != 1:
                raise RuntimeError("Inconsistent historical seed")
            seed = int(seeds[0])

            preview = core.get_preview_state(
                Gc=graph,
                onset_time=onset,
                attack_information_ratio=info,
                infection_rate=0.01,
                activation_rate=0.15,
                detection_rate=0.00001,
                seed=seed
            )
            true_state = to_state(preview, n)
            t_preview = max(1, int(onset * 0.4))
            remaining = onset - t_preview

            for condition in CONDITIONS:
                obs = make_observation(
                    true_state, graph, info, seed,
                    t_preview, condition
                )
                obs_I = np.flatnonzero(
                    obs == seid.INFECTIOUS
                ).tolist()
                obs_E = np.flatnonzero(
                    obs == seid.EXPOSED
                ).tolist()

                dynamic = mpc.generate_candidate_policies(
                    pattern=pattern,
                    defense_budget=budget,
                    attack_information_ratio=info,
                    active_nodes=obs_I,
                    exposed_nodes=obs_E,
                    table_dir="outputs/tables",
                    seed=seed
                )

                static = mpc.generate_candidate_policies(
                    pattern=pattern,
                    defense_budget=budget,
                    attack_information_ratio=info,
                    active_nodes=[],
                    exposed_nodes=[],
                    table_dir="outputs/tables",
                    seed=seed
                )

                cdp = tuple(sorted(
                    int(x) for x in dynamic["CDP"]
                ))

                drad = tuple(sorted(
                    int(x) for x in
                    mpc.candidate_drad_score_nodes(
                        pattern, budget, info,
                        obs_I, obs_E,
                        "outputs/tables", seed
                    )
                ))

                frozen_drad = tuple(sorted(
                    int(x) for x in
                    mpc.candidate_drad_score_nodes(
                        pattern, budget, info,
                        [], [],
                        "outputs/tables", seed
                    )
                ))

                dynamic = {
                    key: tuple(sorted(int(x) for x in nodes))
                    for key, nodes in dynamic.items()
                }
                dynamic["CDP"] = cdp
                dynamic["DRAD_SCORE"] = drad

                frozen = dict(dynamic)
                frozen["DRAD_SCORE"] = frozen_drad

                if "RANK_ENSEMBLE" in static:
                    frozen["RANK_ENSEMBLE"] = tuple(sorted(
                        int(x)
                        for x in static["RANK_ENSEMBLE"]
                    ))
                elif "RANK_ENSEMBLE" in frozen:
                    del frozen["RANK_ENSEMBLE"]

                estimated = np.zeros(n, dtype=np.int8)
                estimated[obs == seid.EXPOSED] = seid.EXPOSED
                estimated[obs == seid.INFECTIOUS] = seid.INFECTIOUS
                estimated[obs == seid.DETECTED] = seid.DETECTED

                select_seeds = [
                    seed + 60000000
                    + CONDITIONS.index(condition) * 100003
                    + j * 1013
                    for j in range(k)
                ]

                cache = {}

                def predict(nodes):
                    if nodes in cache:
                        return cache[nodes]

                    samples = []
                    failures = []

                    for trial_seed in select_seeds:
                        end, failed = ref.advance(
                            estimated, adj, nodes,
                            trial_seed, remaining
                        )
                        infectious = np.flatnonzero(
                            end == seid.INFECTIOUS
                        )
                        pas = (
                            0.0 if failed else
                            float(weights[infectious].sum()
                                  / denominator)
                        )
                        samples.append(pas)
                        failures.append(int(failed))

                    mean_pas = float(np.mean(samples))
                    ps = 1.0 - float(np.mean(failures))
                    ep = mean_pas * ps

                    rt_vals = []
                    ri_vals = []
                    for pas in samples:
                        rt, ri, _ = (
                            mpc.evaluate_resilience_proxy_mpc(
                                pas, budget, max_budget=100
                            )
                        )
                        rt_vals.append(rt)
                        ri_vals.append(ri)

                    objective = (
                        0.68 * ep
                        + 0.24 * mean_pas
                        + 0.06 * float(np.mean(rt_vals)) / 21.0
                        - 0.02 * float(np.mean(ri_vals))
                    )

                    cache[nodes] = (
                        float(objective),
                        np.asarray(samples, dtype=float)
                    )
                    return cache[nodes]

                def choose(pool, guarded):
                    scores = {
                        name: predict(nodes)[0]
                        for name, nodes in pool.items()
                    }
                    best = min(scores, key=scores.get)
                    best_nodes = pool[best]
                    if not guarded or best_nodes == cdp:
                        return best_nodes

                    base_score, base_samples = predict(cdp)
                    best_score, best_samples = predict(best_nodes)

                    delta = (
                        (base_score - best_score)
                        / max(abs(base_score), 1e-9)
                    )
                    differences = base_samples - best_samples
                    stderr = (
                        float(np.std(differences, ddof=1)
                              / np.sqrt(len(differences)))
                        if len(differences) > 1 else 0.0
                    )
                    conservative = (
                        float(np.mean(differences))
                        - 0.5 * stderr
                    )
                    return (
                        best_nodes
                        if delta >= 0.10 and conservative > 0
                        else cdp
                    )

                choices = {
                    "CDP": cdp,
                    "FULL": choose(dynamic, True),
                    "NO_GUARD": choose(dynamic, False),
                    "NO_DYNAMIC": choose(frozen, True),
                    "NO_CF": drad
                }

                validation_cache = {}
                for method, nodes in choices.items():
                    if len(nodes) != budget:
                        raise RuntimeError(
                            f"Invalid budget {sid}/{method}"
                        )

                    if nodes not in validation_cache:
                        values = []
                        failed_values = []
                        for j in range(eval_runs):
                            test_seed = (
                                seed + 90000000
                                + CONDITIONS.index(condition) * 300007
                                + j * 7001
                            )
                            end, failed = ref.advance(
                                true_state, adj, nodes,
                                test_seed, remaining
                            )
                            infectious = np.flatnonzero(
                                end == seid.INFECTIOUS
                            )
                            value = (
                                0.0 if failed else
                                float(weights[infectious].sum()
                                      / denominator)
                            )
                            values.append(value)
                            failed_values.append(int(failed))
                        validation_cache[nodes] = (
                            float(np.mean(values)),
                            float(np.mean(failed_values))
                        )

                    mean_pas, fail_rate = validation_cache[nodes]
                    rows.append({
                        "scenario_id": int(sid),
                        "case_size": int(n),
                        "pattern": pattern,
                        "info": info,
                        "budget": budget,
                        "onset": onset,
                        "run_index": int(run_index),
                        "condition": condition,
                        "method": method,
                        "PAS": mean_pas,
                        "attack_failure_rate": fail_rate,
                        "loss_above_030": int(mean_pas > 0.30),
                        "switched_from_CDP": int(nodes != cdp),
                        "preview_I": int(
                            (true_state == seid.INFECTIOUS).sum()
                        ),
                        "observed_I": len(obs_I),
                        "protected_nodes": list(nodes),
                        "selection_rollouts": k,
                        "independent_eval_runs": eval_runs
                    })
    finally:
        risk.build_comm_graph = old_builder
        os.chdir(old_cwd)

    return {
        "scenario_id": int(sid),
        "records": rows
    }

def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    sorted_adj = np.minimum(
        1.0,
        np.maximum.accumulate(
            p[order] * (len(p) - np.arange(len(p)))
        )
    )
    out = np.empty(len(p), dtype=float)
    out[order] = sorted_adj
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"],
                    required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if not HISTORY.exists():
        raise FileNotFoundError(HISTORY)

    history = pd.read_csv(HISTORY)
    if len(history) != 960:
        raise RuntimeError("Expected 960 historical records")

    tasks = [
        (int(sid), block.to_dict("records"),
         1 if args.mode == "pilot" else 4,
         3 if args.mode == "pilot" else 5,
         2 if args.mode == "pilot" else 3)
        for sid, block in history.groupby(
            "scenario_id", sort=True
        )
    ]
    if len(tasks) != 80:
        raise RuntimeError("Expected 80 scenarios")

    if args.mode == "pilot":
        one_per_size = {}
        for t in tasks:
            n = int(t[1][0]["case_size"])
            one_per_size.setdefault(n, t)
        tasks = list(one_per_size.values())

    run_dir = RESULTS / f"39_mechanism_{args.mode}"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    config = {
        "mode": args.mode,
        "methods": METHODS,
        "conditions": CONDITIONS,
        "scenario_ids": [t[0] for t in tasks],
        "history_sha256": digest(HISTORY),
        "exp38_sha256": digest(EXP / "38_same_state_validation.py"),
        "source_hashes": {
            name: digest(ROOT / "src" / name)
            for name in (
                "seid_malware.py",
                "drad_mpc.py",
                "dynamic_risk_defense.py",
                "run_experiments_core.py"
            )
        },
        "driver_sha256": digest(Path(__file__).resolve()),
        "protocol": (
            "New common-preview post-observation experiment. "
            "K=5 prediction rollouts and independent K=3 "
            "evaluation for full; original objective and "
            "gamma=0.1,kappa=0.5. Stale observation measured "
            "in simulated SEID steps. Not a reproduction of "
            "the original pre-infection intervention."
        )
    }

    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        if json.loads(
            manifest_path.read_text(encoding="utf-8")
        ) != config:
            raise RuntimeError("Protocol hash changed")
    else:
        manifest_path.write_text(
            json.dumps(config, indent=2),
            encoding="utf-8"
        )

    pending = [
        t for t in tasks
        if not (
            checkpoints / f"scenario_{t[0]:03d}.json"
        ).exists()
    ]
    print("EXPERIMENT 39", args.mode,
          "scenarios", len(tasks),
          "pending", len(pending), flush=True)

    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(tasks), 8)
    ) as executor:
        future_map = {
            executor.submit(worker, t): t
            for t in pending
        }
        done = len(tasks) - len(pending)

        for future in as_completed(future_map):
            task = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"Scenario {task[0]} failed"
                ) from exc

            expected = (
                task[2] * len(METHODS) * len(CONDITIONS)
            )
            if len(result["records"]) != expected:
                raise RuntimeError(
                    f"Unexpected row count for scenario {task[0]}"
                )

            dest = checkpoints / (
                f"scenario_{task[0]:03d}.json"
            )
            temp = dest.with_suffix(f".{os.getpid()}.tmp")
            temp.write_text(
                json.dumps(result), encoding="utf-8"
            )
            temp.replace(dest)
            done += 1
            print(f"[{done}/{len(tasks)}] "
                  f"scenario={task[0]}", flush=True)

    records = []
    for t in tasks:
        p = checkpoints / f"scenario_{t[0]:03d}.json"
        obj = json.loads(p.read_text(encoding="utf-8"))
        expected = t[2] * len(METHODS) * len(CONDITIONS)
        if len(obj["records"]) != expected:
            raise RuntimeError("Incomplete checkpoint")
        records.extend(obj["records"])

    df = pd.DataFrame(records)
    expected = len(tasks) * tasks[0][2] * 5 * 6
    if len(df) != expected:
        raise RuntimeError(
            f"Expected {expected} rows, got {len(df)}"
        )
    if df.duplicated([
        "scenario_id", "run_index",
        "condition", "method"
    ]).any():
        raise RuntimeError("Duplicate evaluations")

    df.to_csv(run_dir / "per_decision.csv", index=False)

    summary = df.groupby(
        ["case_size", "condition", "method"],
        as_index=False
    ).agg(
        trials=("PAS", "size"),
        mean_PAS=("PAS", "mean"),
        mean_failure_rate=("attack_failure_rate", "mean"),
        tail_event_rate=("loss_above_030", "mean"),
        switch_rate=("switched_from_CDP", "mean")
    )
    summary.to_csv(run_dir / "condition_summary.csv",
                   index=False)

    rows = []
    clean = df[df["condition"] == "clean"]
    rng = np.random.default_rng(20261020)

    for n in (14, 57, 118, 300):
        scenario_means = clean[
            clean["case_size"] == n
        ].groupby(["scenario_id", "method"])[
            "PAS"
        ].mean().unstack("method")

        if len(scenario_means) != (
            1 if args.mode == "pilot" else 20
        ):
            raise RuntimeError("Scenario comparison incomplete")

        for method, baseline in COMPARISONS:
            diff = (
                scenario_means[baseline]
                - scenario_means[method]
            ).to_numpy()

            boot = rng.choice(
                diff, size=(3000, len(diff)),
                replace=True
            ).mean(axis=1)

            nonzero = diff[np.abs(diff) > 1e-12]
            p = (
                float(wilcoxon(nonzero).pvalue)
                if len(nonzero) else 1.0
            )
            rows.append({
                "case_size": n,
                "comparison": f"{method} vs {baseline}",
                "absolute_gain": float(diff.mean()),
                "CI_low": float(np.quantile(boot, 0.025)),
                "CI_high": float(np.quantile(boot, 0.975)),
                "wilcoxon_p": p,
                "better_scenarios": int((diff > 1e-12).sum()),
                "worse_scenarios": int((diff < -1e-12).sum())
            })

    stats = pd.DataFrame(rows)
    stats["holm_p_16"] = holm(stats["wilcoxon_p"])
    stats.to_csv(run_dir / "clean_paired_statistics.csv",
                 index=False)

    print("\nEXPERIMENT 39 - CLEAN STATE ABLATIONS")
    print(summary[
        summary["condition"] == "clean"
    ].to_string(index=False))

    print("\nEXPERIMENT 39 - PAIRED STATISTICS")
    print(stats.to_string(index=False))

    print("\nEXPERIMENT 39 - OBSERVATION STRESS")
    print(summary[
        summary["method"] == "FULL"
    ].to_string(index=False))

    print("\nEXPERIMENT 39 COMPLETED SUCCESSFULLY",
          flush=True)

if __name__ == "__main__":
    main()
