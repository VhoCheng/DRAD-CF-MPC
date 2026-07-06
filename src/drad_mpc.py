import os
import hashlib
import numpy as np
import pandas as pd
import networkx as nx

from seid_malware import simulate_seid_once
from baseline_defense import select_protected_nodes
from dynamic_risk_defense import compute_dynamic_risk_score


def stable_int_hash(text: str, modulo: int = 10_000_000) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def normalize_array(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    if np.max(x) - np.min(x) < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def build_comm_graph(n_nodes=118, seed=42):
    return nx.barabasi_albert_graph(n=n_nodes, m=2, seed=seed)


def load_tables(pattern, table_dir="outputs/tables"):
    comm = pd.read_csv(os.path.join(table_dir, "comm_features.csv"))
    power = pd.read_csv(os.path.join(table_dir, "power_features.csv"))
    coupling = pd.read_csv(os.path.join(table_dir, f"coupling_{pattern}.csv"))
    merged = coupling.merge(comm, on="comm_node", how="left")
    merged = merged.merge(power, on="power_node", how="left")
    return comm, power, coupling, merged


def compute_physics_impact_table(pattern, table_dir="outputs/tables"):
    _, _, _, df = load_tables(pattern=pattern, table_dir=table_dir)
    df = df.copy()
    df["impact_score_raw"] = (
        0.35 * df["load_mw_norm"]
        + 0.25 * df["capacity_proxy_norm"]
        + 0.20 * df["power_degree_norm"]
        + 0.20 * df["power_betweenness_norm"]
    )
    df["impact_score_norm"] = normalize_array(df["impact_score_raw"].values)
    return df


def evaluate_pas_proxy_mpc(active_infectious_nodes, pattern, table_dir="outputs/tables"):
    df = compute_physics_impact_table(pattern=pattern, table_dir=table_dir)
    active_infectious_nodes = set(active_infectious_nodes)
    if not active_infectious_nodes:
        return 0.0
    affected = df[df["comm_node"].isin(active_infectious_nodes)].copy()
    if len(affected) == 0:
        return 0.0
    total_impact = float(df["impact_score_raw"].sum()) + 1e-9
    affected_impact = float(affected["impact_score_raw"].sum())
    return float(np.clip(affected_impact / total_impact, 0.0, 1.0))


def evaluate_resilience_proxy_mpc(pas, defense_budget, max_budget=100):
    defense_factor = defense_budget / max_budget
    recovery_time = 1.0 + 20.0 * pas * (1.0 - 0.35 * defense_factor)
    recovery_time = max(1.0, recovery_time)
    resilience_index = 1.0 - 0.65 * pas + 0.10 * defense_factor
    resilience_index = float(np.clip(resilience_index, 0.0, 1.0))
    recovery_efficiency = (1.0 - pas) / (1.0 + defense_budget / max_budget)
    recovery_efficiency = float(np.clip(recovery_efficiency, 0.0, 1.0))
    return float(recovery_time), float(resilience_index), float(recovery_efficiency)


def _top_by_score(df, score_col, defense_budget):
    return (
        df.sort_values(score_col, ascending=False)
        .head(defense_budget)["comm_node"]
        .astype(int)
        .sort_values()
        .tolist()
    )


def candidate_physics_nodes(pattern, defense_budget, table_dir="outputs/tables"):
    df = compute_physics_impact_table(pattern=pattern, table_dir=table_dir)
    return _top_by_score(df, "impact_score_norm", defense_budget)


def candidate_hybrid_nodes(pattern, defense_budget, table_dir="outputs/tables"):
    df = compute_physics_impact_table(pattern=pattern, table_dir=table_dir).copy()
    df["hybrid_score"] = (
        0.40 * df["comm_degree_norm"]
        + 0.20 * df["comm_betweenness_norm"]
        + 0.10 * df["power_degree_norm"]
        + 0.10 * df["capacity_proxy_norm"]
        + 0.20 * df["impact_score_norm"]
    )
    return _top_by_score(df, "hybrid_score", defense_budget)


def candidate_cdp_physics_mix_nodes(pattern, defense_budget, mix_ratio=0.90, table_dir="outputs/tables"):
    if defense_budget <= 0:
        return []
    cdp_budget = int(round(defense_budget * mix_ratio))
    cdp_budget = min(max(cdp_budget, 0), defense_budget)
    cdp_nodes = select_protected_nodes(
        strategy="CDP", defense_budget=cdp_budget, pattern=pattern, table_dir=table_dir
    )
    physics_nodes = candidate_physics_nodes(
        pattern=pattern, defense_budget=max(defense_budget, defense_budget - cdp_budget), table_dir=table_dir
    )
    selected = list(cdp_nodes)
    for node in physics_nodes:
        if node not in selected:
            selected.append(node)
        if len(selected) >= defense_budget:
            break
    return sorted(selected[:defense_budget])


def candidate_cdp_pdp_mix_nodes(pattern, defense_budget, table_dir="outputs/tables"):
    if defense_budget <= 0:
        return []
    cdp_budget = int(round(defense_budget * 0.90))
    cdp_nodes = select_protected_nodes(
        strategy="CDP", defense_budget=cdp_budget, pattern=pattern, table_dir=table_dir
    )
    pdp_nodes = select_protected_nodes(
        strategy="PDP", defense_budget=defense_budget, pattern=pattern, table_dir=table_dir
    )
    selected = list(cdp_nodes)
    for node in pdp_nodes:
        if node not in selected:
            selected.append(node)
        if len(selected) >= defense_budget:
            break
    return sorted(selected[:defense_budget])


def candidate_drad_score_nodes(pattern, defense_budget, attack_information_ratio=0.5,
                               active_nodes=None, exposed_nodes=None,
                               table_dir="outputs/tables", seed=42):
    risk_table = compute_dynamic_risk_score(
        pattern=pattern,
        attack_information_ratio=attack_information_ratio,
        active_nodes=active_nodes,
        exposed_nodes=exposed_nodes,
        protected_nodes=None,
        table_dir=table_dir,
        seed=seed,
        weights={
            "comm_degree": 0.36,
            "comm_betweenness": 0.18,
            "power_degree": 0.08,
            "power_capacity": 0.08,
            "infection_risk": 0.20,
            "uncertainty": 0.10,
        },
    )
    impact = compute_physics_impact_table(pattern=pattern, table_dir=table_dir)
    df = risk_table.merge(impact[["comm_node", "impact_score_norm"]], on="comm_node", how="left")
    df["drad_score"] = 0.80 * df["dynamic_risk_score_norm"] + 0.20 * df["impact_score_norm"]
    return _top_by_score(df, "drad_score", defense_budget)


def candidate_rank_ensemble_nodes(pattern, defense_budget, attack_information_ratio=0.5,
                                  active_nodes=None, exposed_nodes=None,
                                  table_dir="outputs/tables", seed=42):
    risk_table = compute_dynamic_risk_score(
        pattern=pattern,
        attack_information_ratio=attack_information_ratio,
        active_nodes=active_nodes,
        exposed_nodes=exposed_nodes,
        protected_nodes=None,
        table_dir=table_dir,
        seed=seed,
        weights={
            "comm_degree": 0.32,
            "comm_betweenness": 0.18,
            "power_degree": 0.08,
            "power_capacity": 0.08,
            "infection_risk": 0.24,
            "uncertainty": 0.10,
        },
    )
    impact = compute_physics_impact_table(pattern=pattern, table_dir=table_dir)
    df = risk_table.merge(impact[["comm_node", "impact_score_norm"]], on="comm_node", how="left")
    df["rank_comm_degree"] = df["comm_degree_norm"].rank(ascending=False, method="average")
    df["rank_comm_between"] = df["comm_betweenness_norm"].rank(ascending=False, method="average")
    df["rank_power_capacity"] = df["capacity_proxy_norm"].rank(ascending=False, method="average")
    df["rank_infection"] = df["infection_risk_norm"].rank(ascending=False, method="average")
    df["rank_impact"] = df["impact_score_norm"].rank(ascending=False, method="average")
    df["ensemble_rank"] = (
        0.32 * df["rank_comm_degree"]
        + 0.18 * df["rank_comm_between"]
        + 0.08 * df["rank_power_capacity"]
        + 0.22 * df["rank_infection"]
        + 0.20 * df["rank_impact"]
    )
    return (
        df.sort_values("ensemble_rank", ascending=True)
        .head(defense_budget)["comm_node"]
        .astype(int)
        .sort_values()
        .tolist()
    )


def generate_candidate_policies(pattern, defense_budget, attack_information_ratio=0.5,
                                active_nodes=None, exposed_nodes=None,
                                table_dir="outputs/tables", seed=42):
    candidates = {}
    for policy in ["CDP", "PDP", "CP", "RP"]:
        candidates[policy] = select_protected_nodes(
            strategy=policy,
            defense_budget=defense_budget,
            pattern=pattern,
            seed=seed,
            table_dir=table_dir,
        )
    candidates["PHYSICS"] = candidate_physics_nodes(pattern, defense_budget, table_dir)
    candidates["HYBRID"] = candidate_hybrid_nodes(pattern, defense_budget, table_dir)
    candidates["CDP_PHYSICS_95"] = candidate_cdp_physics_mix_nodes(pattern, defense_budget, 0.95, table_dir)
    candidates["CDP_PHYSICS_90"] = candidate_cdp_physics_mix_nodes(pattern, defense_budget, 0.90, table_dir)
    candidates["CDP_PDP_90"] = candidate_cdp_pdp_mix_nodes(pattern, defense_budget, table_dir)
    candidates["DRAD_SCORE"] = candidate_drad_score_nodes(
        pattern, defense_budget, attack_information_ratio, active_nodes, exposed_nodes, table_dir, seed
    )
    candidates["RANK_ENSEMBLE"] = candidate_rank_ensemble_nodes(
        pattern, defense_budget, attack_information_ratio, active_nodes, exposed_nodes, table_dir, seed
    )
    cleaned = {}
    seen = set()
    for name, nodes in candidates.items():
        key = tuple(sorted(nodes))
        if key not in seen:
            cleaned[name] = sorted(nodes)
            seen.add(key)
    return cleaned


def evaluate_candidate_with_common_random_numbers(candidate_nodes, common_seeds, Gc, pattern,
                                                  attack_information_ratio, defense_budget,
                                                  onset_time, infection_rate=0.01,
                                                  activation_rate=0.15,
                                                  detection_rate=0.00001):
    pas_values, failed_values, rt_values, ri_values, re_values = [], [], [], [], []
    for sim_seed in common_seeds:
        result = simulate_seid_once(
            Gc=Gc,
            onset_time=onset_time,
            infection_rate=infection_rate,
            activation_rate=activation_rate,
            detection_rate=detection_rate,
            attack_information_ratio=attack_information_ratio,
            protected_nodes=candidate_nodes,
            strict_detection_failure=True,
            seed=int(sim_seed),
        )
        pas = 0.0 if result["attack_failed"] else evaluate_pas_proxy_mpc(
            active_infectious_nodes=result["active_infectious_nodes"],
            pattern=pattern,
            table_dir="outputs/tables",
        )
        rt, ri, re = evaluate_resilience_proxy_mpc(pas=pas, defense_budget=defense_budget, max_budget=100)
        pas_values.append(pas)
        failed_values.append(int(result["attack_failed"]))
        rt_values.append(rt)
        ri_values.append(ri)
        re_values.append(re)
    mean_pas = float(np.mean(pas_values))
    ps = 1.0 - float(np.mean(failed_values))
    ep = mean_pas * ps
    mean_rt = float(np.mean(rt_values))
    mean_ri = float(np.mean(ri_values))
    mean_re = float(np.mean(re_values))
    normalized_rt = mean_rt / 21.0
    objective = 0.68 * ep + 0.24 * mean_pas + 0.06 * normalized_rt - 0.02 * mean_ri
    return {
        "objective": float(objective),
        "pred_PAS": mean_pas,
        "pred_PS": ps,
        "pred_EP": ep,
        "pred_RT": mean_rt,
        "pred_RI": mean_ri,
        "pred_RE": mean_re,
        "pas_samples": pas_values,
    }


def paired_improvement(candidate_metrics, baseline_metrics):
    cand = np.asarray(candidate_metrics["pas_samples"], dtype=float)
    base = np.asarray(baseline_metrics["pas_samples"], dtype=float)
    if len(cand) != len(base) or len(cand) == 0:
        return 0.0, 0.0
    diff = base - cand
    mean_diff = float(np.mean(diff))
    stderr = 0.0 if len(diff) <= 1 else float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
    return mean_diff, stderr


def select_drad_mpc_protected_nodes(defense_budget, pattern="RC", attack_information_ratio=0.5,
                                    active_nodes=None, exposed_nodes=None, onset_time=120,
                                    infection_rate=0.01, activation_rate=0.15,
                                    detection_rate=0.00001, lookahead_runs=20,
                                    switch_margin=0.10, confidence_k=0.50,
                                    table_dir="outputs/tables", seed=42):
    defense_budget = int(defense_budget)
    if defense_budget <= 0:
        return [], {
            "selected_policy": "NP",
            "candidate_scores": {},
            "candidate_metrics": {},
            "best_policy_before_guard": "NP",
            "safe_guard": "zero_budget",
        }
    Gc = build_comm_graph(n_nodes=118, seed=42)
    candidates = generate_candidate_policies(
        pattern=pattern,
        defense_budget=defense_budget,
        attack_information_ratio=attack_information_ratio,
        active_nodes=active_nodes,
        exposed_nodes=exposed_nodes,
        table_dir=table_dir,
        seed=seed,
    )
    if "CDP" not in candidates:
        candidates["CDP"] = select_protected_nodes(
            strategy="CDP", defense_budget=defense_budget, pattern=pattern, seed=seed, table_dir=table_dir
        )
    common_seeds = [
        seed + 777_000 + 10_007 * k + stable_int_hash(f"{pattern}-{onset_time}-{defense_budget}", 100_000)
        for k in range(max(1, int(lookahead_runs)))
    ]
    candidate_metrics, candidate_scores = {}, {}
    for name, nodes in candidates.items():
        metrics = evaluate_candidate_with_common_random_numbers(
            candidate_nodes=nodes,
            common_seeds=common_seeds,
            Gc=Gc,
            pattern=pattern,
            attack_information_ratio=attack_information_ratio,
            defense_budget=defense_budget,
            onset_time=onset_time,
            infection_rate=infection_rate,
            activation_rate=activation_rate,
            detection_rate=detection_rate,
        )
        candidate_metrics[name] = metrics
        candidate_scores[name] = metrics["objective"]
    cdp_score = candidate_scores["CDP"]
    best_policy = min(candidate_scores, key=candidate_scores.get)
    best_score = candidate_scores[best_policy]
    selected_policy = "CDP"
    safe_guard = "cdp_anchor"
    if best_policy != "CDP":
        objective_improvement = (cdp_score - best_score) / (abs(cdp_score) + 1e-9)
        mean_diff, stderr = paired_improvement(candidate_metrics[best_policy], candidate_metrics["CDP"])
        conservative_gain = mean_diff - confidence_k * stderr
        if objective_improvement >= switch_margin and conservative_gain > 0:
            selected_policy = best_policy
            safe_guard = "switched_with_counterfactual_gain"
        else:
            selected_policy = "CDP"
            safe_guard = "fallback_to_cdp"
    else:
        selected_policy = "CDP"
        safe_guard = "cdp_best"
    selected_nodes = candidates[selected_policy]
    meta = {
        "selected_policy": selected_policy,
        "candidate_scores": candidate_scores,
        "candidate_metrics": {k: {kk: vv for kk, vv in v.items() if kk != "pas_samples"}
                              for k, v in candidate_metrics.items()},
        "best_policy_before_guard": best_policy,
        "best_score_before_guard": best_score,
        "cdp_score": cdp_score,
        "safe_guard": safe_guard,
    }
    return sorted(selected_nodes), meta
