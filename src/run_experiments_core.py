import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from seid_malware import simulate_seid_once
from baseline_defense import select_protected_nodes
from dynamic_risk_defense import select_drad_protected_nodes
from drad_mpc import select_drad_mpc_protected_nodes


def build_comm_graph(n_nodes: int = 118, seed: int = 42):
    return nx.barabasi_albert_graph(n=n_nodes, m=2, seed=seed)


def load_tables(pattern: str, table_dir: str = "outputs/tables"):
    comm_features = pd.read_csv(os.path.join(table_dir, "comm_features.csv"))
    power_features = pd.read_csv(os.path.join(table_dir, "power_features.csv"))
    coupling = pd.read_csv(os.path.join(table_dir, f"coupling_{pattern}.csv"))
    merged = coupling.merge(power_features, on="power_node", how="left")
    return comm_features, power_features, coupling, merged


def evaluate_pas_proxy(active_infectious_nodes, pattern: str, table_dir: str = "outputs/tables"):
    """
    Physics-informed percentage-of-affected-service proxy.

    This is a fast surrogate used before full DC power-flow load shedding.
    It measures the physical consequence of affected coupled power nodes.
    """
    _, _, _, merged = load_tables(pattern=pattern, table_dir=table_dir)
    active_infectious_nodes = set(active_infectious_nodes)
    if len(active_infectious_nodes) == 0:
        return 0.0

    affected = merged[merged["comm_node"].isin(active_infectious_nodes)].copy()
    if len(affected) == 0:
        return 0.0

    all_nodes = merged.copy()
    all_nodes["service_impact"] = (
        0.35 * all_nodes["load_mw_norm"]
        + 0.25 * all_nodes["capacity_proxy_norm"]
        + 0.20 * all_nodes["power_degree_norm"]
        + 0.20 * all_nodes["power_betweenness_norm"]
    )
    affected["service_impact"] = (
        0.35 * affected["load_mw_norm"]
        + 0.25 * affected["capacity_proxy_norm"]
        + 0.20 * affected["power_degree_norm"]
        + 0.20 * affected["power_betweenness_norm"]
    )
    total_impact = float(all_nodes["service_impact"].sum()) + 1e-9
    affected_impact = float(affected["service_impact"].sum())
    return float(np.clip(affected_impact / total_impact, 0.0, 1.0))


def evaluate_resilience_metrics(pas, defense_budget, max_budget=100):
    defense_factor = defense_budget / max_budget
    recovery_time = 1.0 + 20.0 * pas * (1.0 - 0.35 * defense_factor)
    recovery_time = max(1.0, recovery_time)
    resilience_index = 1.0 - 0.65 * pas + 0.10 * defense_factor
    resilience_index = float(np.clip(resilience_index, 0.0, 1.0))
    recovery_efficiency = (1.0 - pas) / (1.0 + defense_budget / max_budget)
    recovery_efficiency = float(np.clip(recovery_efficiency, 0.0, 1.0))
    return {
        "recovery_time": float(recovery_time),
        "resilience_index": float(resilience_index),
        "recovery_efficiency": float(recovery_efficiency),
    }


def get_preview_state(Gc, onset_time, attack_information_ratio,
                      infection_rate, activation_rate, detection_rate, seed):
    preview_time = max(1, int(onset_time * 0.4))
    preview = simulate_seid_once(
        Gc=Gc,
        onset_time=preview_time,
        infection_rate=infection_rate,
        activation_rate=activation_rate,
        detection_rate=detection_rate,
        attack_information_ratio=attack_information_ratio,
        protected_nodes=None,
        strict_detection_failure=False,
        seed=seed,
    )
    return preview


def get_drad_nodes_from_pre_attack_state(Gc, onset_time, pattern, attack_information_ratio,
                                         defense_budget, infection_rate, activation_rate,
                                         detection_rate, seed):
    preview = get_preview_state(
        Gc=Gc,
        onset_time=onset_time,
        attack_information_ratio=attack_information_ratio,
        infection_rate=infection_rate,
        activation_rate=activation_rate,
        detection_rate=detection_rate,
        seed=seed,
    )
    selected_nodes, _ = select_drad_protected_nodes(
        defense_budget=defense_budget,
        pattern=pattern,
        attack_information_ratio=attack_information_ratio,
        active_nodes=preview["active_infectious_nodes"],
        exposed_nodes=preview["exposed_nodes"],
        already_protected_nodes=None,
        table_dir="outputs/tables",
        seed=seed,
    )
    return selected_nodes


def get_drad_mpc_nodes_from_pre_attack_state(Gc, onset_time, pattern, attack_information_ratio,
                                             defense_budget, infection_rate, activation_rate,
                                             detection_rate, lookahead_runs, seed):
    preview = get_preview_state(
        Gc=Gc,
        onset_time=onset_time,
        attack_information_ratio=attack_information_ratio,
        infection_rate=infection_rate,
        activation_rate=activation_rate,
        detection_rate=detection_rate,
        seed=seed,
    )
    protected_nodes, meta = select_drad_mpc_protected_nodes(
        defense_budget=defense_budget,
        pattern=pattern,
        attack_information_ratio=attack_information_ratio,
        active_nodes=preview["active_infectious_nodes"],
        exposed_nodes=preview["exposed_nodes"],
        onset_time=onset_time,
        infection_rate=infection_rate,
        activation_rate=activation_rate,
        detection_rate=detection_rate,
        lookahead_runs=lookahead_runs,
        switch_margin=0.10,
        confidence_k=0.50,
        table_dir="outputs/tables",
        seed=seed,
    )
    return protected_nodes, meta


def run_single_setting(pattern, strategy, attack_information_ratio, defense_budget,
                       onset_time, n_runs=20, lookahead_runs=20,
                       infection_rate=0.01, activation_rate=0.15,
                       detection_rate=0.00001, seed=42):
    Gc = build_comm_graph(n_nodes=118, seed=42)
    pas_values, failed_values = [], []
    active_counts, exposed_counts, detected_counts = [], [], []
    rt_values, ri_values, re_values = [], [], []
    selected_policy_records, best_before_guard_records, safe_guard_records = [], [], []

    for r in range(n_runs):
        run_seed = seed + r + int(onset_time) * 1000 + int(defense_budget) * 17
        strategy_upper = strategy.upper()

        if strategy_upper == "DRAD":
            protected_nodes = get_drad_nodes_from_pre_attack_state(
                Gc=Gc,
                onset_time=onset_time,
                pattern=pattern,
                attack_information_ratio=attack_information_ratio,
                defense_budget=defense_budget,
                infection_rate=infection_rate,
                activation_rate=activation_rate,
                detection_rate=detection_rate,
                seed=run_seed,
            )
            selected_policy_records.append("DRAD_SCORE")
            best_before_guard_records.append("DRAD_SCORE")
            safe_guard_records.append("NA")

        elif strategy_upper in {"DRAD-MPC", "DRAD-CF-MPC"}:
            protected_nodes, meta = get_drad_mpc_nodes_from_pre_attack_state(
                Gc=Gc,
                onset_time=onset_time,
                pattern=pattern,
                attack_information_ratio=attack_information_ratio,
                defense_budget=defense_budget,
                infection_rate=infection_rate,
                activation_rate=activation_rate,
                detection_rate=detection_rate,
                lookahead_runs=lookahead_runs,
                seed=run_seed,
            )
            selected_policy_records.append(meta.get("selected_policy", "UNKNOWN"))
            best_before_guard_records.append(meta.get("best_policy_before_guard", "UNKNOWN"))
            safe_guard_records.append(meta.get("safe_guard", "UNKNOWN"))

        else:
            protected_nodes = select_protected_nodes(
                strategy=strategy,
                defense_budget=defense_budget,
                pattern=pattern,
                seed=run_seed,
                table_dir="outputs/tables",
            )
            selected_policy_records.append(strategy_upper)
            best_before_guard_records.append(strategy_upper)
            safe_guard_records.append("NA")

        result = simulate_seid_once(
            Gc=Gc,
            onset_time=onset_time,
            infection_rate=infection_rate,
            activation_rate=activation_rate,
            detection_rate=detection_rate,
            attack_information_ratio=attack_information_ratio,
            protected_nodes=protected_nodes,
            strict_detection_failure=True,
            seed=run_seed,
        )

        pas = 0.0 if result["attack_failed"] else evaluate_pas_proxy(
            active_infectious_nodes=result["active_infectious_nodes"],
            pattern=pattern,
            table_dir="outputs/tables",
        )
        resilience = evaluate_resilience_metrics(pas=pas, defense_budget=defense_budget, max_budget=100)

        pas_values.append(pas)
        failed_values.append(int(result["attack_failed"]))
        active_counts.append(result["n_active_infectious"])
        exposed_counts.append(result["n_exposed"])
        detected_counts.append(result["n_detected"])
        rt_values.append(resilience["recovery_time"])
        ri_values.append(resilience["resilience_index"])
        re_values.append(resilience["recovery_efficiency"])

    mean_pas = float(np.mean(pas_values))
    survival_probability = 1.0 - float(np.mean(failed_values))
    expected_payoff = mean_pas * survival_probability

    selected_policy_mode = pd.Series(selected_policy_records).mode().iloc[0]
    best_before_guard_mode = pd.Series(best_before_guard_records).mode().iloc[0]
    safe_guard_mode = pd.Series(safe_guard_records).mode().iloc[0]

    return {
        "pattern": pattern,
        "strategy": strategy,
        "attack_information_ratio": attack_information_ratio,
        "defense_budget": defense_budget,
        "onset_time": onset_time,
        "n_runs": n_runs,
        "lookahead_runs": lookahead_runs,
        "infection_rate": infection_rate,
        "activation_rate": activation_rate,
        "detection_rate": detection_rate,
        "PAS": mean_pas,
        "PS": survival_probability,
        "EP": expected_payoff,
        "mean_active_infectious": float(np.mean(active_counts)),
        "mean_exposed": float(np.mean(exposed_counts)),
        "mean_detected": float(np.mean(detected_counts)),
        "RT": float(np.mean(rt_values)),
        "RI": float(np.mean(ri_values)),
        "RE": float(np.mean(re_values)),
        "selected_policy_mode": selected_policy_mode,
        "best_before_guard_mode": best_before_guard_mode,
        "safe_guard_mode": safe_guard_mode,
    }


def run_experiments(patterns=None, strategies=None, attack_information_ratios=None,
                    defense_budgets=None, onset_times=None, n_runs=20,
                    lookahead_runs=20, out_dir="outputs/tables"):
    os.makedirs(out_dir, exist_ok=True)
    if patterns is None:
        patterns = ["DDAC", "DDDC", "DCAC", "DCDC", "RC"]
    if strategies is None:
        strategies = ["NP", "RP", "CDP", "PDP", "CP", "DRAD", "DRAD-CF-MPC"]
    if attack_information_ratios is None:
        attack_information_ratios = [0.25, 0.50, 1.00]
    if defense_budgets is None:
        defense_budgets = [0, 20, 40, 60, 80, 100]
    if onset_times is None:
        onset_times = list(range(0, 301, 30))

    rows = []
    total = len(patterns) * len(strategies) * len(attack_information_ratios) * len(defense_budgets) * len(onset_times)
    pbar = tqdm(total=total, desc="Running experiments")

    for pattern in patterns:
        for strategy in strategies:
            for info_ratio in attack_information_ratios:
                for budget in defense_budgets:
                    for onset_time in onset_times:
                        actual_budget = 0 if strategy.upper() == "NP" else budget
                        row = run_single_setting(
                            pattern=pattern,
                            strategy=strategy,
                            attack_information_ratio=info_ratio,
                            defense_budget=actual_budget,
                            onset_time=onset_time,
                            n_runs=n_runs,
                            lookahead_runs=lookahead_runs,
                        )
                        rows.append(row)
                        pbar.update(1)
    pbar.close()

    df = pd.DataFrame(rows)
    raw_out = os.path.join(out_dir, "experiment_results_raw.csv")
    df.to_csv(raw_out, index=False)

    group_cols = ["pattern", "strategy", "attack_information_ratio", "defense_budget"]
    idx = df.groupby(group_cols)["EP"].idxmax()
    opt_df = df.loc[idx].copy()
    opt_df = opt_df.rename(columns={
        "onset_time": "T_opt",
        "EP": "EP_max",
        "PAS": "PAS_at_T_opt",
        "PS": "PS_at_T_opt",
    })
    opt_out = os.path.join(out_dir, "experiment_results_optimal.csv")
    opt_df.to_csv(opt_out, index=False)

    summary_df = (
        opt_df.groupby("strategy")
        .agg(
            mean_EP_max=("EP_max", "mean"),
            mean_PAS_at_T_opt=("PAS_at_T_opt", "mean"),
            mean_PS_at_T_opt=("PS_at_T_opt", "mean"),
            mean_RT=("RT", "mean"),
            mean_RI=("RI", "mean"),
            mean_RE=("RE", "mean"),
        )
        .reset_index()
        .sort_values("mean_EP_max", ascending=True)
    )
    summary_out = os.path.join(out_dir, "experiment_strategy_summary.csv")
    summary_df.to_csv(summary_out, index=False)

    mpc_df = opt_df[opt_df["strategy"].isin(["DRAD-MPC", "DRAD-CF-MPC"])].copy()
    policy_out = None
    if len(mpc_df) > 0:
        policy_summary = mpc_df["selected_policy_mode"].value_counts().reset_index()
        policy_summary.columns = ["selected_policy", "count"]
        policy_out = os.path.join(out_dir, "drad_mpc_policy_switching_summary.csv")
        policy_summary.to_csv(policy_out, index=False)

        guard_summary = mpc_df["safe_guard_mode"].value_counts().reset_index()
        guard_summary.columns = ["safe_guard", "count"]
        guard_out = os.path.join(out_dir, "drad_mpc_safe_guard_summary.csv")
        guard_summary.to_csv(guard_out, index=False)
    else:
        policy_summary = None
        guard_summary = None
        guard_out = None

    print("\nExperiments finished.")
    print(f"Raw results saved to: {raw_out}")
    print(f"Optimal results saved to: {opt_out}")
    print(f"Strategy summary saved to: {summary_out}")
    if policy_out:
        print(f"DRAD-MPC policy switching summary saved to: {policy_out}")
        print(f"DRAD-MPC safe guard summary saved to: {guard_out}")

    print("\nOptimal result preview:")
    print(opt_df.head(10))
    print("\nStrategy-level summary, lower EP/PAS/RT is better; higher RI/RE is better:")
    print(summary_df)
    if policy_summary is not None:
        print("\nDRAD-CF-MPC selected policy distribution:")
        print(policy_summary)
        print("\nDRAD-CF-MPC safe guard distribution:")
        print(guard_summary)
    return df, opt_df, summary_df


if __name__ == "__main__":
    # Local validation setting. On AutoDL, increase n_runs to 100-300 and lookahead_runs to 30-50.
    run_experiments(
        patterns=["DDAC", "DDDC", "DCAC", "DCDC", "RC"],
        strategies=["NP", "RP", "CDP", "PDP", "CP", "DRAD", "DRAD-CF-MPC"],
        attack_information_ratios=[0.25, 0.50, 1.00],
        defense_budgets=[0, 20, 40, 60, 80, 100],
        onset_times=list(range(0, 301, 30)),
        n_runs=20,
        lookahead_runs=20,
    )
