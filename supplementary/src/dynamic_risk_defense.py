import os
import numpy as np
import pandas as pd
import networkx as nx


def normalize_array(x: np.ndarray) -> np.ndarray:
    """Normalize numeric array to [0, 1]."""
    x = np.asarray(x, dtype=float)
    if np.max(x) - np.min(x) < 1e-12:
        return np.zeros_like(x)
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def load_basic_tables(pattern: str = "RC", table_dir: str = "outputs/tables"):
    """Load communication features, power features, and coupling table."""
    comm_features = pd.read_csv(os.path.join(table_dir, "comm_features.csv"))
    power_features = pd.read_csv(os.path.join(table_dir, "power_features.csv"))
    coupling = pd.read_csv(os.path.join(table_dir, f"coupling_{pattern}.csv"))

    merged = coupling.merge(comm_features, on="comm_node", how="left")
    merged = merged.merge(power_features, on="power_node", how="left")

    return comm_features, power_features, coupling, merged


def build_comm_graph(n_nodes: int = 118, seed: int = 42):
    """Rebuild BA communication network."""
    return nx.barabasi_albert_graph(n=n_nodes, m=2, seed=seed)


def estimate_infection_risk(
    Gc: nx.Graph,
    active_nodes=None,
    exposed_nodes=None,
    protected_nodes=None,
):
    """
    Estimate node-level infection risk using local neighborhood exposure.

    This is a lightweight risk estimator:
    - active infectious nodes have the highest risk.
    - exposed nodes have high latent risk.
    - susceptible nodes near active/exposed nodes have elevated risk.
    - protected nodes receive lower risk.
    """
    n = Gc.number_of_nodes()

    active_nodes = set(active_nodes) if active_nodes is not None else set()
    exposed_nodes = set(exposed_nodes) if exposed_nodes is not None else set()
    protected_nodes = set(protected_nodes) if protected_nodes is not None else set()

    risk = np.zeros(n, dtype=float)

    for node in range(n):
        if node in active_nodes:
            risk[node] = 1.0
        elif node in exposed_nodes:
            risk[node] = 0.75
        else:
            neighbors = list(Gc.neighbors(node))
            if len(neighbors) == 0:
                risk[node] = 0.0
            else:
                n_active = sum(1 for nb in neighbors if nb in active_nodes)
                n_exposed = sum(1 for nb in neighbors if nb in exposed_nodes)
                local_pressure = (n_active + 0.5 * n_exposed) / len(neighbors)
                risk[node] = local_pressure

        if node in protected_nodes:
            risk[node] *= 0.2

    return normalize_array(risk)


def estimate_information_uncertainty(
    Gc: nx.Graph,
    attack_information_ratio: float = 0.5,
):
    """
    Estimate uncertainty/exposure risk under incomplete attacker information.

    Intuition:
    - When the attacker has limited information, high-degree nodes that are likely
      to be discovered are more exposed.
    - When the attacker has full information, exposure uncertainty approaches degree-based visibility.

    We model this as degree centrality scaled by information ratio.
    """
    degrees = np.array([Gc.degree(i) for i in range(Gc.number_of_nodes())], dtype=float)
    degree_norm = normalize_array(degrees)

    # Exposure uncertainty is higher for structurally visible nodes,
    # and it is amplified as the attacker's information coverage increases.
    uncertainty = attack_information_ratio * degree_norm + (1.0 - attack_information_ratio) * 0.5 * degree_norm

    return normalize_array(uncertainty)


def compute_dynamic_risk_score(
    pattern: str = "RC",
    attack_information_ratio: float = 0.5,
    active_nodes=None,
    exposed_nodes=None,
    protected_nodes=None,
    weights=None,
    table_dir: str = "outputs/tables",
    seed: int = 42,
):
    """
    Compute dynamic cyber-physical risk score for each communication node.

    Risk components:
    1. Communication degree
    2. Communication betweenness
    3. Coupled power-node degree
    4. Coupled power-node capacity
    5. Estimated infection risk
    6. Information uncertainty / exposure risk
    """
    if weights is None:
        weights = {
            "comm_degree": 0.20,
            "comm_betweenness": 0.15,
            "power_degree": 0.15,
            "power_capacity": 0.15,
            "infection_risk": 0.20,
            "uncertainty": 0.15,
        }

    _, _, _, merged = load_basic_tables(pattern=pattern, table_dir=table_dir)
    Gc = build_comm_graph(n_nodes=len(merged), seed=seed)

    infection_risk = estimate_infection_risk(
        Gc=Gc,
        active_nodes=active_nodes,
        exposed_nodes=exposed_nodes,
        protected_nodes=protected_nodes,
    )

    uncertainty = estimate_information_uncertainty(
        Gc=Gc,
        attack_information_ratio=attack_information_ratio,
    )

    merged = merged.sort_values("comm_node").reset_index(drop=True)

    merged["infection_risk_norm"] = infection_risk
    merged["uncertainty_norm"] = uncertainty

    merged["dynamic_risk_score"] = (
        weights["comm_degree"] * merged["comm_degree_norm"]
        + weights["comm_betweenness"] * merged["comm_betweenness_norm"]
        + weights["power_degree"] * merged["power_degree_norm"]
        + weights["power_capacity"] * merged["capacity_proxy_norm"]
        + weights["infection_risk"] * merged["infection_risk_norm"]
        + weights["uncertainty"] * merged["uncertainty_norm"]
    )

    merged["dynamic_risk_score_norm"] = normalize_array(merged["dynamic_risk_score"].values)

    return merged


def select_drad_protected_nodes(
    defense_budget: int,
    pattern: str = "RC",
    attack_information_ratio: float = 0.5,
    active_nodes=None,
    exposed_nodes=None,
    already_protected_nodes=None,
    weights=None,
    table_dir: str = "outputs/tables",
    seed: int = 42,
):
    """
    Select protected nodes according to DRAD-CPPS.
    """
    defense_budget = int(defense_budget)
    if defense_budget <= 0:
        return [], None

    already_protected_nodes = set(already_protected_nodes) if already_protected_nodes is not None else set()

    risk_table = compute_dynamic_risk_score(
        pattern=pattern,
        attack_information_ratio=attack_information_ratio,
        active_nodes=active_nodes,
        exposed_nodes=exposed_nodes,
        protected_nodes=already_protected_nodes,
        weights=weights,
        table_dir=table_dir,
        seed=seed,
    )

    candidates = risk_table[~risk_table["comm_node"].isin(already_protected_nodes)].copy()

    selected = (
        candidates.sort_values("dynamic_risk_score_norm", ascending=False)
        .head(defense_budget)["comm_node"]
        .astype(int)
        .tolist()
    )

    return sorted(selected), risk_table


def build_drad_defense_table(
    patterns=None,
    attack_information_ratios=None,
    defense_budgets=None,
    seed: int = 42,
    out_dir: str = "outputs/tables",
):
    """
    Generate DRAD-CPPS protected-node table for comparison with baselines.
    """
    os.makedirs(out_dir, exist_ok=True)

    if patterns is None:
        patterns = ["DDAC", "DDDC", "DCAC", "DCDC", "RC"]

    if attack_information_ratios is None:
        attack_information_ratios = [0.25, 0.50, 1.00]

    if defense_budgets is None:
        defense_budgets = [0, 20, 40, 60, 80, 100]

    rows = []
    risk_examples = []

    # For this first static DRAD table, no active/exposed nodes are assumed yet.
    # In the full simulation, active/exposed states will be updated dynamically.
    for pattern in patterns:
        for info_ratio in attack_information_ratios:
            for budget in defense_budgets:
                selected_nodes, risk_table = select_drad_protected_nodes(
                    defense_budget=budget,
                    pattern=pattern,
                    attack_information_ratio=info_ratio,
                    active_nodes=None,
                    exposed_nodes=None,
                    already_protected_nodes=None,
                    table_dir=out_dir,
                    seed=seed,
                )

                rows.append({
                    "pattern": pattern,
                    "strategy": "DRAD",
                    "attack_information_ratio": info_ratio,
                    "defense_budget": budget,
                    "n_protected": len(selected_nodes),
                    "protected_nodes": ",".join(map(str, selected_nodes)),
                })

                if budget == 20 and info_ratio == 0.50:
                    tmp = risk_table.copy()
                    tmp["pattern"] = pattern
                    tmp["attack_information_ratio"] = info_ratio
                    risk_examples.append(tmp)

    df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "drad_defense_nodes.csv")
    df.to_csv(out_path, index=False)

    if len(risk_examples) > 0:
        risk_df = pd.concat(risk_examples, ignore_index=True)
        risk_out = os.path.join(out_dir, "drad_risk_scores_examples.csv")
        risk_df.to_csv(risk_out, index=False)

    print("\nDRAD-CPPS defense table generated.")
    print(df.head(10))
    print(f"\nSaved to: {out_path}")

    if len(risk_examples) > 0:
        print(f"Risk score examples saved to: {risk_out}")

    return df


if __name__ == "__main__":
    build_drad_defense_table()