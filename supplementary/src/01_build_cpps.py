import os
import json
import numpy as np
import pandas as pd
import networkx as nx

try:
    import pandapower.networks as pn
except ImportError as exc:
    raise ImportError(
        "pandapower is not installed. Please run: pip install pandapower"
    ) from exc


def normalize_array(x: np.ndarray) -> np.ndarray:
    """Normalize a numeric array to [0, 1]."""
    x = np.asarray(x, dtype=float)
    if np.max(x) - np.min(x) < 1e-12:
        return np.zeros_like(x)
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def build_power_network_ieee118():
    """
    Build the IEEE 118-bus physical power network.

    Returns
    -------
    net : pandapowerNet
        IEEE 118-bus power system.
    Gp : networkx.Graph
        Topological graph of the physical power network.
    power_features : pandas.DataFrame
        Node-level physical-layer features.
    """
    net = pn.case118()

    Gp = nx.Graph()

    for bus_idx in net.bus.index:
        Gp.add_node(int(bus_idx))

    for _, row in net.line.iterrows():
        u = int(row["from_bus"])
        v = int(row["to_bus"])
        Gp.add_edge(u, v)

    n = len(Gp.nodes)

    degree = dict(Gp.degree())
    betweenness = nx.betweenness_centrality(Gp, normalized=True)

    load = np.zeros(n)
    gen = np.zeros(n)

    if len(net.load) > 0:
        for _, row in net.load.iterrows():
            bus = int(row["bus"])
            load[bus] += float(row["p_mw"])

    if len(net.gen) > 0:
        for _, row in net.gen.iterrows():
            bus = int(row["bus"])
            gen[bus] += float(row["p_mw"])

    # The slack bus is approximated with a generation proxy.
    if len(net.ext_grid) > 0:
        total_load = float(net.load["p_mw"].sum()) if len(net.load) > 0 else 1.0
        for _, row in net.ext_grid.iterrows():
            bus = int(row["bus"])
            gen[bus] += max(total_load * 0.1, 1.0)

    capacity_proxy = load + gen

    power_features = pd.DataFrame({
        "power_node": list(range(n)),
        "power_degree": [degree.get(i, 0) for i in range(n)],
        "power_betweenness": [betweenness.get(i, 0.0) for i in range(n)],
        "load_mw": load,
        "gen_proxy_mw": gen,
        "capacity_proxy": capacity_proxy,
    })

    for col in [
        "power_degree",
        "power_betweenness",
        "load_mw",
        "gen_proxy_mw",
        "capacity_proxy",
    ]:
        power_features[col + "_norm"] = normalize_array(power_features[col].values)

    return net, Gp, power_features


def build_communication_network_ba(n_nodes: int = 118, m: int = 2, seed: int = 42):
    """
    Build a scale-free communication network using the Barabasi-Albert model.
    """
    Gc = nx.barabasi_albert_graph(n=n_nodes, m=m, seed=seed)

    degree = dict(Gc.degree())
    betweenness = nx.betweenness_centrality(Gc, normalized=True)
    closeness = nx.closeness_centrality(Gc)

    comm_features = pd.DataFrame({
        "comm_node": list(range(n_nodes)),
        "comm_degree": [degree.get(i, 0) for i in range(n_nodes)],
        "comm_betweenness": [betweenness.get(i, 0.0) for i in range(n_nodes)],
        "comm_closeness": [closeness.get(i, 0.0) for i in range(n_nodes)],
    })

    for col in ["comm_degree", "comm_betweenness", "comm_closeness"]:
        comm_features[col + "_norm"] = normalize_array(comm_features[col].values)

    return Gc, comm_features


def generate_coupling(comm_features, power_features, pattern: str = "RC", seed: int = 42):
    """
    Generate one-to-one coupling between communication nodes and power nodes.

    Coupling patterns:
    DDAC: degree-degree assortative coupling
    DDDC: degree-degree disassortative coupling
    DCAC: degree-capacity assortative coupling
    DCDC: degree-capacity disassortative coupling
    RC: random coupling
    """
    rng = np.random.default_rng(seed)

    comm_nodes = comm_features["comm_node"].values.copy()
    power_nodes = power_features["power_node"].values.copy()

    if pattern == "DDAC":
        comm_sorted = comm_features.sort_values(
            "comm_degree", ascending=False
        )["comm_node"].values
        power_sorted = power_features.sort_values(
            "power_degree", ascending=False
        )["power_node"].values

    elif pattern == "DDDC":
        comm_sorted = comm_features.sort_values(
            "comm_degree", ascending=False
        )["comm_node"].values
        power_sorted = power_features.sort_values(
            "power_degree", ascending=True
        )["power_node"].values

    elif pattern == "DCAC":
        comm_sorted = comm_features.sort_values(
            "comm_degree", ascending=False
        )["comm_node"].values
        power_sorted = power_features.sort_values(
            "capacity_proxy", ascending=False
        )["power_node"].values

    elif pattern == "DCDC":
        comm_sorted = comm_features.sort_values(
            "comm_degree", ascending=False
        )["comm_node"].values
        power_sorted = power_features.sort_values(
            "capacity_proxy", ascending=True
        )["power_node"].values

    elif pattern == "RC":
        comm_sorted = comm_nodes.copy()
        power_sorted = power_nodes.copy()
        rng.shuffle(power_sorted)

    else:
        raise ValueError(f"Unknown coupling pattern: {pattern}")

    coupling = pd.DataFrame({
        "comm_node": comm_sorted,
        "power_node": power_sorted,
        "pattern": pattern,
    })

    return coupling


def build_all_coupled_networks(seed: int = 42, out_dir: str = "outputs/tables"):
    os.makedirs(out_dir, exist_ok=True)

    net, Gp, power_features = build_power_network_ieee118()
    Gc, comm_features = build_communication_network_ba(
        n_nodes=len(Gp.nodes),
        m=2,
        seed=seed,
    )

    power_features.to_csv(os.path.join(out_dir, "power_features.csv"), index=False)
    comm_features.to_csv(os.path.join(out_dir, "comm_features.csv"), index=False)

    patterns = ["DDAC", "DDDC", "DCAC", "DCDC", "RC"]
    for pattern in patterns:
        coupling = generate_coupling(
            comm_features=comm_features,
            power_features=power_features,
            pattern=pattern,
            seed=seed,
        )
        coupling.to_csv(os.path.join(out_dir, f"coupling_{pattern}.csv"), index=False)

    summary = {
        "n_power_nodes": len(Gp.nodes),
        "n_power_edges": len(Gp.edges),
        "n_comm_nodes": len(Gc.nodes),
        "n_comm_edges": len(Gc.edges),
        "coupling_patterns": patterns,
        "seed": seed,
    }

    with open(os.path.join(out_dir, "network_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nCPPS network construction finished.")
    print(json.dumps(summary, indent=2))

    print("\nPower-layer feature preview:")
    print(power_features.head())

    print("\nCommunication-layer feature preview:")
    print(comm_features.head())


if __name__ == "__main__":
    build_all_coupled_networks(seed=42)