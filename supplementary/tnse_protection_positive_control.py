import importlib.util
import sys
from pathlib import Path

import numpy as np
import networkx as nx

ROOT = Path.cwd()
P = ROOT / "revision_experiments/47_final_closed_loop_validation.py"

spec = importlib.util.spec_from_file_location(
    "tnse_positive_control_ref", P
)

ref = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ref
spec.loader.exec_module(ref)

graph = nx.barabasi_albert_graph(
    118, 2, seed=42
)

adj = ref.adjacency(graph)

degree = np.array(
    [graph.degree(i) for i in range(118)]
)

# Seed infection at high-degree nodes to ensure
# the test actually exercises transmission.
infected = np.argsort(-degree)[:10]

initial = np.zeros(118, dtype=np.int8)
initial[infected] = 2

protected_all = tuple(
    i for i in range(118)
    if i not in set(infected)
)

records = []

for s in range(40):

    tape = np.random.default_rng(
        550000+s
    ).random((60, 3, 118))

    uncontrolled = ref.propagate(
        initial.copy(),
        tuple(),
        tape.copy(),
        adj
    )

    protected = ref.propagate(
        initial.copy(),
        protected_all,
        tape.copy(),
        adj
    )

    # Cumulative-ever-affected status proxy.
    # State labels must match the SEID source.
    spread_none = int(
        (uncontrolled != 0).sum()
    )

    spread_protected = int(
        (protected != 0).sum()
    )

    records.append(
        (
            s,
            spread_none,
            spread_protected,
            spread_none-spread_protected
        )
    )

import pandas as pd

df = pd.DataFrame(
    records,
    columns=[
        "seed",
        "affected_unprotected",
        "affected_protected",
        "protection_difference"
    ]
)

out = (
    ROOT /
    "revision_experiments/results/"
    "55_final_evidence_closeout/"
    "PROTECTION_POSITIVE_CONTROL.csv"
)

out.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    out,
    index=False
)

print(df.describe().to_string())

print(
    "\nDifferent terminal outcomes:",
    int(
        (
            df.affected_unprotected
            != df.affected_protected
        ).sum()
    ),
    "/",
    len(df)
)

if (
    df.affected_unprotected
    == df.affected_protected
).all():
    print(
        "\nWARNING: No protective-action effect "
        "detected in this stress test. "
        "Inspect propagate() before paper claims."
    )
else:
    print(
        "\nPROTECTION EFFECT DETECTED. "
        "This verifies an interface response, "
        "not optimality or field safety."
    )
