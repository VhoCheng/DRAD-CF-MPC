#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exp56: source-pinned PGLib AC OPF benchmark."""

from pathlib import Path
import copy
import json
import hashlib
import re
import urllib.request
import traceback
import time
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
import networkx as nx

from pypower.runopf import runopf
from pypower.ppoption import ppoption
from pypower.idx_bus import PD, QD, VM, VA, VMIN, VMAX
from pypower.idx_gen import (
    PG, QG, PMIN, PMAX, QMIN, QMAX, GEN_STATUS
)
from pypower.idx_brch import (
    F_BUS, T_BUS, BR_STATUS, RATE_A,
    PF, QF, PT, QT
)
from pypower.ext2int import ext2int
from pypower.makeYbus import makeYbus
from pypower.makeSbus import makeSbus

ROOT = Path.cwd()
OUT = (
    ROOT / "revision_experiments/results/"
    "56_pglib_v2307_acopf"
)
SOURCE = OUT / "source_cases"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE.mkdir(parents=True, exist_ok=True)

SIZES = [14, 57, 118, 300]
LOAD_SCALES = [0.90, 1.00, 1.10]
N1_PER_CASE = 2

URL = (
    "https://raw.githubusercontent.com/"
    "power-grid-lib/pglib-opf/v23.07/"
    "pglib_opf_case{n}_ieee.m"
)


def get_case(n):
    dest = SOURCE / f"pglib_opf_case{n}_ieee.m"

    if not dest.exists():
        request = urllib.request.Request(
            URL.format(n=n),
            headers={
                "User-Agent": "TNSE-research-validation/1.0"
            }
        )
        with urllib.request.urlopen(
            request, timeout=35
        ) as resp:
            body = resp.read()

        if (
            b"mpc.branch" not in body
            or b"mpc.gencost" not in body
        ):
            raise ValueError("Unexpected source data")

        dest.write_bytes(body)

    raw = dest.read_text(
        encoding="utf-8", errors="replace"
    )

    if "v23.07" not in raw:
        raise ValueError(
            f"Source version mismatch: {dest}"
        )

    match = re.search(
        r"mpc\.baseMVA\s*=\s*([0-9.eE+-]+)\s*;",
        raw
    )
    if match is None:
        raise ValueError("Missing baseMVA")

    case = {
        "version": "2",
        "baseMVA": float(match.group(1))
    }

    for name in ("bus", "gen", "branch", "gencost"):
        match = re.search(
            r"mpc\." + name +
            r"\s*=\s*\[(.*?)\]\s*;",
            raw, re.S
        )

        if match is None:
            raise ValueError(
                f"Missing MATPOWER field: {name}"
            )

        clean = "\n".join(
            line.split("%", 1)[0].replace("...", "")
            for line in match.group(1).splitlines()
        )

        rows = [
            [float(v) for v in item.split()]
            for item in clean.split(";")
            if item.strip()
        ]

        if len(set(map(len, rows))) != 1:
            raise ValueError(
                f"Inconsistent matrix columns: {name}"
            )

        case[name] = np.array(rows, float)

        if not np.isfinite(case[name]).all():
            raise ValueError(
                f"Nonfinite values in {name}"
            )

    branch = case["branch"]
    rates = branch[:, RATE_A]
    active = branch[:, BR_STATUS] > 0

    rated = (
        active & (rates > 0) & (rates < 9000)
    )

    if not rated.any():
        raise ValueError("No usable branch ratings")

    meta = {
        "case_size": n,
        "source_url": URL.format(n=n),
        "source_sha256": hashlib.sha256(
            dest.read_bytes()
        ).hexdigest(),
        "source_version": "PGLib-OPF v23.07",
        "active_branch_count": int(active.sum()),
        "rated_branch_count": int(rated.sum()),
        "rate_coverage": float(
            rated.sum() / active.sum()
        )
    }

    return case, meta


def connected(case):
    graph = nx.Graph()
    graph.add_nodes_from(
        int(v) for v in case["bus"][:, 0]
    )

    for branch in case["branch"]:
        if branch[BR_STATUS] > 0:
            graph.add_edge(
                int(branch[F_BUS]),
                int(branch[T_BUS])
            )

    return nx.is_connected(graph)


def solve(ppc, scale, source, outage=-1):
    case = copy.deepcopy(ppc)

    case["bus"][:, PD] *= scale
    case["bus"][:, QD] *= scale

    if outage >= 0:
        case["branch"][outage, BR_STATUS] = 0

    result = {
        "case_size": source["case_size"],
        "condition": (
            "N-1" if outage >= 0 else "load_scale"
        ),
        "load_scale": scale,
        "outaged_branch_index": outage,
        "source_sha256": source["source_sha256"],
        "connected": connected(case),
        "success": False,
        "error": "",
        "thermal_coverage": source["rate_coverage"]
    }

    if not result["connected"]:
        result["error"] = (
            "Network disconnected after outage"
        )
        return result, None

    start = time.perf_counter()

    try:
        solved = runopf(
            case,
            ppoption(VERBOSE=0, OUT_ALL=0)
        )

        result["elapsed_s"] = (
            time.perf_counter() - start
        )
        result["success"] = bool(solved["success"])

        if not result["success"]:
            result["error"] = (
                "AC OPF reported unsuccessful"
            )
            return result, None

        bus = solved["bus"]
        gen = solved["gen"]
        branch = solved["branch"]

        rated = (
            (branch[:, BR_STATUS] > 0)
            & (branch[:, RATE_A] > 0)
            & (branch[:, RATE_A] < 9000)
        )

        apparent = np.maximum(
            np.hypot(
                branch[:, PF], branch[:, QF]
            ),
            np.hypot(
                branch[:, PT], branch[:, QT]
            )
        )

        loading = (
            apparent[rated] /
            branch[rated, RATE_A]
        )

        online = gen[:, GEN_STATUS] > 0

        vm_violation = np.maximum(
            bus[:, VMIN] - bus[:, VM],
            bus[:, VM] - bus[:, VMAX]
        )

        pg_violation = np.maximum(
            gen[online, PMIN] - gen[online, PG],
            gen[online, PG] - gen[online, PMAX]
        )

        qg_violation = np.maximum(
            gen[online, QMIN] - gen[online, QG],
            gen[online, QG] - gen[online, QMAX]
        )

        # Independent nodal power-balance residual.
        internal = ext2int(
            copy.deepcopy(solved)
        )

        B = internal["bus"]

        voltage = (
            B[:, VM] *
            np.exp(
                1j * np.pi / 180 * B[:, VA]
            )
        )

        Ybus, _, _ = makeYbus(
            internal["baseMVA"],
            B,
            internal["branch"]
        )

        Sbus = makeSbus(
            internal["baseMVA"],
            B,
            internal["gen"]
        )

        residual = (
            voltage *
            np.conj(Ybus.dot(voltage))
            - Sbus
        )

        result.update({
            "rated_thermal_violations": int(
                (loading > 1 + 1e-4).sum()
            ),
            "max_thermal_utilization": float(
                loading.max()
            ),
            "voltage_violations": int(
                (vm_violation > 1e-4).sum()
            ),
            "Pg_violations": int(
                (pg_violation > 1e-3).sum()
            ),
            "Qg_violations": int(
                (qg_violation > 1e-3).sum()
            ),
            "max_nodal_balance_error_MVA": float(
                np.max(np.abs(residual)) *
                solved["baseMVA"]
            ),
            "opf_cost": float(solved["f"]),
            "total_load_MW": float(
                bus[:, PD].sum()
            )
        })

        result["constraint_pass"] = bool(
            source["rate_coverage"] >= 1 - 1e-12
            and result["rated_thermal_violations"] == 0
            and result["voltage_violations"] == 0
            and result["Pg_violations"] == 0
            and result["Qg_violations"] == 0
            and result["max_nodal_balance_error_MVA"] <= .10
        )

        return result, solved

    except Exception as ex:
        result["elapsed_s"] = (
            time.perf_counter() - start
        )
        result["error"] = repr(ex)[:600]
        return result, None


def main():
    protocol = {
        "pglib_version": "v23.07",
        "sizes": SIZES,
        "load_scales": LOAD_SCALES,
        "N1_per_case": N1_PER_CASE,
        "scope": (
            "Independent AC OPF benchmark; "
            "not integrated cyber-physical guarantee"
        )
    }

    frozen = OUT / "PROTOCOL.json"

    if (
        frozen.exists()
        and json.loads(frozen.read_text()) != protocol
    ):
        raise RuntimeError(
            "Frozen protocol mismatch"
        )

    frozen.write_text(
        json.dumps(protocol, indent=2)
    )

    records = []
    sources = []

    for n in SIZES:
        try:
            case, source = get_case(n)
            sources.append(source)

            print(
                "[SOURCE]",
                n,
                source["rated_branch_count"],
                "/",
                source["active_branch_count"],
                flush=True
            )

            solved_cases = {}

            for scale in LOAD_SCALES:
                row, solved = solve(
                    case, scale, source
                )

                records.append(row)

                if solved is not None:
                    solved_cases[scale] = solved

                print(
                    "[OPF]",
                    n,
                    scale,
                    "success=", row["success"],
                    "constraints=",
                    row.get("constraint_pass"),
                    "error=", row["error"],
                    flush=True
                )

            # Select highly loaded in-service lines
            # from the 100% benchmark solution.
            if 1.0 in solved_cases:
                branch = (
                    solved_cases[1.0]["branch"]
                )

                rated = (
                    (branch[:, BR_STATUS] > 0)
                    & (branch[:, RATE_A] > 0)
                    & (branch[:, RATE_A] < 9000)
                )

                utilization = np.zeros(
                    len(branch)
                )

                utilization[rated] = np.maximum(
                    np.hypot(
                        branch[rated, PF],
                        branch[rated, QF]
                    ),
                    np.hypot(
                        branch[rated, PT],
                        branch[rated, QT]
                    )
                ) / branch[rated, RATE_A]

                worst = np.argsort(
                    -utilization
                )[:N1_PER_CASE]

                for idx in worst:
                    row, _ = solve(
                        case, 1.0,
                        source, int(idx)
                    )

                    records.append(row)

                    print(
                        "[N-1]",
                        n,
                        int(idx),
                        "success=", row["success"],
                        "error=", row["error"],
                        flush=True
                    )

        except Exception as ex:
            traceback.print_exc()

            records.append({
                "case_size": n,
                "condition": "input_error",
                "success": False,
                "error": repr(ex)[:900]
            })

        pd.DataFrame(records).to_csv(
            OUT / "ALL_ACOPF_RUNS.csv",
            index=False
        )

    pd.DataFrame(sources).to_csv(
        OUT / "PGLIB_SOURCE_COVERAGE.csv",
        index=False
    )

    if not sources:
        raise RuntimeError(
            "No benchmark cases loaded. "
            "Check network access and sources."
        )

    results = pd.DataFrame(records)

    print(
        "\n=== AC OPF SUMMARY ==="
    )

    print(
        results[
            [
                "case_size",
                "condition",
                "success",
                "error"
            ]
        ].to_string(index=False)
    )

    (OUT / "EXPERIMENT_SCOPE.txt").write_text(
        "PGLib-OPF v23.07 benchmark ratings, "
        "not measured utility ratings.\n"
        "Independent AC OPF load and selected N-1 tests.\n"
        "Not integrated cyber-physical closed-loop safety.\n"
        "All failures retained in ALL_ACOPF_RUNS.csv.\n"
    )

    archive = OUT / "EXP56_EVIDENCE.zip"

    with ZipFile(
        archive,
        "w",
        ZIP_DEFLATED
    ) as z:
        for path in OUT.rglob("*"):
            if path.is_file() and path != archive:
                z.write(
                    path,
                    path.relative_to(ROOT)
                )

    with ZipFile(archive) as z:
        if z.testzip() is not None:
            raise RuntimeError(
                "Archive integrity failure"
            )

    print(
        "\nEXP56 COMPLETE",
        "\nOUTPUT:", OUT,
        flush=True
    )


if __name__ == "__main__":
    main()
