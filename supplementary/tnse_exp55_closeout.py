#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""TNSE Exp55: read-only QA of Exp54 and AC OPF evidence."""

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path.cwd()
R = ROOT / 'revision_experiments' / 'results'

SRC = (
    R / '54_graph_observation_delay' /
    'full_g5_s10_start17000'
)

OUT = R / '55_final_evidence_closeout'
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    'BA': '#2E6287',
    'ER': '#19827F',
    'WS': '#C18A49'
}

METHODS = [
    'CDP',
    'STATIC_RANK',
    'LOCAL_RANK',
    'CF_STATIC',
    'CF_LOCAL',
    'CF_LOCAL_NO_GUARD'
]

KEYS = [
    'topology', 'graph_seed',
    'pattern', 'budget', 'onset',
    'seed', 'mode', 'delay'
]

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
    'savefig.facecolor': 'white'
})


def writefig(fig, name):
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(
            OUT / f'{name}.{ext}',
            dpi=250,
            bbox_inches='tight',
            pad_inches=.22
        )
    plt.close(fig)


def require_csv(path):
    if not path.is_file():
        raise FileNotFoundError(
            f'Input missing: {path}'
        )
    return pd.read_csv(path)


def ci_by_graph(x, B=6000, seed=55100):
    """
    Resample complete graph realizations.
    Do not treat every propagation seed as
    an independent topology sample.
    """

    g = (
        x.groupby('graph_seed', sort=True)
        ['delta']
        .mean()
        .to_numpy(float)
    )

    if len(g) < 2:
        return (
            float('nan'),
            float('nan'),
            len(g),
            float('nan')
        )

    rng = np.random.default_rng(seed)

    ids = rng.integers(
        len(g),
        size=(B, len(g))
    )

    bs = g[ids].mean(axis=1)

    return (
        float(np.quantile(bs, .025)),
        float(np.quantile(bs, .975)),
        len(g),
        float((g > 1e-12).mean())
    )


def main():

    raw = require_csv(
        SRC / 'raw_outcomes.csv'
    )

    events = require_csv(
        SRC / 'stage_actions.csv'
    )

    required = KEYS + [
        'method', 'PAS',
        'terminal_I',
        'decision_compute_s'
    ]

    missing = set(required) - set(raw.columns)

    if missing:
        raise ValueError(
            f'Missing columns: {sorted(missing)}'
        )

    if raw.duplicated(
        KEYS + ['method']
    ).any():
        raise ValueError(
            'Duplicated case-method outcomes'
        )

    if not np.isfinite(
        pd.to_numeric(
            raw.PAS,
            errors='coerce'
        )
    ).all():
        raise ValueError(
            'Invalid PAS values'
        )

    wide = raw.pivot(
        index=KEYS,
        columns='method',
        values='PAS'
    ).sort_index()

    if (
        set(wide.columns) != set(METHODS)
        or wide.isna().any().any()
    ):
        raise ValueError(
            'Incomplete six-method pairing'
        )

    if len(wide) != 9600 or len(raw) != 57600:
        raise ValueError(
            'Incomplete Exp54: '
            f'{len(wide)} cases, '
            f'{len(raw)} records'
        )

    print(
        '[PASS] 9600 paired cases; '
        '57600 method records',
        flush=True
    )

    # ---------- Input integrity ----------

    source_files = [
        SRC / 'raw_outcomes.csv',
        SRC / 'stage_actions.csv',
        SRC / 'FROZEN_PROTOCOL.json'
    ]

    digests = {
        str(p.relative_to(ROOT)):
        hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in source_files
    }

    (
        OUT / 'INPUT_SHA256.json'
    ).write_text(
        json.dumps(digests, indent=2)
    )

    protocol = json.loads(
        (
            SRC / 'FROZEN_PROTOCOL.json'
        ).read_text()
    )

    (
        OUT / 'SOURCE_PROTOCOL.json'
    ).write_text(
        json.dumps(protocol, indent=2)
    )

    # ---------- All paired effects ----------

    contrasts = [
        (
            'CF_STATIC',
            'CF_LOCAL',
            'State-aware versus static CF'
        ),
        (
            'CDP',
            'CF_LOCAL',
            'State-aware CF versus CDP'
        ),
        (
            'CF_LOCAL_NO_GUARD',
            'CF_LOCAL',
            'Independent Guard effect'
        ),
        (
            'STATIC_RANK',
            'LOCAL_RANK',
            'State-aware direct ranking'
        )
    ]

    results = []

    for (topology, mode, delay), group in (
        wide.groupby(
            level=[
                'topology',
                'mode',
                'delay'
            ]
        )
    ):

        for a, b, desc in contrasts:

            delta = (
                group[a] - group[b]
            ).rename(
                'delta'
            ).reset_index()

            lo, hi, ng, fraction = (
                ci_by_graph(delta)
            )

            results.append({
                'topology': topology,
                'mode': mode,
                'delay': int(delay),
                'comparison': f'{b} vs {a}',
                'description': desc,
                'n_scenarios': len(delta),
                'n_graphs': ng,
                'gain_mean': float(
                    delta.delta.mean()
                ),
                'graph_CI95_low': lo,
                'graph_CI95_high': hi,
                'graph_fraction_positive':
                    fraction,
                'better_count': int(
                    (delta.delta > 1e-12).sum()
                ),
                'worse_count': int(
                    (delta.delta < -1e-12).sum()
                ),
                'ties': int(
                    (
                        abs(delta.delta)
                        <= 1e-12
                    ).sum()
                )
            })

    contrast_df = pd.DataFrame(results)

    contrast_df.to_csv(
        OUT / 'A_PAIRED_TOPOLOGY_CONTRASTS.csv',
        index=False
    )

    # ---------- Matched execution delay ----------

    delay_keys = [
        x for x in KEYS
        if x != 'delay'
    ]

    dw = raw.pivot(
        index=delay_keys + ['method'],
        columns='delay',
        values='PAS'
    )

    if (
        not {0, 3}.issubset(dw.columns)
        or dw.isna().any().any()
    ):
        raise ValueError(
            'Incomplete 0/3-step delay pairs'
        )

    dd = (
        dw[3] - dw[0]
    ).rename(
        'delta'
    ).reset_index()

    delay_rows = []

    for (t, m, method), group in dd.groupby(
        ['topology', 'mode', 'method']
    ):

        lo, hi, ng, fraction = (
            ci_by_graph(group)
        )

        delay_rows.append({
            'topology': t,
            'mode': m,
            'method': method,
            'n': len(group),
            'n_graphs': ng,
            'added_PAS_delay3_minus_delay0':
                float(group.delta.mean()),
            'graph_CI95_low': lo,
            'graph_CI95_high': hi
        })

    pd.DataFrame(
        delay_rows
    ).to_csv(
        OUT / 'B_MATCHED_EXECUTION_DELAY.csv',
        index=False
    )

    # ---------- Blind negative control ----------

    blind = wide.xs(
        'blind',
        level='mode'
    )

    spread = (
        blind.max(axis=1)
        - blind.min(axis=1)
    )

    blind_raw = raw[
        raw['mode'] == 'blind'
    ]

    infection = blind_raw.pivot(
        index=KEYS,
        columns='method',
        values='terminal_I'
    )

    infection_spread = (
        infection.max(axis=1)
        - infection.min(axis=1)
    )

    blind_events = events[
        events['mode'] == 'blind'
    ]

    event_wide = blind_events.pivot(
        index=KEYS + ['stage'],
        columns='method',
        values='protected_nodes'
    )

    if event_wide.isna().any().any():
        raise ValueError(
            'Missing protection actions'
        )

    action_diff = {
        m: float(
            (
                event_wide[m].astype(str)
                != event_wide['CDP'].astype(str)
            ).mean()
        )
        for m in METHODS
        if m != 'CDP'
    }

    infection_positive = float(
        (
            blind_raw[
                blind_raw.method == 'CDP'
            ].terminal_I > 0
        ).mean()
    )

    blind_audit = {
        'blind_paired_cases': len(blind),
        'different_PAS_fraction': float(
            (spread > 1e-12).mean()
        ),
        'different_terminal_I_fraction':
            float(
                (infection_spread > 0).mean()
            ),
        'CDP_terminal_infected_positive_fraction':
            infection_positive,
        'stage_action_difference_from_CDP':
            action_diff,
        'interpretation':
            'Identical loss despite different '
            'actions requires causal verification.'
    }

    (
        OUT / 'C_BLIND_NEGATIVE_CONTROL.json'
    ).write_text(
        json.dumps(
            blind_audit,
            indent=2
        )
    )

    # ---------- Controller timing ----------

    time_required = [
        'topology',
        'method',
        'mode',
        'delay',
        'decision_time_s'
    ]

    for column in time_required:
        if column not in events.columns:
            raise ValueError(
                'Missing timing field: '
                + column
            )

    timings = (
        events.groupby(
            [
                'topology',
                'method',
                'mode',
                'delay'
            ]
        )['decision_time_s']
        .agg(
            n='size',
            mean='mean',
            median='median',
            p95=lambda x: x.quantile(.95),
            max='max'
        )
        .reset_index()
    )

    timings.to_csv(
        OUT / 'D_CONTROLLER_TIMING_ONLY.csv',
        index=False
    )

    # ---------- AC OPF coverage audit ----------

    ac_path = (
        R /
        '45_acopf_corrective_dispatch' /
        'full_summary.csv'
    )

    ac_rows = []

    if ac_path.exists():

        ac = require_csv(ac_path)

        for item in ac.itertuples(index=False):

            usable = int(
                getattr(
                    item,
                    'branches_with_nonplaceholder_RATING_A',
                    0
                )
            )

            placeholder = int(
                getattr(
                    item,
                    'branches_with_placeholder_RATING_A',
                    0
                )
            )

            missing_rating = int(
                getattr(
                    item,
                    'branches_without_RATING_A',
                    0
                )
            )

            total = (
                usable
                + placeholder
                + missing_rating
            )

            ac_rows.append({
                'case_size': int(
                    item.case_size
                ),
                'profile': str(
                    item.profile
                ),
                'opf_success': bool(
                    item.opf_success
                ),
                'usable_rated_branches':
                    usable,
                'placeholder_ratings':
                    placeholder,
                'missing_ratings':
                    missing_rating,
                'thermal_coverage':
                    usable / total
                    if total else float('nan'),
                'new_voltage_violations':
                    getattr(
                        item,
                        'new_voltage_violations',
                        float('nan')
                    ),
                'new_Pg_violations':
                    getattr(
                        item,
                        'new_Pg_violations',
                        float('nan')
                    ),
                'new_Qg_violations':
                    getattr(
                        item,
                        'new_Qg_violations',
                        float('nan')
                    )
            })

    pd.DataFrame(ac_rows).to_csv(
        OUT / 'E_ACOPF_RATING_COVERAGE.csv',
        index=False
    )

    # ---------- Figure 1: topology forest ----------

    main = contrast_df[
        (
            contrast_df.comparison
            == 'CF_LOCAL vs CF_STATIC'
        )
        &
        (
            contrast_df['mode']
            != 'blind'
        )
    ].copy()

    fig, axes = plt.subplots(
        1, 2,
        figsize=(13.8, 6.0),
        sharey=True
    )

    labels = [
        f'{t} | {m}'
        for t in ('BA', 'ER', 'WS')
        for m in (
            'oracle',
            'partial80',
            'partial50'
        )
    ]

    for ax, delay in zip(
        axes, (0, 3)
    ):

        sub = (
            main[
                main.delay == delay
            ]
            .set_index(
                ['topology', 'mode']
            )
        )

        for i, title in enumerate(labels):

            topology, mode = title.split(
                ' | '
            )

            row = sub.loc[
                (topology, mode)
            ]

            value = row.gain_mean

            ax.errorbar(
                value, i,
                xerr=[
                    [
                        max(
                            0,
                            value-row.graph_CI95_low
                        )
                    ],
                    [
                        max(
                            0,
                            row.graph_CI95_high-value
                        )
                    ]
                ],
                fmt='o',
                capsize=3,
                lw=1.5,
                color=COLORS[topology],
                ms=5
            )

        ax.axvline(
            0,
            color='#A1B0B9',
            ls='--',
            lw=1
        )

        letter = (
            'a' if delay == 0
            else 'b'
        )

        ax.set_title(
            f'({letter}) Delay = {delay} steps',
            loc='left',
            pad=15
        )

        ax.set_xlabel(
            'PAS(CF_STATIC) - PAS(CF_LOCAL)'
        )

        ax.grid(
            axis='x',
            alpha=.17
        )

        ax.set_axisbelow(True)

    axes[0].set_yticks(
        range(len(labels)),
        labels
    )

    axes[0].invert_yaxis()

    fig.subplots_adjust(
        left=.18,
        right=.98,
        top=.89,
        bottom=.15,
        wspace=.27
    )

    writefig(
        fig,
        'Fig55A_cross_topology_effects'
    )

    # ---------- Figure 2: delay sensitivity ----------

    selected = contrast_df[
        (
            contrast_df.comparison
            == 'CF_LOCAL vs CDP'
        )
        &
        (
            contrast_df['mode'].isin(
                [
                    'oracle',
                    'partial80',
                    'partial50'
                ]
            )
        )
    ]

    fig, ax = plt.subplots(
        figsize=(10.7, 4.9)
    )

    loc = np.arange(3)

    for topology, color in COLORS.items():

        values = []

        for mode in [
            'oracle',
            'partial80',
            'partial50'
        ]:

            x = selected[
                (selected.topology == topology)
                &
                (selected['mode'] == mode)
            ]

            values.append([
                float(
                    x[
                        x.delay == j
                    ].gain_mean.iloc[0]
                )
                for j in (0, 3)
            ])

        values = np.asarray(values)

        shift = {
            'BA': -.16,
            'ER': 0,
            'WS': .16
        }[topology]

        ax.plot(
            loc+shift,
            values[:, 0],
            '-o',
            lw=1.5,
            ms=5,
            color=color,
            label=f'{topology}: delay 0'
        )

        ax.plot(
            loc+shift,
            values[:, 1],
            '--s',
            lw=1.25,
            ms=4,
            color=color,
            alpha=.75,
            label=f'{topology}: delay 3'
        )

    ax.axhline(
        0,
        color='#A1B0B9',
        ls=':',
        lw=1
    )

    ax.set_xticks(
        loc,
        [
            'Oracle',
            '80% recall',
            '50% recall'
        ]
    )

    ax.set_ylabel(
        'PAS(CDP) - PAS(CF_LOCAL)'
    )

    ax.set_title(
        'Cross-topology gain under delayed actuation',
        loc='left',
        pad=14
    )

    ax.grid(axis='y', alpha=.16)

    fig.legend(
        *ax.get_legend_handles_labels(),
        loc='lower center',
        ncol=3,
        frameon=False,
        bbox_to_anchor=(.5, .02),
        fontsize=8
    )

    fig.subplots_adjust(
        left=.10,
        right=.98,
        top=.88,
        bottom=.29
    )

    writefig(
        fig,
        'Fig55B_delay_sensitivity'
    )

    # ---------- Scientific closeout flags ----------

    flags = [
        'CF_LOCAL is a revised algorithm, '
        'not original DRAD-CF-MPC verbatim.',

        'Five graph realizations per topology '
        'limit population-level inference.',

        'Graph-bootstrap intervals are descriptive '
        'and do not replace confirmatory testing.',

        'Full thermal feasibility requires valid '
        'non-placeholder branch ratings.',

        'Controller timing excludes network, '
        'sensor and actuation latency.',

        'Check the causal action effect in '
        'Blind and other observation conditions.',

        'Guard may worsen some scenarios; '
        'report all signed paired contrasts.'
    ]

    if (
        ac_rows
        and max(
            r['usable_rated_branches']
            for r in ac_rows
        ) == 0
    ):
        flags.insert(
            0,
            'CRITICAL: Exp45 has no usable '
            'non-placeholder line thermal limits.'
        )

    if (
        blind_audit[
            'different_PAS_fraction'
        ] == 0
        and
        max(
            action_diff.values(),
            default=0
        ) > 0
    ):
        flags.insert(
            0,
            'IMPORTANT: different protection '
            'actions produced identical Blind PAS. '
            'Verify transmission/protection behavior.'
        )

    (
        OUT / 'F_REVIEWER_CLOSEOUT_FLAGS.txt'
    ).write_text(
        '\n'.join(
            '- ' + flag
            for flag in flags
        ) + '\n'
    )

    print(
        '\n===== AUDIT FLAGS =====',
        flush=True
    )

    for flag in flags:
        print('-', flag)

    print(
        '\n===== MAIN CROSS-TOPOLOGY EFFECTS =====',
        flush=True
    )

    relevant = contrast_df[
        (
            contrast_df['mode']
            == 'partial80'
        )
        &
        (
            contrast_df.delay == 0
        )
        &
        (
            contrast_df.comparison
            == 'CF_LOCAL vs CF_STATIC'
        )
    ]

    print(
        relevant[
            [
                'topology',
                'gain_mean',
                'graph_CI95_low',
                'graph_CI95_high'
            ]
        ].to_string(index=False)
    )

    # ---------- Package ----------

    target = OUT / 'EXP55_FINAL_AUDIT.zip'

    with ZipFile(
        target,
        'w',
        ZIP_DEFLATED,
        compresslevel=6
    ) as z:

        for p in sorted(OUT.iterdir()):

            if (
                p.is_file()
                and p != target
            ):
                z.write(
                    p,
                    p.name
                )

        z.write(
            Path(__file__),
            'SOURCE/tnse_exp55_closeout.py'
        )

    with ZipFile(target) as z:
        if z.testzip() is not None:
            raise RuntimeError(
                'ZIP integrity failure'
            )

    print(
        '\n[DONE] FINAL AUDIT:',
        target,
        flush=True
    )


if __name__ == '__main__':
    main()
