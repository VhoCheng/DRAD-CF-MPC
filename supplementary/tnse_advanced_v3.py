# -*- coding: utf-8 -*-
from pathlib import Path
import json
import traceback
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import networkx as nx

BASE = Path.cwd() / 'revision_experiments'
R = BASE / 'results'
W = BASE / 'topology_workspaces'
OUT = BASE / 'figures' / 'TNSE_ADVANCED_V3'
OUT.mkdir(parents=True, exist_ok=True)

NAVY = '#233B53'
BLUE = '#3D6F96'
TEAL = '#16858A'
AMBER = '#D58C3B'
RED = '#C75B56'
GRAY = '#95A6B3'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
    'axes.edgecolor': '#BDCAD3',
    'axes.linewidth': .7,
    'savefig.facecolor': 'white'
})


def csv(pattern):
    paths = sorted(R.glob(pattern))
    if len(paths) != 1:
        raise FileNotFoundError(
            f'{pattern}: found {len(paths)}'
        )
    return pd.read_csv(paths[0])


def title(ax, caption):
    ax.set_title(
        caption,
        loc='left',
        fontweight='bold',
        color=NAVY,
        pad=12
    )


def grid(ax, direction='y'):
    ax.grid(
        axis=direction,
        alpha=.20,
        lw=.7,
        color=GRAY
    )
    ax.set_axisbelow(True)


def out(fig, name, note=None):
    if note:
        fig.text(
            .50, .013, note,
            color=RED,
            ha='center',
            fontsize=8
        )

    fig.tight_layout(
        rect=[0, .035, 1, .98],
        pad=2.0,
        w_pad=2.6,
        h_pad=3
    )

    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(
            OUT / f'{name}.{ext}',
            dpi=260,
            bbox_inches='tight',
            pad_inches=.15
        )

    plt.close(fig)
    print('[OK]', name, flush=True)


def ci(ax, x, y, low, high, color=TEAL):
    ax.errorbar(
        x, y,
        xerr=[
            [max(0, x-low)],
            [max(0, high-x)]
        ],
        fmt='o',
        capsize=3,
        color=color,
        ecolor=color,
        markersize=5,
        lw=1.7,
        zorder=4
    )


# ========================================================
# FIGURE 5
# NETWORK TOPOLOGY + DISTRIBUTION + ROBUSTNESS
# ========================================================

def fig5():

    structure = csv(
        '07_topology_audit_*_structure.csv'
    )

    perf = csv(
        '07_topology_audit_*_summary.csv'
    ).set_index('topology').loc[['BA', 'WS', 'ER']]

    patterns = sorted(
        W.glob(
            '06_topology_full_*/BA_seed42/graph.json'
        )
    )

    if not patterns:
        raise FileNotFoundError(
            'topology graph.json not found'
        )

    folder = patterns[0].parent.parent

    fig, axs = plt.subplots(
        2, 3,
        figsize=(13.6, 7.6),
        gridspec_kw={
            'height_ratios': [1.35, 1]
        }
    )

    for i, t in enumerate(['BA', 'WS', 'ER']):

        ax = axs[0, i]

        meta = json.loads(
            (
                folder / f'{t}_seed42/graph.json'
            ).read_text()
        )

        graph = nx.Graph()

        graph.add_nodes_from(
            range(int(meta['nodes']))
        )

        graph.add_edges_from(
            meta['edge_list']
        )

        pos = nx.spring_layout(
            graph,
            seed=22,
            iterations=120
        )

        degrees = np.array([
            graph.degree(k)
            for k in graph.nodes()
        ])

        top = np.argsort(degrees)[
            -max(4, len(graph)//20):
        ]

        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            edge_color=GRAY,
            alpha=.26,
            width=.55
        )

        nx.draw_networkx_nodes(
            graph,
            pos,
            ax=ax,
            node_size=8+degrees*2,
            node_color=degrees,
            cmap='Blues',
            vmin=0,
            vmax=max(degrees),
            linewidths=0
        )

        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=top.tolist(),
            ax=ax,
            node_size=(
                8+degrees[top]*2.5
            ).tolist(),
            node_color=AMBER,
            linewidths=.4,
            edgecolors='white'
        )

        title(
            ax,
            f'({"abc"[i]}) {t} topology '
            '(118 communication nodes)'
        )

        ax.axis('off')

        st = structure[
            (structure.topology == t)
            &
            (structure.graph_seed == 42)
        ].iloc[0]

        ax.text(
            .5, -.03,
            f'{int(st.edges)} links | '
            f'clustering={st.average_clustering:.2f} | '
            f'path={st.average_shortest_path:.2f}',
            transform=ax.transAxes,
            ha='center',
            fontsize=8,
            color=NAVY
        )

        # Degree CCDF
        ax = axs[1, i]

        for seed in [42, 43, 44]:

            m = json.loads(
                (
                    folder /
                    f'{t}_seed{seed}/graph.json'
                ).read_text()
            )

            g = nx.Graph()

            g.add_nodes_from(
                range(int(m['nodes']))
            )

            g.add_edges_from(
                m['edge_list']
            )

            ds = np.array([
                x for _, x in g.degree()
            ])

            xx = np.arange(
                1, int(ds.max())+1
            )

            ccdf = np.array([
                (ds >= k).mean()
                for k in xx
            ])

            ax.step(
                xx,
                ccdf,
                where='post',
                color=TEAL if seed == 42 else GRAY,
                alpha=1 if seed == 42 else .5,
                lw=2 if seed == 42 else 1,
                label=f'seed {seed}'
            )

        ax.set(
            xscale='log',
            yscale='log',
            xlabel='Node degree k',
            ylabel='P(K ≥ k)',
            ylim=(.005, 1.15)
        )

        title(
            ax,
            f'({"def"[i]}) Degree distribution · {t}'
        )

        grid(ax, 'both')

        ax.legend(
            frameon=False,
            fontsize=7,
            ncol=3,
            loc='lower left'
        )

    out(
        fig,
        'Fig5A_topology_and_degree_distribution',
        'Archived BA/WS/ER instances; '
        'structures do not establish defense superiority.'
    )

    # ----------------------------------------------------
    # Figure 5B: effect and structural statistics
    # ----------------------------------------------------

    fig, axs = plt.subplots(
        1, 3,
        figsize=(13.2, 4.1)
    )

    ax = axs[0]

    for i, t in enumerate(['BA', 'WS', 'ER']):

        row = perf.loc[t]

        ci(
            ax,
            row.relative_gain_pct,
            i,
            row.bootstrap_pct_CI_low,
            row.bootstrap_pct_CI_high,
            [BLUE, AMBER, TEAL][i]
        )

    ax.set_yticks(
        range(3),
        ['BA', 'WS', 'ER']
    )

    ax.invert_yaxis()

    ax.axvline(
        0,
        color=RED,
        ls='--',
        lw=1
    )

    ax.set_xlabel(
        'Legacy EP change relative to CDP (%)'
    )

    title(
        ax,
        '(a) Paired topology effects (95% CI)'
    )

    grid(ax, 'x')

    # Clustering coefficient and path length
    ax = axs[1]

    avg = structure.groupby('topology')[
        [
            'average_clustering',
            'average_shortest_path'
        ]
    ].mean().loc[['BA', 'WS', 'ER']]

    xx = np.arange(3)

    ax.bar(
        xx-.18,
        avg.average_clustering,
        width=.36,
        color=BLUE,
        label='Clustering'
    )

    ax2 = ax.twinx()

    ax2.bar(
        xx+.18,
        avg.average_shortest_path,
        width=.36,
        color=AMBER,
        label='Path length'
    )

    ax.set_xticks(
        xx,
        ['BA', 'WS', 'ER']
    )

    ax.set_ylabel(
        'Clustering coefficient'
    )

    ax2.set_ylabel(
        'Mean shortest path'
    )

    title(
        ax,
        '(b) Structural heterogeneity'
    )

    ax.legend(
        frameon=False,
        loc='upper left',
        fontsize=7
    )

    ax2.legend(
        frameon=False,
        loc='upper right',
        fontsize=7
    )

    # Better/worse/tied outcomes
    ax = axs[2]

    xx = np.arange(3)
    bottom = np.zeros(3)

    for field, color, label in [
        ('better_settings', TEAL, 'Better'),
        ('worse_settings', RED, 'Worse'),
        ('tied_settings', GRAY, 'Tie')
    ]:

        v = perf[field].to_numpy()

        ax.bar(
            xx,
            v,
            bottom=bottom,
            color=color,
            width=.58,
            label=label
        )

        bottom += v

    ax.set_xticks(
        xx,
        ['BA', 'WS', 'ER']
    )

    ax.set_ylabel(
        'Number of paired settings'
    )

    title(
        ax,
        '(c) Legacy EP result breakdown'
    )

    ax.legend(
        frameon=False,
        ncol=3,
        fontsize=7
    )

    grid(ax)

    out(
        fig,
        'Fig5B_topology_effects_and_statistics',
        'Exp07 legacy EP formula remains under audit.'
    )


# ========================================================
# FIGURE 6
# SCALABILITY + RUNTIME + CONFIDENCE INTERVALS
# ========================================================

def fig6():

    d = csv(
        '35_37_final_evidence/'
        '37_serial_timing_summary.csv'
    ).sort_values('case_size')

    p = csv(
        '38_same_state_full/paired_statistics.csv'
    )

    f = csv(
        '35_37_final_evidence/'
        '35_physical_overview.csv'
    )

    fig, axs = plt.subplots(
        2, 2,
        figsize=(12.5, 7.25)
    )

    # Runtime
    ax = axs[0, 0]

    series = [
        ('mean_stage1', 'Stage 1 mean', BLUE, '-'),
        ('p95_stage1', 'Stage 1 P95', BLUE, '--'),
        ('mean_stage2', 'Stage 2 mean', AMBER, '-'),
        ('p95_stage2', 'Stage 2 P95', AMBER, '--')
    ]

    for field, label, color, ls in series:

        ax.plot(
            d.case_size,
            d[field],
            ls,
            marker='o',
            lw=1.8,
            color=color,
            label=label
        )

    ax.set_xticks(d.case_size)

    ax.set(
        xlabel='IEEE power buses',
        ylabel='Serial computation time (s)'
    )

    title(
        ax,
        '(a) Time complexity · empirical runtime'
    )

    ax.legend(
        frameon=False,
        ncol=2,
        fontsize=7
    )

    grid(ax)

    # Total wall time
    ax = axs[0, 1]

    b = ax.bar(
        d.case_size.astype(str),
        d.mean_three_method_wall,
        color=[
            GRAY, BLUE, TEAL, NAVY
        ],
        width=.62
    )

    for bar, v in zip(
        b,
        d.mean_three_method_wall
    ):

        ax.text(
            bar.get_x()+bar.get_width()/2,
            v+.4,
            f'{v:.1f}s',
            ha='center',
            fontsize=8
        )

    ax.set(
        xlabel='IEEE power buses',
        ylabel='Three-method batch time (s)'
    )

    ax.set_ylim(
        0,
        d.mean_three_method_wall.max()*1.18
    )

    title(
        ax,
        '(b) Serial batch wall time'
    )

    grid(ax)

    # Independent paired effect
    ax = axs[1, 0]

    sizes = [14, 57, 118, 300]

    for i, n in enumerate(sizes):

        part = p[
            (p.case_size == n)
            &
            (
                p.comparison.str.contains(
                    'vs CDP',
                    regex=False
                )
            )
        ]

        for row in part.itertuples():

            robust = 'Robust' in row.comparison

            ci(
                ax,
                row.mean_PAS_gain,
                i+(0.12 if robust else -.12),
                row.CI_low,
                row.CI_high,
                TEAL if robust else GRAY
            )

    ax.set_yticks(
        range(4),
        [f'IEEE {n}' for n in sizes]
    )

    ax.invert_yaxis()

    ax.axvline(
        0,
        color=RED,
        ls='--'
    )

    ax.plot(
        [], [], 'o',
        color=GRAY,
        label='Legacy'
    )

    ax.plot(
        [], [], 'o',
        color=TEAL,
        label='Robust'
    )

    ax.legend(
        frameon=False,
        fontsize=7,
        ncol=2
    )

    ax.set_xlabel(
        'CDP - method PAS (positive favors method)'
    )

    title(
        ax,
        '(c) Exp38 independently paired PAS effect'
    )

    grid(ax, 'x')

    # Unserved load
    ax = axs[1, 1]

    for method, col in [
        ('CDP', GRAY),
        ('Legacy-K5', AMBER),
        ('Confirm15-Robust', TEAL)
    ]:

        sub = f[
            f.variant == method
        ].sort_values('case_size')

        ax.plot(
            sub.case_size,
            100*sub.mean_unserved,
            '-o',
            color=col,
            label=method,
            lw=1.8
        )

    ax.set_xticks(sizes)

    ax.set(
        xlabel='IEEE power buses',
        ylabel='Unserved-load proxy (%)'
    )

    grid(ax)

    ax.legend(
        frameon=False,
        fontsize=7
    )

    title(
        ax,
        '(d) Exp35 physical-loss proxy by scale'
    )

    out(
        fig,
        'Fig6_scalability_and_paired_effects',
        'Exp37 time excludes measured communication '
        'and field-actuation delay.'
    )


# ========================================================
# FIGURE 7
# AC OPF + PHYSICAL FEASIBILITY + SAFETY BOUNDARY
# ========================================================

def fig7():

    d = csv(
        '45_acopf_corrective_dispatch/full_summary.csv'
    )

    r = csv(
        '46_final_closeout/'
        '46_opf_recovery_summary.csv'
    ).sort_values('case_size')

    sizes = [14, 57, 118, 300]

    profiles = [
        'Morning',
        'Solar-Peak',
        'Evening-Peak'
    ]

    fig, axs = plt.subplots(
        2, 2,
        figsize=(12.8, 7.2)
    )

    # AC OPF status
    ax = axs[0, 0]

    status = d.pivot(
        index='case_size',
        columns='profile',
        values='opf_success'
    ).reindex(
        index=sizes,
        columns=profiles
    )

    ax.imshow(
        status.astype(float),
        aspect='auto',
        vmin=0,
        vmax=1,
        cmap=colors.ListedColormap([
            '#F3DCD9',
            '#D9ECE9'
        ])
    )

    ax.set_xticks(
        range(3),
        profiles
    )

    ax.set_yticks(
        range(4),
        [f'IEEE {n}' for n in sizes]
    )

    for i in range(4):
        for j in range(3):

            ax.text(
                j, i,
                'SOLVED' if status.iloc[i, j] else 'FAILED',
                ha='center',
                va='center',
                fontsize=8,
                fontweight='bold',
                color=NAVY
            )

    title(
        ax,
        '(a) Corrective AC OPF solver status'
    )

    # Redispatch burden
    d['redispatch_pct'] = (
        100 *
        d.generator_Pg_redispatch_L1_MW /
        d.total_net_load_MW
    )

    pivot = d.pivot(
        index='case_size',
        columns='profile',
        values='redispatch_pct'
    ).reindex(
        index=sizes,
        columns=profiles
    )

    ax = axs[0, 1]

    vals = pivot.to_numpy(float)

    im = ax.imshow(
        vals,
        aspect='auto',
        cmap='YlGnBu',
        vmin=0
    )

    ax.set_xticks(
        range(3),
        profiles
    )

    ax.set_yticks(
        range(4),
        [f'IEEE {n}' for n in sizes]
    )

    for i in range(4):
        for j in range(3):

            ax.text(
                j, i,
                f'{vals[i, j]:.1f}%'
                if np.isfinite(vals[i, j])
                else 'N/A',
                ha='center',
                va='center',
                fontsize=8
            )

    fig.colorbar(
        im,
        ax=ax,
        fraction=.043,
        pad=.02,
        label='Redispatch / load (%)'
    )

    title(
        ax,
        '(b) Corrective redispatch burden'
    )

    # Thermal rating completeness
    s = d.drop_duplicates(
        'case_size'
    ).set_index(
        'case_size'
    ).reindex(sizes)

    counts = [
        (
            'branches_with_placeholder_RATING_A',
            AMBER,
            'Placeholder'
        ),
        (
            'branches_with_nonplaceholder_RATING_A',
            TEAL,
            'Usable'
        ),
        (
            'branches_without_RATING_A',
            GRAY,
            'Missing'
        )
    ]

    ax = axs[1, 0]

    bottom = np.zeros(4)

    for col, color, label in counts:

        values = s[col].to_numpy(float)

        ax.bar(
            range(4),
            values,
            width=.60,
            bottom=bottom,
            color=color,
            label=label
        )

        bottom += values

    ax.set_xticks(
        range(4),
        [str(n) for n in sizes]
    )

    ax.set(
        xlabel='IEEE case',
        ylabel='Transmission branches'
    )

    ax.legend(
        frameon=False,
        ncol=3,
        fontsize=7
    )

    title(
        ax,
        '(c) Thermal rating coverage'
    )

    grid(ax)

    # Load curtailment
    ax = axs[1, 1]

    for j, row in enumerate(r.itertuples()):

        pct = (
            row.load_curtailment_fraction * 100
        )

        color = AMBER if j == 0 else RED

        ax.hlines(
            j,
            0,
            pct,
            color=color,
            lw=5
        )

        ax.scatter(
            pct,
            j,
            s=85,
            color=color,
            zorder=5
        )

        ax.text(
            pct+.35,
            j,
            f'{pct:.2f}% / '
            f'{row.curtailed_MW:,.1f} MW',
            va='center',
            fontsize=9
        )

    ax.set_xlim(0, 15.5)

    ax.set_ylim(
        -.6,
        len(r)-.4
    )

    ax.set_yticks(
        range(len(r)),
        [
            f'IEEE {n} evening'
            for n in r.case_size
        ]
    )

    ax.set_xlabel(
        'Tested load curtailment (%)'
    )

    title(
        ax,
        '(d) Exp46 recoverable feasible points'
    )

    grid(ax, 'x')

    out(
        fig,
        'Fig7_ACOPF_safety_scope_and_recovery',
        'No independently usable branch thermal ratings; '
        'AC OPF convergence is not a full safety proof.'
    )


# ========================================================
# FIGURE 4
# EXP47 PAIRED ABLATION
# ========================================================

def exp47():

    d = csv(
        '47_independent_closed_loop/'
        'full/paired_statistics.csv'
    )

    d = d[
        d.A == 'FIXED_EP'
    ].copy()

    fig, axs = plt.subplots(
        1, 2,
        figsize=(12, 4.8),
        gridspec_kw={
            'width_ratios': [1.4, 1]
        }
    )

    ax = axs[0]

    for i, row in enumerate(d.itertuples()):

        color = (
            TEAL if row.CI_low > 0
            else RED if row.CI_high < 0
            else BLUE
        )

        ci(
            ax,
            row.gain_B_minus_A,
            i,
            row.CI_low,
            row.CI_high,
            color
        )

    ax.set_yticks(
        range(len(d)),
        d.B.str.replace(
            'FIXED_EP_',
            '',
            regex=False
        )
    )

    ax.invert_yaxis()

    ax.axvline(
        0,
        color=RED,
        ls='--'
    )

    ax.set_xlabel(
        'PAS(B) - PAS(FIXED_EP), '
        'positive favors FIXED_EP'
    )

    title(
        ax,
        '(a) Exp47: paired effect and 95% CI'
    )

    grid(ax, 'x')

    # Absolute metric
    ax = axs[1]

    summ = csv(
        '47_independent_closed_loop/'
        'full/by_setting_summary.csv'
    )

    agg = summ.groupby(
        'method'
    ).mean(
        numeric_only=True
    ).sort_values('mean_PAS')

    y = np.arange(len(agg))

    ax.scatter(
        agg.mean_PAS,
        y,
        s=32+25*agg.average_switches,
        color=BLUE
    )

    ax.hlines(
        y,
        0,
        agg.mean_PAS,
        color='#C9D6DF',
        lw=2
    )

    ax.set_yticks(
        y,
        [
            s.replace('FIXED_EP_', '')
            for s in agg.index
        ]
    )

    ax.invert_yaxis()

    ax.set_xlim(
        0,
        agg.mean_PAS.max()*1.22
    )

    ax.set_xlabel(
        'Mean PAS loss (lower better)'
    )

    title(
        ax,
        '(b) PAS and switching frequency'
    )

    grid(ax, 'x')

    ax.tick_params(
        axis='y',
        labelsize=7
    )

    out(
        fig,
        'Fig4_Exp47_independent_paired_audit',
        'Exp47 policy mapping requires final '
        'methodology verification before citing.'
    )


# ========================================================
# FIGURE 3
# PARAMETER SENSITIVITY
# ========================================================

def sensitivity():

    d = csv(
        '02_safeguard_full_*_parameter_aggregate.csv'
    )

    fig, axs = plt.subplots(
        1, 3,
        figsize=(12.6, 3.5)
    )

    metrics = [
        'mean_EP_gain_pct',
        'mean_switch_rate',
        'mean_negative_PAS_rate'
    ]

    captions = [
        '(a) Legacy EP gain (%)',
        '(b) Switching frequency',
        '(c) Negative PAS-change rate'
    ]

    for ax, metric, caption in zip(
        axs,
        metrics,
        captions
    ):

        pivot = d.pivot(
            index='kappa',
            columns='gamma',
            values=metric
        ).sort_index().sort_index(axis=1)

        im = ax.imshow(
            pivot.values,
            aspect='auto',
            cmap=(
                'RdBu_r'
                if metric == 'mean_EP_gain_pct'
                else 'YlGnBu'
            )
        )

        ax.set_xticks(
            range(len(pivot.columns)),
            [
                f'{v:g}'
                for v in pivot.columns
            ]
        )

        ax.set_yticks(
            range(len(pivot.index)),
            [
                f'{v:g}'
                for v in pivot.index
            ]
        )

        ax.set_xlabel('gamma')
        ax.set_ylabel('kappa')

        title(ax, caption)

        fig.colorbar(
            im,
            ax=ax,
            fraction=.043,
            pad=.03
        )

    out(
        fig,
        'Fig3_Exp02_sensitivity_dashboard',
        'Legacy EP accounting remains under audit.'
    )


# ========================================================
# MAIN
# ========================================================

JOBS = [
    ('Fig5', fig5),
    ('Fig6', fig6),
    ('Fig7', fig7),
    ('Exp47', exp47),
    ('Exp02', sensitivity)
]

if __name__ == '__main__':

    errors = []

    for k, fn in JOBS:

        try:
            print('[RUN]', k, flush=True)
            fn()

        except Exception as e:
            errors.append((k, str(e)))
            traceback.print_exc()

    print('\nOUTPUT:', OUT)
    print('FAILED:', errors)

    if errors:
        raise SystemExit(1)

    print('ALL FIGURES GENERATED SUCCESSFULLY')
