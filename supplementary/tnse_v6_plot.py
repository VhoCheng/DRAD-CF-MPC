from pathlib import Path
import json, sys, traceback, csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm, ListedColormap
from scipy.stats import gaussian_kde
import networkx as nx

ROOT = Path.cwd()
E = ROOT / 'revision_experiments'
R = E / 'results'
W = E / 'topology_workspaces'
OUT = E / 'figures' / 'TNSE_V6_NO_OVERLAP'
REAL = OUT / '01_REAL_EVIDENCE'
SIM = OUT / '02_SYNTHETIC_TUTORIAL'
REAL.mkdir(parents=True, exist_ok=True)
SIM.mkdir(parents=True, exist_ok=True)

BLUE = '#315E80'
TEAL = '#188584'
ORANGE = '#CC8A3D'
RED = '#BE5E64'
INK = '#263B4D'
GRAY = '#899AAA'
PINK = '#B66B8B'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'axes.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#B7C3CD',
    'axes.linewidth': .7,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'grid.alpha': .16,
    'grid.linewidth': .6
})

REPORT = []

def read(pattern):
    files = sorted(R.glob(pattern))
    assert len(files) == 1, (
        f'Expected one CSV: {pattern}, found {len(files)}'
    )
    d = pd.read_csv(files[0])
    print('DATA:', files[0].relative_to(ROOT))
    return d

def label(ax, text):
    ax.set_title(text, loc='left', pad=13, color=INK)

def grid(ax, axis='y'):
    ax.grid(axis=axis, color='#A9B8C2')
    ax.set_axisbelow(True)

def save(fig, folder, name, origin):
    if folder == SIM:
        fig.text(
            .5, .5,
            'SYNTHETIC - NOT EXPERIMENTAL DATA',
            ha='center', va='center',
            color=RED, fontsize=24,
            alpha=.115, rotation=25,
            weight='bold', zorder=20
        )
        fig.text(
            .5, -.095,
            'SIMULATED FOR VISUALIZATION PRACTICE'
            ' - NOT FOR PAPER OR RESPONSE LETTER',
            ha='center', fontsize=8,
            color=RED
        )

    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(
            folder / f'{name}.{ext}',
            dpi=240,
            bbox_inches='tight',
            pad_inches=.16
        )

    plt.close(fig)
    REPORT.append([folder.name, name, origin])
    print('[OK]', name, flush=True)

def raincloud(ax, groups, names, colors, seed=42):
    rng = np.random.default_rng(seed)

    for i, (vec, nm, color) in enumerate(
        zip(groups, names, colors)
    ):
        v = np.asarray(vec, float)
        v = v[np.isfinite(v)]
        assert len(v) > 2

        xx = np.linspace(
            v.min()-.07, v.max()+.07, 250
        )

        if np.std(v) > 1e-12:
            density = gaussian_kde(v)(xx)
        else:
            density = np.exp(
                -.5*((xx-v.mean())/.01)**2
            )

        density = density / max(
            density.max(), 1e-10
        ) * .27

        ax.fill_between(
            xx, i+.03, i+.03+density,
            alpha=.32, color=color, lw=0
        )
        ax.plot(
            xx, i+.03+density,
            color=color, lw=1.2
        )

        n = min(len(v), 85)
        chosen = rng.choice(
            len(v), size=n, replace=False
        )

        yy = i-.18+rng.uniform(
            -.09, .08, size=n
        )

        ax.scatter(
            v[chosen], yy, s=8,
            color=color, alpha=.5,
            rasterized=True, lw=0
        )

        q = np.quantile(
            v, [.05, .25, .5, .75, .95]
        )

        ax.plot(
            [q[0], q[4]],
            [i-.035, i-.035],
            lw=1.5, color=INK
        )
        ax.plot(
            [q[1], q[3]],
            [i-.035, i-.035],
            lw=5, color=color,
            solid_capstyle='round'
        )
        ax.scatter(
            [q[2]], [i-.035],
            s=12, color='white',
            edgecolor=INK, zorder=7
        )

    ax.set_yticks(
        range(len(names)), names
    )
    ax.set_ylim(-.42, len(names)-.45)
    ax.invert_yaxis()
    grid(ax, 'x')

def heat(ax, df, text, unit='',
         center=None, precision=2):
    z = df.to_numpy(float)
    finite = z[np.isfinite(z)]
    assert len(finite)

    kwargs = {}

    if center is not None:
        delta = max(
            abs(finite.min()-center),
            abs(finite.max()-center),
            .001
        )
        kwargs['norm'] = TwoSlopeNorm(
            vcenter=center,
            vmin=center-delta,
            vmax=center+delta
        )

    im = ax.imshow(
        z, aspect='auto',
        cmap=(
            'RdBu_r' if center is not None
            else 'YlGnBu'
        ),
        **kwargs
    )

    ax.set_xticks(
        np.arange(len(df.columns)),
        [str(s).replace('_',' ')
         for s in df.columns]
    )
    ax.set_yticks(
        np.arange(len(df.index)),
        [str(s).replace('_',' ')
         for s in df.index]
    )
    ax.tick_params(length=0)

    if len(df.columns) > 6:
        ax.tick_params(
            axis='x', rotation=28
        )

    if z.size <= 36:
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                if np.isfinite(z[i,j]):
                    color = (
                        'white'
                        if im.norm(z[i,j]) > .75
                        or im.norm(z[i,j]) < .14
                        else INK
                    )
                    ax.text(
                        j, i,
                        f'{z[i,j]:.{precision}f}',
                        ha='center', va='center',
                        fontsize=7, color=color
                    )

    label(ax, text)

    if unit:
        ax.figure.colorbar(
            im, ax=ax, shrink=.83,
            pad=.025, label=unit
        )

    return im


# ===================================================
# R01: TRUE RAINCLOUD + ECDF
# ===================================================

def real1():
    d = read(
        '47_independent_closed_loop/'
        'full/paired_outcomes.csv'
    )

    ix = ['pattern','budget','onset','seed']
    w = d.pivot_table(
        index=ix, columns='method',
        values='PAS'
    )

    methods = [
        'CDP', 'FIXED_EP',
        'FIXED_EP_NO_GUARD',
        'FIXED_EP_HEURISTIC_NO_GUARD'
    ]
    w = w[methods].dropna()

    fig, (a,b) = plt.subplots(
        1, 2, figsize=(13.8,4.6),
        layout='constrained',
        gridspec_kw={
            'width_ratios':[1.17,1]
        }
    )

    raincloud(
        a, [w[x] for x in methods],
        [
            'CDP', 'Fixed EP',
            'No Guard', 'Heuristic No Guard'
        ],
        [GRAY,TEAL,ORANGE,RED]
    )
    a.set_xlabel(
        'Terminal PAS (lower is better)'
    )
    label(
        a,
        '(a) Raincloud: matched distributions'
    )

    for method,c in zip(
        methods,[GRAY,TEAL,ORANGE,RED]
    ):
        x = np.sort(w[method].to_numpy())
        y = np.arange(1,len(x)+1)/len(x)

        b.step(
            x, y, where='post',
            color=c, lw=1.8,
            label=method.replace(
                'FIXED_EP_',''
            ).replace('FIXED_EP','Full')
        )

    b.set_xlabel('Terminal PAS')
    b.set_ylabel(
        'Empirical cumulative probability'
    )
    label(
        b, '(b) ECDF: central and tail behavior'
    )
    grid(b,'both')

    fig.legend(
        *b.get_legend_handles_labels(),
        loc='outside lower center',
        ncol=4, frameon=False, fontsize=8
    )

    save(
        fig, REAL,
        'R01_raincloud_and_ECDF',
        'Exp47, 400 paired scenarios'
    )


# ===================================================
# R02: PAIRED FOREST AND SLOPE GRAPH
# ===================================================

def real2():
    d = read(
        '47_independent_closed_loop/'
        'full/paired_statistics.csv'
    )
    d = d[d.A=='FIXED_EP'].copy()

    s = read(
        '47_independent_closed_loop/'
        'full/paired_outcomes.csv'
    )

    ix = ['pattern','budget','onset','seed']
    w = s.pivot_table(
        index=ix, columns='method',
        values='PAS'
    )

    fig,(a,b) = plt.subplots(
        1,2,figsize=(13,4.8),
        layout='constrained',
        gridspec_kw={'width_ratios':[1.25,1]}
    )

    for i,row in enumerate(d.itertuples()):
        x = row.gain_B_minus_A
        lo,hi = row.CI_low,row.CI_high

        color = (
            TEAL if lo>0
            else RED if hi<0
            else BLUE
        )

        a.errorbar(
            x,i,
            xerr=[
                [max(0,x-lo)],
                [max(0,hi-x)]
            ],
            fmt='o', color=color,
            capsize=3, elinewidth=1.5
        )

    a.set_yticks(
        range(len(d)),
        d.B.str.replace(
            'FIXED_EP_','',regex=False
        ).str.replace(
            'FIXED_EP','Full',regex=False
        )
    )
    a.invert_yaxis()
    a.axvline(0,color=GRAY,ls='--')
    a.set_xlabel(
        'PAS(comparator) - PAS(Fixed EP)'
    )
    label(
        a,'(a) Matched effect with 95% CI'
    )
    grid(a,'x')

    good = w[['CDP','FIXED_EP']].dropna()
    rng = np.random.default_rng(10)
    chosen = rng.choice(
        len(good),
        min(70,len(good)),
        replace=False
    )
    subset = good.iloc[chosen]

    for v in subset.itertuples(index=False):
        b.plot(
            [0,1],[v[0],v[1]],
            lw=.7,
            color=(
                TEAL if v[1]<v[0] else RED
            ),
            alpha=.29
        )

    b.scatter(
        np.zeros(len(subset)),
        subset.CDP, s=6,
        color=GRAY,alpha=.5
    )
    b.scatter(
        np.ones(len(subset)),
        subset.FIXED_EP,s=6,
        color=TEAL,alpha=.5
    )

    b.plot(
        [0,1],
        [
            good.CDP.mean(),
            good.FIXED_EP.mean()
        ],
        '-o', lw=3,color=INK,
        ms=6,zorder=5
    )

    b.set_xlim(-.2,1.2)
    b.set_xticks(
        [0,1],['CDP','Fixed EP']
    )
    b.set_ylabel('PAS per matched scenario')
    label(
        b,'(b) Paired scenarios (70 of 400)'
    )
    grid(b)

    save(
        fig,REAL,
        'R02_paired_forest_and_slope',
        'Exp47 paired statistics'
    )


# ===================================================
# R03: THREE-PANEL PARAMETER HEATMAP
# ===================================================

def real3():
    d = read(
        '02_safeguard_full_*'
        '_parameter_aggregate.csv'
    )

    fields = [
        (
            'mean_EP_gain_pct',
            '(a) Legacy EP gain (%)',0
        ),
        (
            'mean_switch_rate',
            '(b) Switching rate',None
        ),
        (
            'mean_negative_PAS_rate',
            '(c) Negative-gain rate',None
        )
    ]

    fig,axs = plt.subplots(
        1,3,figsize=(14,4.5),
        layout='constrained'
    )

    for a,(col,t,mid) in zip(axs,fields):
        p = d.pivot(
            index='kappa',
            columns='gamma',
            values=col
        ).sort_index(ascending=False)

        heat(
            a,p,t,
            col.replace('_',' '),
            center=mid,
            precision=(
                1 if col=='mean_EP_gain_pct'
                else 2
            )
        )
        a.set_xlabel('gamma')
        a.set_ylabel('kappa')

    save(
        fig,REAL,
        'R03_parameter_triptych',
        'Exp02 archived EP sensitivity'
    )


# ===================================================
# R04: MEAN VS CVaR95 AND GUARD BENEFIT
# ===================================================

def real4():
    d = read(
        '42_final_guard_audit/'
        '42_tail_risk_and_regret.csv'
    )
    g = read(
        '42_final_guard_audit/'
        '42_guard_tradeoff_by_condition.csv'
    )

    fig,(a,b) = plt.subplots(
        1,2,figsize=(13.9,5.7),
        layout='constrained',
        gridspec_kw={
            'width_ratios':[1,1.23]
        }
    )

    for method,c in [
        ('CDP',GRAY),
        ('FULL',TEAL),
        ('NO_GUARD',ORANGE)
    ]:
        q = d[
            d.method==method
        ].sort_values('case_size')

        a.plot(
            q.mean_PAS,
            q.CVaR95_PAS,
            '-o',color=c,
            lw=1.35,ms=5,
            label=method
        )

        for row in q.itertuples():
            a.annotate(
                str(row.case_size),
                (
                    row.mean_PAS,
                    row.CVaR95_PAS
                ),
                xytext=(4,3),
                textcoords='offset points',
                fontsize=7
            )

    a.set_xlabel('Mean PAS (lower better)')
    a.set_ylabel(
        'CVaR95 PAS (lower better)'
    )
    label(a,'(a) Mean-tail risk tradeoff')
    grid(a,'both')
    a.legend(
        frameon=False,
        loc='upper left',fontsize=8
    )

    p = g.pivot(
        index='condition',
        columns='case_size',
        values='guard_benefit'
    ).reindex(
        columns=[14,57,118,300]
    )

    heat(
        b,p,
        '(b) Guard gain by stress condition',
        'No Guard PAS - Full PAS',
        center=0,precision=3
    )

    b.set_xlabel('IEEE bus count')
    b.tick_params(axis='y',labelsize=7)

    save(
        fig,REAL,
        'R04_guard_mean_tail_tradeoff',
        'Exp42 mean and tail risk'
    )


# ===================================================
# R05: NETWORK GRAPHS + DEGREE CCDF
# ===================================================

def real5():
    files = sorted(
        W.glob(
            '06_topology_full_*/'
            'BA_seed42/graph.json'
        )
    )
    assert len(files)==1, (
        'Topology graph files missing'
    )
    base = files[0].parent.parent

    fig = plt.figure(
        figsize=(13.5,6.7),
        layout='constrained'
    )
    gs = GridSpec(
        2,3,figure=fig,
        height_ratios=[1.6,1],
        hspace=.05
    )

    for j,t in enumerate(['BA','ER','WS']):
        obj = json.loads(
            (
                base/f'{t}_seed42/graph.json'
            ).read_text()
        )

        g = nx.Graph()
        g.add_nodes_from(
            range(obj['nodes'])
        )
        g.add_edges_from(obj['edge_list'])

        a = fig.add_subplot(gs[0,j])
        pos = nx.spring_layout(
            g,seed=10,iterations=85
        )

        deg = np.array([
            g.degree(i)
            for i in g.nodes()
        ])

        nx.draw_networkx_edges(
            g,pos,ax=a,
            alpha=.28,
            width=.43,
            edge_color=GRAY
        )
        nx.draw_networkx_nodes(
            g,pos,ax=a,
            node_color=deg,
            cmap='Blues',
            node_size=10+1.7*deg,
            linewidths=0
        )

        a.set_aspect('equal')
        a.axis('off')
        label(
            a,
            f'({chr(97+j)}) {t} | '
            f'N={len(g)}, E={g.number_of_edges()}'
        )

        b = fig.add_subplot(gs[1,j])
        ds = np.sort(deg)
        xx = np.arange(1,ds.max()+1)
        yy = np.array([
            (ds>=n).mean()
            for n in xx
        ])

        b.step(
            xx,yy,where='post',
            color=TEAL,lw=1.9
        )
        b.set_xscale('log')
        b.set_yscale('log')
        b.set_xlabel('Degree k')
        b.set_ylabel('P(K >= k)')
        b.set_ylim(.005,1.1)
        label(
            b,
            f'({chr(100+j)}) Degree-tail CCDF'
        )
        grid(b,'both')

    save(
        fig,REAL,
        'R05_topology_and_CCDF',
        'Exp06 archived simulated graphs'
    )


# ===================================================
# R06: SCALE AND COMPUTATION
# ===================================================

def real6():
    t = read(
        '35_37_final_evidence/'
        '37_serial_timing_summary.csv'
    ).sort_values('case_size')

    fig,(a,b) = plt.subplots(
        1,2,figsize=(12,4.3),
        layout='constrained'
    )

    series = [
        (
            'mean_stage1',
            'Stage 1 mean',BLUE,'-'
        ),
        (
            'p95_stage1',
            'Stage 1 P95',BLUE,'--'
        ),
        (
            'mean_stage2',
            'Stage 2 mean',ORANGE,'-'
        ),
        (
            'p95_stage2',
            'Stage 2 P95',ORANGE,'--'
        )
    ]

    for col,desc,c,ls in series:
        a.plot(
            t.case_size,t[col],
            marker='o',
            lw=1.6,ms=4,
            color=c,ls=ls,
            label=desc
        )

    a.set_xticks(t.case_size)
    a.set_xlabel('IEEE buses')
    a.set_ylabel(
        'Serial calculation time (s)'
    )
    label(
        a,'(a) Controller computation profile'
    )
    grid(a)

    b.bar(
        t.case_size.astype(str),
        t.mean_three_method_wall,
        color=[GRAY,BLUE,TEAL,INK],
        width=.62
    )
    b.set_xlabel('IEEE buses')
    b.set_ylabel(
        'Three-method batch time (s)'
    )
    b.set_ylim(
        0,
        t.mean_three_method_wall.max()*1.22
    )
    label(
        b,'(b) Batch wall-clock cost'
    )
    grid(b)

    fig.legend(
        *a.get_legend_handles_labels(),
        loc='outside lower center',
        ncol=4,frameon=False,
        fontsize=8
    )

    save(
        fig,REAL,
        'R06_runtime_scalability',
        'Exp37 computation, not full latency'
    )


# ===================================================
# R07: AC OPF + THERMAL COVERAGE
# ===================================================

def real7():
    d = read(
        '45_acopf_corrective_dispatch/'
        'full_summary.csv'
    )
    r = read(
        '46_final_closeout/'
        '46_opf_recovery_summary.csv'
    )

    sizes = [14,57,118,300]
    profiles = [
        'Morning',
        'Solar-Peak',
        'Evening-Peak'
    ]

    fig,axs = plt.subplots(
        2,2,figsize=(12.9,7.3),
        layout='constrained'
    )

    a = axs[0,0]
    p = d.pivot(
        index='case_size',
        columns='profile',
        values='opf_success'
    ).reindex(
        index=sizes,
        columns=profiles
    )

    a.imshow(
        p.to_numpy(float),
        aspect='auto',
        cmap=ListedColormap([
            '#F2DCDD','#D3EBE6'
        ]),
        vmin=0,vmax=1
    )
    a.set_xticks(
        range(3),
        [
            'Morning',
            'Solar peak',
            'Evening peak'
        ]
    )
    a.set_yticks(
        range(4),
        [f'IEEE {n}' for n in sizes]
    )
    label(a,'(a) OPF convergence')

    for i in range(4):
        for j in range(3):
            a.text(
                j,i,
                (
                    'SOLVED'
                    if bool(p.iloc[i,j])
                    else 'FAILED'
                ),
                ha='center',va='center',
                fontsize=8,color=INK
            )

    a = axs[0,1]
    p = d.pivot(
        index='case_size',
        columns='profile',
        values='new_voltage_violations'
    ).reindex(
        index=sizes,
        columns=profiles
    )

    heat(
        a,p,
        '(b) Checked voltage violations',
        unit='Violation count',
        precision=0
    )
    a.set_xlabel('Operating profile')

    a = axs[1,0]
    q = d.drop_duplicates(
        'case_size'
    ).set_index(
        'case_size'
    ).reindex(sizes)

    xx = np.arange(4)
    bottom = np.zeros(4)

    sources = [
        (
            'branches_with_placeholder_RATING_A',
            ORANGE,'Placeholder'
        ),
        (
            'branches_with_nonplaceholder_RATING_A',
            TEAL,'Usable'
        ),
        (
            'branches_without_RATING_A',
            GRAY,'Missing'
        )
    ]

    for key,c,desc in sources:
        v = q[key].to_numpy(float)
        a.bar(
            xx,v,bottom=bottom,
            width=.6,color=c,label=desc
        )
        bottom += v

    a.set_xticks(
        xx,[str(n) for n in sizes]
    )
    a.set_xlabel('IEEE bus count')
    a.set_ylabel('Transmission branches')
    label(
        a,'(c) Thermal-limit availability'
    )
    grid(a)

    a = axs[1,1]
    r = r.sort_values('case_size')

    for i,row in enumerate(r.itertuples()):
        pct = (
            row.load_curtailment_fraction*100
        )
        col = [ORANGE,RED][i]

        a.hlines(
            i,0,pct,lw=4,
            colors=col
        )
        a.scatter(
            [pct],[i],
            color=col,s=55,zorder=4
        )
        a.text(
            pct+.4,i,
            f'{pct:.2f}% / '
            f'{row.curtailed_MW:,.0f} MW',
            va='center',fontsize=8
        )

    a.set_yticks(
        range(len(r)),
        [
            f'IEEE {n}'
            for n in r.case_size
        ]
    )
    a.set_xlim(
        0,
        max(
            17,
            r.load_curtailment_fraction.max()*175
        )
    )
    a.set_ylim(-.5,len(r)-.5)
    a.set_xlabel(
        'Tested load curtailment (%)'
    )
    label(
        a,'(d) Recovery by curtailment'
    )
    grid(a,'x')

    axs[1,0].legend(
        loc='upper center',
        bbox_to_anchor=(.5,-.22),
        ncol=3,frameon=False,
        fontsize=8
    )

    save(
        fig,REAL,
        'R07_ACOPF_constraints_and_recovery',
        'Exp45/46; incomplete thermal validation'
    )


# ===================================================
# R08: OBSERVATION STRESS AND STATISTICAL TESTS
# ===================================================

def real8():
    d = read(
        '40_paired_observation_full/'
        'paired_stress_statistics.csv'
    )

    fig,(a,b) = plt.subplots(
        1,2,figsize=(12,5),
        layout='constrained'
    )

    p = d.pivot(
        index='condition',
        columns='case_size',
        values='delta_PAS_stress_minus_clean'
    ).reindex(
        columns=[14,57,118,300]
    )

    heat(
        a,p,
        '(a) Stress minus clean PAS',
        unit='Delta PAS',
        center=0,
        precision=3
    )
    a.set_xlabel('IEEE bus count')
    a.tick_params(
        axis='y',labelsize=8
    )

    p = d.pivot(
        index='condition',
        columns='case_size',
        values='holm_p_20'
    ).reindex(
        columns=[14,57,118,300]
    )

    heat(
        b,p,
        '(b) Holm-adjusted p-values',
        unit='Adjusted p',
        precision=2
    )
    b.set_xlabel('IEEE bus count')
    b.tick_params(
        axis='y',labelsize=8
    )

    save(
        fig,REAL,
        'R08_observation_stress_and_significance',
        'Exp40 paired stress evaluation'
    )


# ===================================================
# SIMULATED TEACHING DATA
# INDEPENDENT OF REAL EXPERIMENT FILES
# ===================================================

rng = np.random.default_rng(20261023)


# SIM01: SYNTHETIC RAINCLOUD
def sim1():
    names = [
        'CDP (mock)',
        'FULL (mock)',
        'NO CF (mock)',
        'NO Guard (mock)'
    ]

    n = 180
    common = rng.normal(0,.085,n)

    values = [
        np.clip(
            mean + common +
            rng.normal(0,sd,n),
            0,1
        )
        for mean,sd in [
            (.48,.055),
            (.43,.045),
            (.465,.055),
            (.41,.10)
        ]
    ]

    fig,(a,b) = plt.subplots(
        1,2,figsize=(12,4.5),
        layout='constrained'
    )

    raincloud(
        a,values,names,
        [GRAY,TEAL,ORANGE,RED]
    )
    a.set_xlabel(
        'Mock terminal PAS (lower better)'
    )
    label(
        a,'(a) Raincloud / raw points / IQR'
    )

    for vec,nm,c in zip(
        values,names,
        [GRAY,TEAL,ORANGE,RED]
    ):
        b.step(
            np.sort(vec),
            np.arange(1,len(vec)+1)/len(vec),
            where='post',
            label=nm,
            color=c,lw=1.8
        )

    label(
        b,'(b) Synthetic tail comparison'
    )
    b.set_xlabel('Mock PAS')
    b.set_ylabel('Empirical CDF')
    grid(b,'both')

    fig.legend(
        *b.get_legend_handles_labels(),
        loc='outside lower center',
        ncol=4,frameon=False,
        fontsize=8
    )

    save(
        fig,SIM,
        'SIM01_raincloud_ECDF',
        'Synthetic correlated normal samples'
    )


# SIM02: 3D SURFACE AND CONTOUR
def sim2():
    X = np.linspace(0,1,31)
    Y = np.linspace(0,1,31)
    xx,yy = np.meshgrid(X,Y)

    zz = (
        .055
        -.14*(xx-.48)**2
        -.11*(yy-.36)**2
        +.003*np.cos(7*xx)*np.sin(5*yy)
    )

    fig = plt.figure(
        figsize=(11.5,4.9),
        layout='constrained'
    )

    a = fig.add_subplot(
        121,projection='3d'
    )

    a.plot_surface(
        xx,yy,zz,
        cmap='viridis',
        rstride=1,cstride=1,
        linewidth=0,
        antialiased=True,
        alpha=.96
    )
    a.view_init(elev=25,azim=-60)
    a.set_xlabel('Mock gamma')
    a.set_ylabel('Mock kappa')
    a.set_zlabel(
        'Assumed gain',labelpad=1
    )
    a.set_title(
        '(a) Assumed response surface',
        fontsize=10
    )

    b = fig.add_subplot(122)
    im = b.contourf(
        xx,yy,zz,
        levels=16,cmap='viridis'
    )
    b.contour(
        xx,yy,zz,
        levels=7,colors='white',
        linewidths=.5,alpha=.6
    )

    b.scatter(
        [.48],[.36],
        marker='x',
        s=90,color=RED,lw=2
    )
    b.set_xlabel('Mock gamma')
    b.set_ylabel('Mock kappa')
    label(
        b,'(b) Contours / local optimum'
    )

    fig.colorbar(
        im,ax=b,shrink=.75,
        pad=.04,label='Assumed gain'
    )

    save(
        fig,SIM,
        'SIM02_3D_surface_and_contours',
        'Analytic synthetic response surface'
    )


# SIM03: CALIBRATION AND FALSE ACCEPTANCE
def sim3():
    predicted = rng.normal(
        .025,.055,360
    )
    independent = (
        .74*predicted +
        rng.normal(0,.036,360)
    )
    accepted = predicted > .025

    fig,(a,b) = plt.subplots(
        1,2,figsize=(11.8,4.3),
        layout='constrained'
    )

    for mask,c,desc in [
        (
            ~accepted,GRAY,
            'Rejected (mock)'
        ),
        (
            accepted,TEAL,
            'Accepted (mock)'
        )
    ]:
        a.scatter(
            predicted[mask],
            independent[mask],
            s=13,alpha=.49,
            color=c,label=desc,
            rasterized=True
        )

    low = min(
        predicted.min(),
        independent.min()
    )-.01

    high = max(
        predicted.max(),
        independent.max()
    )+.01

    a.plot(
        [low,high],
        [low,high],
        ls='--',color=INK,lw=1
    )
    a.axvline(
        .025,color=ORANGE,ls=':'
    )
    a.set_xlabel('Predicted gain')
    a.set_ylabel(
        'Independent mock gain'
    )
    label(
        a,'(a) Synthetic prediction calibration'
    )
    grid(a,'both')

    a.legend(
        loc='upper center',
        bbox_to_anchor=(.5,-.22),
        ncol=2,
        frameon=False,fontsize=8
    )

    v = independent[accepted]
    w = independent[~accepted]

    b.hist(
        v,bins=18,color=TEAL,
        alpha=.5,density=True,
        label='Accepted'
    )
    b.hist(
        w,bins=18,color=GRAY,
        alpha=.4,density=True,
        label='Rejected'
    )
    b.axvline(
        0,color=RED,ls='--'
    )
    b.set_xlabel('Independent gain')
    b.set_ylabel('Density')
    label(
        b,'(b) False-positive tail stays visible'
    )
    grid(b)

    b.legend(
        loc='upper center',
        bbox_to_anchor=(.5,-.22),
        ncol=2,
        frameon=False,fontsize=8
    )

    save(
        fig,SIM,
        'SIM03_calibration_and_false_acceptance',
        'Synthetic paired gain model'
    )


# SIM04: UPSET-LIKE CHART AND SKEWED DISTRIBUTION
def sim4():
    rng2 = np.random.default_rng(601)
    n,m = 160,5

    binary = (
        rng2.random((n,m)) <
        [.25,.42,.30,.20,.32]
    )

    labels = [
        'Hub','Load','Coupling',
        'Exposure','Budget'
    ]

    patterns = sorted(
        set(tuple(row) for row in binary),
        key=lambda p: -sum(
            np.all(
                binary==np.array(p),axis=1
            )
        )
    )[:10]

    counts = [
        sum(
            np.all(
                binary==np.array(p),axis=1
            )
        )
        for p in patterns
    ]

    fig = plt.figure(
        figsize=(11.5,5.8),
        layout='constrained'
    )
    gs = GridSpec(
        2,2,figure=fig,
        width_ratios=[1.5,1],
        height_ratios=[1,1]
    )

    a = fig.add_subplot(gs[0,0])
    a.bar(
        range(len(patterns)),
        counts,color=BLUE
    )
    a.set_xticks(
        range(len(patterns)),[]
    )
    a.set_ylabel('Mock pattern count')
    label(
        a,'(a) UpSet-like intersection frequency'
    )
    grid(a)

    b = fig.add_subplot(
        gs[1,0],sharex=a
    )

    for i,p in enumerate(patterns):
        present = [
            j for j,val in enumerate(p)
            if val
        ]
        missing = [
            j for j in range(m)
            if j not in present
        ]

        b.scatter(
            [i]*len(present),
            present,color=TEAL,
            s=42,zorder=4
        )

        if len(present)>1:
            b.plot(
                [i,i],
                [min(present),max(present)],
                color=TEAL,lw=2
            )

        if missing:
            b.scatter(
                [i]*len(missing),
                missing,
                facecolor='white',
                edgecolor='#D3DEE5',
                s=28
            )

    b.set_yticks(
        range(m),labels
    )
    b.set_xlabel(
        'Intersection pattern (synthetic)'
    )
    b.set_ylim(-.5,m-.5)
    grid(b,'x')

    a = fig.add_subplot(gs[:,1])
    groups = ['Low','Medium','High']

    values = rng2.lognormal(
        mean=[-.1,.2,.5],
        sigma=[.27,.4,.56],
        size=(150,3)
    )

    a.boxplot(
        [
            values[:,i]
            for i in range(3)
        ],
        tick_labels=groups,
        patch_artist=True,
        showfliers=False,
        boxprops={
            'facecolor':'#DDEBEA',
            'edgecolor':TEAL
        },
        medianprops={
            'color':RED
        }
    )
    a.set_ylabel(
        'Hypothetical impact severity'
    )
    label(
        a,'(b) Asymmetric synthetic severity'
    )
    grid(a)

    save(
        fig,SIM,
        'SIM04_upset_intersections_and_boxplot',
        'Synthetic intersections and lognormal values'
    )


# ===================================================
# EXECUTE AND REGISTER ALL FIGURES
# ===================================================

JOBS = [
    real1,real2,real3,real4,
    real5,real6,real7,real8,
    sim1,sim2,sim3,sim4
]

if __name__=='__main__':
    errors = []

    for i,fn in enumerate(JOBS,1):
        print(
            f'RUN {i}/{len(JOBS)}: '
            f'{fn.__name__}',
            flush=True
        )

        try:
            fn()
        except Exception as ex:
            traceback.print_exc()
            errors.append(
                (fn.__name__,str(ex))
            )

    with (
        OUT/'MANIFEST.csv'
    ).open('w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'source_kind',
            'figure',
            'description'
        ])
        writer.writerows(REPORT)

    print('OUTPUT:',OUT)
    print('GENERATED:',len(REPORT))
    print('ERRORS:',errors)

    if errors:
        sys.exit(2)
