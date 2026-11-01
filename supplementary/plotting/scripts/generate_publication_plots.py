"""Draw the two main experimental figures from recorded summaries only.

Run from any directory: python scripts/generate_publication_plots.py
Optional --root and --out paths are relative to the caller or project folder.
No simulation, filtering by outcome, or source-data edits are performed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd

WIDTH = 7.2
HEIGHT = 3.65
TOPOLOGY_COLORS = {'BA': '#0072B2', 'ER': '#009E73', 'WS': '#B87D14'}
TOPOLOGY_MARKERS = {'BA': 'o', 'ER': 's', 'WS': 'D'}
METHOD_COLORS = {'CDP': '#6C7883', 'LOCAL_RANK': '#009E73',
                 'CF_STATIC': '#0072B2', 'CF_LOCAL_NO_GUARD': '#B87D14',
                 'CF_LOCAL': '#D55E00'}


def set_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8.8,
        'axes.labelsize': 8.8, 'axes.titlesize': 9.4,
        'xtick.labelsize': 8.2, 'ytick.labelsize': 8.7,
        'legend.fontsize': 8.1, 'axes.linewidth': .6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.major.width': .6, 'ytick.major.width': .6,
        'xtick.major.size': 3, 'ytick.major.size': 0,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'axes.unicode_minus': True,
    })


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, out, stem):
    # Fixed canvas: a 7.1-in manuscript placement gives the intended font size.
    fig.savefig(out / (stem + '.pdf'), metadata={'CreationDate': None})
    fig.savefig(out / (stem + '.png'), dpi=300)
    plt.close(fig)


def group_bands(ax):
    for lower, upper in [(-.5, 2.5), (5.5, 8.5)]:
        ax.axhspan(lower, upper, facecolor='#F1F5F7', zorder=0)
    for y in [2.5, 5.5]:
        ax.axhline(y, color='#D1D9DE', lw=.5, zorder=1)


def draw_local(local, out):
    data = pd.read_csv(local)
    data = data[(data.comparison == 'CF_LOCAL vs CF_STATIC') &
                data['mode'].isin(['oracle', 'partial80', 'partial50'])].copy()
    assert len(data) == 18 and not data.duplicated(['topology', 'mode', 'delay']).any()
    rows = [(t, m) for t in ['BA', 'ER', 'WS']
            for m in ['oracle', 'partial80', 'partial50']]
    modes = {'oracle': 'Oracle', 'partial80': '80% recall', 'partial50': '50% recall'}
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    grid = fig.add_gridspec(1, 4, width_ratios=[1, .19, 1, .19], wspace=.08)
    plots = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 2])]
    values = [fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 3])]
    for ax, val, delay in zip(plots, values, [0, 3]):
        group_bands(ax)
        group_bands(val)
        ax.axvline(0, color='#647782', lw=.75, ls=(0, (3, 3)), zorder=2)
        for y, (topology, mode) in enumerate(rows):
            row = data[(data.topology == topology) & (data['mode'] == mode) &
                       (data.delay == delay)].iloc[0]
            mu, lo, hi = row.mean_PAS_gain, row.graph_cluster_CI_low, row.graph_cluster_CI_high
            assert lo <= mu <= hi
            ax.errorbar(mu, y, xerr=[[mu-lo], [hi-mu]],
                        fmt=TOPOLOGY_MARKERS[topology], color=TOPOLOGY_COLORS[topology],
                        markersize=4.8, markeredgewidth=.6, markeredgecolor='white',
                        capsize=2.4, elinewidth=1.25, capthick=.85, zorder=4)
            val.text(.50, y, f'{mu:.4f}', ha='center', va='center',
                     fontsize=8.1, color='#263B46')
        ax.set(xlim=(-.0035, .043), ylim=(8.6, -.7),
               xlabel='PAS gain over static candidate')
        ax.set_xticks([0, .01, .02, .03, .04])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.grid(axis='x', color='#D9E1E5', lw=.45, alpha=.85)
        ax.spines['left'].set_visible(False)
        val.set(xlim=(0, 1), ylim=(8.6, -.7))
        val.set_xticks([]); val.set_yticks([])
        for spine in val.spines.values():
            spine.set_visible(False)
        val.text(.5, 1.025, 'Mean', transform=val.transAxes,
                 ha='center', va='bottom', fontsize=8.2, color='#445662')
    plots[0].set_yticks(range(9), [f'{t}  |  {modes[m]}' for t, m in rows])
    plots[1].set_yticks(range(9), ['']*9)
    fig.subplots_adjust(left=.188, right=.985, top=.84, bottom=.205)
    for i, (ax, val, delay) in enumerate(zip(plots, values, [0, 3])):
        left, right = ax.get_position().x0, val.get_position().x1
        fig.text((left+right)/2, .945, f'Execution delay: {delay} steps',
                 ha='center', va='center', fontsize=9.5)
        fig.text((left+right)/2, .035, f'({chr(97+i)})',
                 ha='center', va='center', fontsize=9.2, fontweight='bold')
    save(fig, out, 'local_effects')
    return {'size_inches': [WIDTH, HEIGHT], 'source_rows': 18,
            'identical_x_limits': [-.0035, .043],
            'source_interval_semantics': 'supplied descriptive graph-bootstrap 95% interval',
            'value_label_rounding_only': 'four decimal places',
            'mean_and_ci_rows': data[['topology', 'mode', 'delay', 'mean_PAS_gain',
                'graph_cluster_CI_low', 'graph_cluster_CI_high']].to_dict('records')}


def draw_reliability(risk, out):
    data = pd.read_csv(risk).set_index('method')
    order = ['CDP', 'LOCAL_RANK', 'CF_STATIC', 'CF_LOCAL_NO_GUARD', 'CF_LOCAL']
    labels = ['CDP', 'Direct local', 'Static candidate', 'Local, no\nconfirmation', 'Guarded local']
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, HEIGHT), sharey=True,
                             gridspec_kw={'width_ratios': [1.03, 1.08, 1.04]})
    for ax in axes:
        for y in [0, 2, 4]:
            ax.axhspan(y-.47, y+.47, color='#F1F5F7', zorder=0)
        ax.grid(axis='x', color='#D9E1E5', lw=.45, zorder=1)
        ax.set_axisbelow(True)
        ax.spines['left'].set_visible(False)
    for y, method in enumerate(order):
        row = data.loc[method]
        color = METHOD_COLORS[method]
        mu = row.harmful_fraction * 100
        lo = row.harmful_fraction_descriptive_ci_low * 100
        hi = row.harmful_fraction_descriptive_ci_high * 100
        assert lo <= mu <= hi
        axes[0].errorbar(mu, y, xerr=[[mu-lo], [hi-mu]], fmt='o',
                         color=color, markersize=5.1, markeredgecolor='white',
                         markeredgewidth=.6, capsize=2.8, elinewidth=1.3,
                         capthick=.9, zorder=3)
        axes[0].text(hi+.75, y, f'{mu:.2f}', va='center', fontsize=8.0,
                     color='#263B46')
        for field, marker, shift in [('P95_PAS', 'o', -.18),
                                     ('P99_PAS', 's', 0),
                                     ('CVaR95_PAS', 'D', .18)]:
            axes[1].scatter(row[field], y+shift, s=27, marker=marker,
                            facecolor=color, linewidths=.55, edgecolors='white', zorder=3)
        regret = row.CVaR95_regret
        axes[2].hlines(y, 0, regret, lw=1.1, color=color, alpha=.65, zorder=2)
        axes[2].scatter(regret, y, s=28, marker='o', color=color,
                        edgecolors='white', linewidth=.6, zorder=3)
        axes[2].text(regret+.0020, y, f'{regret:.4f}', va='center',
                     fontsize=8.0, color='#263B46')
    axes[0].set(yticks=range(5), yticklabels=labels, xlim=(-1.0, 31),
                ylim=(4.55, -.55), xlabel='Worse than CDP (%)')
    axes[0].set_xticks([0, 10, 20, 30])
    axes[1].set(xlim=(.75, .93), xlabel='Empirical PAS')
    axes[1].set_xticks([.75, .80, .85, .90])
    axes[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes[2].set(xlim=(-.003, .104), xlabel='ES95 of positive regret')
    axes[2].set_xticks([0, .04, .08])
    axes[2].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fig.subplots_adjust(left=.17, right=.995, top=.79, bottom=.20, wspace=.27)
    for i, (ax, title) in enumerate(zip(axes,
                ['CDP-relative harm', 'Absolute PAS tails', 'Positive-regret tail'])):
        pos = ax.get_position(); center = (pos.x0+pos.x1)/2
        fig.text(center, .945, title, ha='center', va='center', fontsize=9.1)
        fig.text(center, .035, f'({chr(97+i)})', ha='center', va='center',
                 fontsize=9.2, fontweight='bold')
    handles = [Line2D([], [], marker=mark, ls='', color='#495C69',
                      markersize=4.7, label=label)
               for mark, label in [('o', 'P95'), ('s', 'P99'), ('D', 'ES95')]]
    pos = axes[1].get_position()
    fig.legend(handles=handles, loc='center', bbox_to_anchor=((pos.x0+pos.x1)/2, .86),
               ncol=3, frameon=False, handletextpad=.22, columnspacing=.65,
               borderaxespad=0, fontsize=8.0)
    save(fig, out, 'reliability_profile')
    return {'size_inches': [WIDTH, HEIGHT], 'methods': order,
            'display_names': dict(zip(order, labels)),
            'values': data.loc[order, ['harmful_fraction',
                'harmful_fraction_descriptive_ci_low', 'harmful_fraction_descriptive_ci_high',
                'P95_PAS', 'P99_PAS', 'CVaR95_PAS', 'CVaR95_regret']].reset_index().to_dict('records')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out.resolve() if args.out else root / 'figures/new'
    out.mkdir(parents=True, exist_ok=True)
    local = root / 'data/54_graph_observation_delay/full_g5_s10_start17000/paired_by_topology_observation_delay.csv'
    risk = root / 'data/reliability/exp54_risk_overall.csv'
    sources = [local, risk]
    before = {p.relative_to(root).as_posix(): sha(p) for p in sources}
    set_style()
    report = {'source_sha256': before, 'outcome_generation': False,
              'source_csv_changes': False, 'axis_label_points_at_7_1_in': 8.8*7.1/WIDTH,
              'local_effects': draw_local(local, out),
              'reliability_profile': draw_reliability(risk, out)}
    assert before == {p.relative_to(root).as_posix(): sha(p) for p in sources}
    report['output_sha256'] = {p.name: sha(p) for p in sorted(out.glob('*.pdf'))
                                if p.stem in ['local_effects', 'reliability_profile']}
    (out / 'plot_audit_main.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
