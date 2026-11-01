"""Draw supplemental confirmation, timing and threshold figures from CSVs.

The graph-bootstrap intervals, all six timing methods and all 25 threshold
configurations are retained. No simulation or source-data edits are performed.
Run: python scripts/generate_supplement_publication_plots.py
"""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from generate_publication_plots import (set_style, sha, save, group_bands,
                                        TOPOLOGY_COLORS, TOPOLOGY_MARKERS,
                                        METHOD_COLORS)


def draw_guard_cost(raw_path, stats_path, timing_path, out):
    raw, stats, timing = (pd.read_csv(p) for p in [raw_path, stats_path, timing_path])
    rows = [(t, m) for t in ['BA', 'ER', 'WS']
            for m in ['oracle', 'partial80', 'partial50']]
    obs_names = {'oracle': 'Oracle', 'partial80': '80%', 'partial50': '50%'}
    fig = plt.figure(figsize=(7.2, 3.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, .28, 1.42])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[0, 3])]
    contrasts = []
    for ax, delay in zip(axes[:2], [0, 3]):
        group_bands(ax)
        for y, (top, mode) in enumerate(rows):
            match = stats[(stats.topology == top) & (stats['mode'] == mode)
                          & (stats.delay == delay)
                          & (stats.comparison == 'CF_LOCAL vs CF_LOCAL_NO_GUARD')]
            assert len(match) == 1
            row = match.iloc[0]
            mu, lo, hi = [float(row[field]) for field in
                          ['mean_PAS_gain', 'graph_cluster_CI_low', 'graph_cluster_CI_high']]
            assert -1.2 <= lo*1000 <= mu*1000 <= hi*1000 <= 1.2
            ax.errorbar(mu*1000, y, xerr=[[(mu-lo)*1000], [(hi-mu)*1000]],
                        fmt=TOPOLOGY_MARKERS[top], color=TOPOLOGY_COLORS[top],
                        capsize=2.3, markersize=4.5, markeredgecolor='white',
                        markeredgewidth=.55, elinewidth=1.15, capthick=.8, zorder=3)
            contrasts.append({'topology': top, 'mode': mode, 'delay': delay,
                              'mean': mu, 'low': lo, 'high': hi})
        ax.axvline(0, color='#647782', linestyle=(0, (3, 3)), linewidth=.75, zorder=2)
        ax.set(xlim=(-1.2, 1.2), ylim=(8.6, -.7),
               yticks=np.arange(9), xticks=[-1, 0, 1],
               xlabel=r'PAS gain ($\times10^{-3}$)')
        ax.grid(axis='x', color='#D9E1E5', linewidth=.45)
        ax.spines['left'].set_visible(False)
    axes[0].set_yticklabels([t + ' | ' + obs_names[m] for t, m in rows], fontsize=8.1)
    axes[1].set_yticklabels([])

    means = raw.groupby('method').PAS.mean()
    ct = timing.assign(total=lambda x: x.n*x.mean_s).groupby('method').agg(
        total=('total', 'sum'), n=('n', 'sum'))
    ct['ms'] = ct.total/ct.n*1000
    order = ['CDP', 'STATIC_RANK', 'LOCAL_RANK', 'CF_STATIC', 'CF_LOCAL',
             'CF_LOCAL_NO_GUARD']
    names = {'CDP': 'CDP', 'STATIC_RANK': 'Static rank', 'LOCAL_RANK': 'Direct local',
             'CF_STATIC': 'Static candidate', 'CF_LOCAL': 'Guarded local',
             'CF_LOCAL_NO_GUARD': 'No confirmation'}
    offsets = {'CDP': (5, 6, 'left'), 'STATIC_RANK': (5, 0, 'left'),
               'LOCAL_RANK': (5, -6, 'left'), 'CF_STATIC': (-4, 9, 'right'),
               'CF_LOCAL': (-2, 13, 'center'),
               'CF_LOCAL_NO_GUARD': (-2, -16, 'center')}
    colors = dict(METHOD_COLORS, STATIC_RANK='#806F96')
    markers = {'CDP': 'o', 'STATIC_RANK': 's', 'LOCAL_RANK': 'D',
               'CF_STATIC': '^', 'CF_LOCAL': 'o', 'CF_LOCAL_NO_GUARD': 'v'}
    costs = []
    ax = axes[2]
    for method in order:
        xx, yy = float(ct.loc[method, 'ms']), float(means[method])
        ax.scatter(xx, yy, s=29, color=colors[method], marker=markers[method],
                   edgecolor='white', linewidth=.5, zorder=4)
        dx, dy, ha = offsets[method]
        kw = {'arrowprops': {'arrowstyle': '-', 'lw': .45, 'color': '#71808A',
                             'shrinkA': 1.5, 'shrinkB': 3.5}} if method.startswith('CF_LOCAL') else {}
        ax.annotate(names[method], (xx, yy), xytext=(dx, dy),
                    textcoords='offset points', ha=ha, va='center', fontsize=7.6,
                    color='#263B46', **kw)
        costs.append({'method': method, 'mean_PAS': yy,
                      'weighted_mean_stage_time_ms': xx, 'n_timing_stages': int(ct.loc[method, 'n'])})
    ax.set(xscale='log', xlim=(.12, 180), ylim=(.247, .289),
           xlabel='Time per stage (ms)', ylabel='Mean PAS')
    ax.set_xticks([.2, 1, 10, 100], labels=['0.2', '1', '10', '100'])
    ax.set_yticks([.25, .26, .27, .28])
    ax.grid(axis='y', color='#D9E1E5', linewidth=.45)
    fig.subplots_adjust(left=.119, right=.988, top=.84, bottom=.22, wspace=.15)
    for i, (a, title) in enumerate(zip(axes, ['Delay: 0 steps', 'Delay: 3 steps', 'Decision cost'])):
        pos = a.get_position(); center = (pos.x0 + pos.x1)/2
        fig.text(center, .94, title, ha='center', va='center', fontsize=9.4)
        fig.text(center, .035, f'({chr(97+i)})', ha='center', va='center',
                 fontsize=9.2, fontweight='bold')
    save(fig, out, 'guard_cost')
    return {'size_inches': [7.2, 3.6], 'contrasts': contrasts,
            'negative_contrast_count': sum(r['mean'] < 0 for r in contrasts),
            'cost_methods': costs,
            'source_interval_semantics': 'supplied descriptive graph-bootstrap 95% interval',
            'time_aggregation': 'sum(n * mean_s) / sum(n), converted to milliseconds'}


def draw_threshold(grid_path, out):
    grid = pd.read_csv(grid_path)
    assert len(grid) == 25 and not grid.duplicated(['gamma', 'kappa']).any()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.95))
    limits = {}
    for ax, field, title, cmap, scale in zip(
            axes, ['harmful_fraction', 'P95_PAS', 'P99_PAS'],
            ['Worse than CDP (%)', 'P95 PAS', 'P99 PAS'],
            ['YlOrRd', 'Blues', 'PuBu'], [100, 1, 1]):
        a = grid.pivot(index='kappa', columns='gamma', values=field)*scale
        assert a.shape == (5, 5) and np.isfinite(a.to_numpy()).all()
        vmin, vmax = float(a.to_numpy().min()), float(a.to_numpy().max())
        limits[field] = {'min': vmin, 'max': vmax, 'display_multiplier': scale}
        im = ax.pcolormesh(np.arange(6)-.5, np.arange(6)-.5, a.to_numpy(),
                           cmap=cmap, vmin=vmin, vmax=vmax, shading='flat',
                           edgecolors='white', linewidth=.65, rasterized=False)
        ax.set(xlim=(-.5, 4.5), ylim=(-.5, 4.5))
        ax.set_xticks(range(5), labels=[f'{v:.2f}' for v in a.columns])
        ax.set_yticks(range(5), labels=[f'{v:.2f}' for v in a.index])
        ax.set(xlabel=r'Margin $\gamma$')
        ax.tick_params(length=0, pad=3, labelsize=7.8)
        for i in range(5):
            for j in range(5):
                value = float(a.iloc[i, j])
                rgb = im.cmap(im.norm(value))[:3]
                luminance = .2126*rgb[0] + .7152*rgb[1] + .0722*rgb[2]
                ax.text(j, i, f'{value:.1f}' if scale == 100 else f'{value:.3f}',
                        ha='center', va='center', fontsize=8.0,
                        color='white' if luminance < .53 else '#172936')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=9.2, pad=12)
    axes[0].set_ylabel(r'Uncertainty $\kappa$')
    fig.subplots_adjust(left=.08, right=.988, top=.82, bottom=.255, wspace=.30)
    for i, ax in enumerate(axes):
        pos = ax.get_position()
        fig.text((pos.x0+pos.x1)/2, .035, f'({chr(97+i)})',
                 ha='center', va='center', fontsize=9.2, fontweight='bold')
    save(fig, out, 'parameter_sensitivity')
    return {'size_inches': [7.2, 2.95], 'cells_per_panel': len(grid),
            'color_ranges': limits,
            'source_values': grid[['gamma', 'kappa', 'harmful_fraction',
                                   'P95_PAS', 'P99_PAS']].to_dict('records')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out.resolve() if args.out else root/'figures/new'
    out.mkdir(parents=True, exist_ok=True)
    trial = root/'data/54_graph_observation_delay/full_g5_s10_start17000'
    sources = [trial/'raw_outcomes.csv', trial/'paired_by_topology_observation_delay.csv',
               trial/'decision_timing.csv', root/'data/reliability/exp02_all_25_threshold_replay_risk.csv']
    before = {p.relative_to(root).as_posix(): sha(p) for p in sources}
    set_style()
    report = {'source_sha256': before, 'outcome_generation': False, 'source_csv_changes': False,
              'guard_cost': draw_guard_cost(*sources[:3], out),
              'parameter_sensitivity': draw_threshold(sources[3], out)}
    assert before == {p.relative_to(root).as_posix(): sha(p) for p in sources}
    report['output_sha256'] = {p.name: sha(p) for p in sorted(out.glob('*.pdf'))
                              if p.stem in ['guard_cost', 'parameter_sensitivity']}
    (out/'plot_audit_supplement.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
