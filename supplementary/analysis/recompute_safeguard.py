#!/usr/bin/env python3
"""Read-only reanalysis of archived results. No simulator runs or favorable filtering.

Run from repository root: /opt/miniconda3/bin/python3 work/v7/evidence/recompute_safeguard.py
Outputs stay beside this script. Requires numpy and pandas. All intervals descriptive.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
import pandas as pd

EPS = 1e-12
KEY = ['topology', 'graph_seed', 'pattern', 'budget', 'onset', 'seed', 'mode', 'delay']
BOOT = 5000
RNG_SEED = 260725
USED = set()


def read(path):
    USED.add(path.resolve())
    return pd.read_csv(path)


def es95(x):
    """Exact integral of empirical inverse CDF over (.95,1), including fractional rank."""
    a = np.sort(np.asarray(x, float))[::-1]
    mass = .05 * len(a)
    whole = int(math.floor(mass + 1e-10))
    fraction = mass - whole
    total = float(a[:whole].sum())
    if fraction > 1e-10:
        total += fraction * a[whole]
    return total / mass


def risk(g, value='PAS', base='CDP_PAS', selection='ever_non_CDP'):
    a, b = g[value].to_numpy(), g[base].to_numpy()
    gain = b - a
    harm = gain < -EPS
    sel = g[selection].astype(bool).to_numpy() if selection in g else np.zeros(len(g), bool)
    return dict(n=len(g), mean_PAS=float(a.mean()), median_PAS=float(np.median(a)),
                P95_PAS=float(np.quantile(a, .95)), P99_PAS=float(np.quantile(a, .99)),
                CVaR95_PAS=es95(a), max_PAS=float(a.max()), mean_gain_vs_CDP=float(gain.mean()),
                harmful_n=int(harm.sum()), harmful_fraction=float(harm.mean()),
                improved_n=int((gain > EPS).sum()), tied_n=int((abs(gain) <= EPS).sum()),
                any_non_CDP_n=int(sel.sum()), any_non_CDP_fraction=float(sel.mean()),
                harmful_conditional_non_CDP_n=int((harm & sel).sum()),
                harmful_conditional_non_CDP_fraction=float(harm[sel].mean()) if sel.any() else None,
                mean_positive_regret=float(np.maximum(-gain, 0).mean()),
                max_regret=float((-gain).max()), P95_regret=float(np.quantile(-gain, .95)),
                CVaR95_regret=es95(-gain))


def cluster_ci(g):
    """Resample graph means within topology; equal graphs/cases in each input cell.

    Observation, delay, attack contexts and repeated method outcomes remain inside
    their graph. This is descriptive between-graph uncertainty, not a new test.
    Conditional-harm CI resamples ratios via graph-level harm/selected totals.
    """
    v = g.assign(gain=g.CDP_PAS-g.PAS, harm=(g.PAS-g.CDP_PAS > EPS).astype(float))
    block = v.groupby(['topology', 'graph_seed']).agg(
        gain=('gain', 'mean'), harm=('harm', 'mean'), selected=('ever_non_CDP', 'sum'),
        harmful_selected=('harmful_selected', 'sum'))
    rng = np.random.default_rng(RNG_SEED)
    samples = []
    for _, frame in block.groupby(level='topology', sort=True):
        a = frame.to_numpy(float)
        ix = rng.integers(0, len(a), (BOOT, len(a)))
        samples.append(a[ix].sum(axis=1))
    draws = sum(samples)
    draws[:, :2] /= len(block)
    conditional = np.divide(draws[:, 3], draws[:, 2], out=np.full(BOOT, np.nan), where=draws[:, 2] > 0)
    out = dict(graph_n=len(block), ci_resamples=BOOT)
    for label, values in [('mean_gain', draws[:, 0]), ('harmful_fraction', draws[:, 1]),
                          ('conditional_harmful_fraction', conditional)]:
        valid = values[np.isfinite(values)]
        ci = np.quantile(valid, [.025, .975]) if len(valid) else [None, None]
        out[label+'_descriptive_ci_low'], out[label+'_descriptive_ci_high'] = ci
    return out


def summarize(df, groups, ci=False):
    rows = []
    iterator = df.groupby(groups, sort=True, dropna=False)
    for labels, g in iterator:
        labels = labels if isinstance(labels, tuple) else (labels,)
        row = dict(zip(groups, labels))
        row.update(risk(g))
        if ci:
            row.update(cluster_ci(g))
        rows.append(row)
    return pd.DataFrame(rows)


def save(df, out, name):
    df.to_csv(out/name, index=False, float_format='%.16g')


def main(root, out):
    out.mkdir(parents=True, exist_ok=True)
    def locate(candidates, required):
        for candidate in candidates:
            if (candidate/required).exists():
                return candidate
        raise FileNotFoundError('Missing '+required+'; checked '+', '.join(str(x) for x in candidates))
    b = locate([root/'work/v5/imported/02_ALL_EXPERIMENT_RESULTS/revision_experiments/results',
                root/'02_ALL_EXPERIMENT_RESULTS/revision_experiments/results',
                root/'revision_experiments/results',root/'results',root],
               '49_split_confirm_revision/tune_validation_all_grid.csv')
    src = locate([root/'work/v5/imported/03_SOURCE_CODE_LOGS',root/'03_SOURCE_CODE_LOGS',root],
                 'tnse_exp54_final.py')
    p = locate([root/'outputs/TNSE_V6_Revision/data/54_graph_observation_delay/full_g5_s10_start17000',
                root/'data/54_graph_observation_delay/full_g5_s10_start17000',
                b/'54_graph_observation_delay/full_g5_s10_start17000'], 'raw_outcomes.csv')
    d, s = read(p/'raw_outcomes.csv'), read(p/'stage_actions.csv')
    assert len(d) == 57600 and len(s) == 172800
    assert not d.duplicated(KEY+['method']).any()
    assert not s.duplicated(KEY+['method', 'stage']).any()
    baseline = d[d.method == 'CDP'][KEY+['PAS']].rename(columns={'PAS':'CDP_PAS'})
    assert len(baseline) == 9600
    d = d.merge(baseline, on=KEY, validate='many_to_one')
    # Source non_CDP denotes actual node-set inequality, not merely a policy name.
    scdp = s[s.method == 'CDP'][KEY+['stage','protected_nodes']].rename(columns={'protected_nodes':'CDP_nodes'})
    s = s.merge(scdp, on=KEY+['stage'], validate='many_to_one')
    sets = lambda x: frozenset(json.loads(x))
    s['actual_non_CDP'] = [int(sets(a) != sets(b)) for a,b in zip(s.protected_nodes,s.CDP_nodes)]
    assert (s.actual_non_CDP == s.non_CDP).all()
    counts = s.groupby(KEY+['method']).actual_non_CDP.sum().rename('verified_non_CDP_actions').reset_index()
    d = d.merge(counts, on=KEY+['method'], validate='one_to_one')
    assert (d.non_CDP_actions == d.verified_non_CDP_actions).all()
    d['ever_non_CDP'] = d.verified_non_CDP_actions > 0
    d['harmful_selected'] = ((d.PAS-d.CDP_PAS > EPS) & d.ever_non_CDP).astype(int)
    assert not ((d.PAS-d.CDP_PAS > EPS) & ~d.ever_non_CDP).any()
    overall = summarize(d, ['method'], True)
    grouped = summarize(d, ['topology','mode','delay','method'], True)
    full = summarize(d, ['topology','pattern','budget','onset','mode','delay','method'], True)
    save(overall, out, 'exp54_risk_overall.csv')
    save(grouped, out, 'exp54_risk_topology_observation_delay.csv')
    save(full, out, 'exp54_risk_all_192_configurations.csv')
    save(d[KEY+['method','PAS','CDP_PAS','verified_non_CDP_actions','ever_non_CDP','harmful_selected']],
         out, 'exp54_verified_episode_pairs.csv')
    wide = d.pivot(index=KEY, columns='method', values='PAS')
    comparisons = [('CF_LOCAL','CDP'),('CF_LOCAL','CF_STATIC'),('CF_LOCAL','LOCAL_RANK'),
                   ('CF_LOCAL','CF_LOCAL_NO_GUARD'),('CF_STATIC','CDP'),('LOCAL_RANK','CDP')]
    contrast_rows = []
    for target, comparator in comparisons:
        temp = wide[[target,comparator]].reset_index().rename(columns={target:'PAS',comparator:'CDP_PAS'})
        temp['ever_non_CDP'] = False
        temp['harmful_selected'] = 0
        for labels,g in [('ALL',temp), *list(temp.groupby(['topology','mode','delay'], sort=True))]:
            row = dict(target=target,comparator=comparator,scope='overall' if labels=='ALL' else 'topology_mode_delay')
            row.update(dict(zip(['topology','mode','delay'], labels)) if labels!='ALL' else {})
            vals = risk(g)
            vals['mean_gain_target_vs_comparator'] = vals.pop('mean_gain_vs_CDP')
            row.update(vals)
            row.update(cluster_ci(g))
            contrast_rows.append(row)
    save(pd.DataFrame(contrast_rows),out,'exp54_paired_contrasts.csv')
    risk_contrasts=[]
    for comparator in ['CDP','CF_LOCAL_NO_GUARD','LOCAL_RANK','CF_STATIC']:
        target='CF_LOCAL'
        a,bv=wide[target],wide[comparator]
        baseline=wide.CDP
        temp=wide[[target,comparator] if comparator!='CDP' else [target]].reset_index()[KEY]
        temp['PAS']=(a.to_numpy()-baseline.to_numpy()>EPS).astype(float)
        temp['CDP_PAS']=(bv.to_numpy()-baseline.to_numpy()>EPS).astype(float)
        temp['ever_non_CDP']=False
        temp['harmful_selected']=0
        for labels,g in [('ALL',temp),*list(temp.groupby(['topology','mode','delay'],sort=True))]:
            row=dict(target=target,comparator=comparator,scope='overall' if labels=='ALL' else 'topology_mode_delay',n=len(g),
                target_harmful_fraction=float(g.PAS.mean()),comparator_harmful_fraction=float(g.CDP_PAS.mean()),
                harmful_fraction_reduction=float((g.CDP_PAS-g.PAS).mean()))
            row.update(dict(zip(['topology','mode','delay'], labels)) if labels!='ALL' else {})
            ci=cluster_ci(g)
            row.update({k:ci[k] for k in ['graph_n','ci_resamples','mean_gain_descriptive_ci_low','mean_gain_descriptive_ci_high']})
            risk_contrasts.append(row)
    save(pd.DataFrame(risk_contrasts),out,'exp54_paired_harmful_fraction_reductions.csv')
    event_rows = []
    for (topo,mode,delay,method),g in s.groupby(['topology','mode','delay','method']):
        event_rows.append(dict(topology=topo,mode=mode,delay=delay,method=method,stage_n=len(g),
            actual_non_CDP_stage_n=int(g.actual_non_CDP.sum()),
            actual_non_CDP_stage_fraction=float(g.actual_non_CDP.mean()),
            reached_confirmation_n=int((g.rollout_count==64).sum()),
            confirmation_rejected_n=int(((g.rollout_count==64)&(g.actual_non_CDP==0)).sum())))
    save(pd.DataFrame(event_rows),out,'exp54_stage_gate_events.csv')
    # Paired harms relative to CDP: compare two policies' harm indicator on same episode.
    guard_gain = wide.CDP-wide.CF_LOCAL
    no_guard_gain = wide.CDP-wide.CF_LOCAL_NO_GUARD
    gh,nh = guard_gain < -EPS, no_guard_gain < -EPS
    transition = dict(both_harmful_n=int((gh&nh).sum()), guard_only_harmful_n=int((gh&~nh).sum()),
        no_guard_only_harmful_n=int((~gh&nh).sum()), neither_harmful_n=int((~gh&~nh).sum()),
        unequal_PAS_n=int((abs(wide.CF_LOCAL-wide.CF_LOCAL_NO_GUARD)>EPS).sum()))
    # Historical Exp02 same-preview replay of all 25 thresholds; not Exp54.
    oldpath = next(b.glob('02_safeguard_full*_per_run.csv'))
    old = read(oldpath)
    (out/'exp02_archived_all_15000_raw_rows.csv').write_bytes(oldpath.read_bytes())
    old['PAS'],old['ever_non_CDP'] = old.DRAD_PAS,old.switched.astype(bool)
    oldrows=[]
    for (gamma,kappa),g in old.groupby(['gamma','kappa']):
        r=dict(gamma=gamma,kappa=kappa,protocol='Exp02_archived_single_action_preview_replay')
        r.update(risk(g));oldrows.append(r)
    oldsummary=pd.DataFrame(oldrows)
    save(oldsummary,out,'exp02_all_25_threshold_replay_risk.csv')
    # More recent split-confirmation validation: complete 12-grid, same 48 attack cases.
    valrows=[]
    for folder in sorted((b/'49_split_confirm_revision').glob('val_gamma_*')):
        t=read(folder/'outcomes.csv'); e=read(folder/'decisions.csv')
        _,_,gamma,_,kappa=folder.name.split('_')
        keys=['pattern','budget','onset','seed']
        base=t[t.method=='CDP'][keys+['PAS']].rename(columns={'PAS':'CDP_PAS'})
        t=t[t.method=='FULL'].merge(base,on=keys,validate='one_to_one')
        # Chosen names distinguish CDP from alternative; decision log lacks every
        # alternative's same-state external counterfactual outcome.
        eb=e[e.method=='CDP'][keys+['stage','protected_nodes']].rename(columns={'protected_nodes':'CDP_nodes'})
        ef=e[e.method=='FULL'].merge(eb,on=keys+['stage'],validate='one_to_one')
        ef['alt']=[sets(a)!=sets(b) for a,b in zip(ef.protected_nodes,ef.CDP_nodes)]
        ever=ef.groupby(keys).alt.any().rename('ever_non_CDP').reset_index()
        t=t.merge(ever,on=keys,validate='one_to_one')
        a=t.PAS.to_numpy()
        legacy=float(np.sort(a)[max(0,int(.95*len(a))-1):].mean())
        row=dict(gamma=float(gamma),kappa=float(kappa),protocol='Exp49_validation_only_split_confirmation',
            legacy_source_CVaR95=legacy, legacy_source_score=float(a.mean()+.2*legacy),
            standardized_mean_plus_point2_ES95=float(a.mean()+.2*es95(a)))
        row.update(risk(t));valrows.append(row)
    vals=pd.DataFrame(valrows)
    assert len(vals)==12 and (vals.n==48).all()
    logged=read(b/'49_split_confirm_revision/tune_validation_all_grid.csv')
    check=vals.merge(logged,on=['gamma','kappa'],validate='one_to_one')
    assert np.allclose(check.legacy_source_CVaR95,check.val_CVaR95)
    assert np.allclose(check.legacy_source_score,check.score)
    save(vals,out,'exp49_all_12_validation_threshold_risk.csv')
    # Frozen holdout for context, not a holdout sweep.
    t=read(b/'49_split_confirm_revision/test/outcomes.csv')
    keys=['pattern','budget','onset','seed']
    base=t[t.method=='CDP'][keys+['PAS']].rename(columns={'PAS':'CDP_PAS'})
    t=t.merge(base,on=keys,validate='many_to_one')
    t['ever_non_CDP']=t.switch_count>0
    save(summarize(t,['method']),out,'exp49_locked_holdout_risk.csv')
    source_files=[p/'FROZEN_PROTOCOL.json',src/'tnse_exp54_final.py',src/'tnse_exp49.py',
        src/'revision_experiments/02_safeguard_full.py',
        src/'revision_experiments/47_final_closed_loop_validation.py',
        b/'49_split_confirm_revision/LOCKED_HYPERPARAMETERS.json',
        next(b.glob('02_safeguard_full*_config.json'))]
    source_files.extend((b/'49_split_confirm_revision').glob('val_gamma_*/protocol.json'))
    USED.update(x.resolve() for x in source_files)
    USED.add(Path(__file__).resolve())
    hashes=[dict(path=str(x.relative_to(root)) if x.is_relative_to(root) else 'external/'+x.name,
        bytes=x.stat().st_size,sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in sorted(USED)]
    (out/'source_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    local=full[full.method=='CF_LOCAL']
    summary=dict(protocol='V7_read_only_safe_guard_reanalysis',epsilon=EPS,quantile_method='numpy linear',
        CVaR95_definition='Exact empirical upper-tail expected shortfall: top 5% mass, fractional boundary if needed.',
        uncertainty='5000 within-topology graph bootstrap draws; descriptive only; paired outcome contexts remain inside graphs.',
        bootstrap_seed=RNG_SEED,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
        episode_pairs=9600,graph_n=15,base_attack_context_n=1200,
        stage_set_flags_verified=True,stage_true_counterfactual_harm_identifiable=False,
        stage_true_counterfactual_reason='No same-true-stage-state, candidate-specific external continuation outcomes in stage_actions or checkpoints; policy trajectory paths diverge.',
        exp54_overall=overall.replace({np.nan:None}).to_dict('records'),
        exp54_guard_harm_transitions=transition,
        exp54_full_configuration_counts=dict(positive=int((local.mean_gain_vs_CDP>EPS).sum()),
            tied=int((abs(local.mean_gain_vs_CDP)<=EPS).sum()),negative=int((local.mean_gain_vs_CDP < -EPS).sum())),
        exp54_negative_mean_configurations=local[local.mean_gain_vs_CDP < -EPS].to_dict('records'),
        exp54_threshold_sweep_available=False,
        exp02_grid_n=len(oldsummary),exp02_all_grid_mean_gain_negative=bool((oldsummary.mean_gain_vs_CDP < -EPS).all()),
        exp49_validation_grid_n=len(vals),exp49_validation_pairs_per_grid=48,
        exp49_locked_gamma=0,exp49_locked_kappa=0,
        limitations=['No deterministic or all-configuration realized baseline floor.',
                    'Conditional non-CDP rate describes selected episodes, not causal stage-wise harm.',
                    'Pooled tails mix configurations/observation modes; subgroup CSVs retained.',
                    'Historical Exp02 and Exp49 sweeps cannot calibrate the revised Exp54 controller.',
                    'PAS zero upon detection can conceal residual infection burden; these are PAS risks, not MW losses.'])
    (out/'evidence_numbers_v7.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(overall=summary['exp54_overall'],configurations=summary['exp54_full_configuration_counts'],
        guard_transitions=transition),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root','--root',dest='root',type=Path,default=Path.cwd(),
        help='Workspace root, or extracted archive root containing 02_ALL_EXPERIMENT_RESULTS and 03_SOURCE_CODE_LOGS. Defaults to cwd.')
    parser.add_argument('--output','--out',dest='out',type=Path,default=Path(__file__).resolve().parent,
        help='Output directory; defaults to the directory containing this script.')
    args=parser.parse_args()
    main(args.root.resolve(),args.out.resolve())
