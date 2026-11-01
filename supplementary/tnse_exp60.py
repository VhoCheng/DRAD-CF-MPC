#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exp60: statistically honest cyber-to-physical bridge replication audit.

Uses genuine Exp54/Exp58 outputs; no synthetic rows, no retuning.
Resamples whole BASE ATTACK SCENARIOS, not four correlated stress copies.
"""
import argparse, hashlib, json, sys
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
R = ROOT/'revision_experiments/results'
BASE = ['topology','graph_seed','pattern','budget','onset','seed']
CONTEXT = ['mode','delay']
STRESS = ['load_factor','demand_interruption_fraction']
METHODS = ['CDP','CF_STATIC','CF_LOCAL']
CONTRASTS = [('CDP','CF_LOCAL'),('CF_STATIC','CF_LOCAL')]


def sourcehash(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def confidence(cluster, seed, B=4000):
    """Nested resampling: graph realizations within topology, then attack cases.

    If only one graph exists per topology, intervals are conditional on
    those fixed graphs and are NOT graph-population confidence intervals.
    """
    rng=np.random.default_rng(seed)
    n=len(cluster)
    boot=np.zeros(B)
    graph_effects=[]
    gcounts=[]
    for _,topo in cluster.groupby('topology',sort=True):
        graphs=[g for _,g in topo.groupby('graph_seed',sort=True)]
        G=len(graphs)
        gcounts.append(G)
        # Each graph should contribute the same number of experimental cases
        # under the balanced factorial design.
        if len({len(g) for g in graphs})!=1:
            raise ValueError('Unbalanced graph sample counts: cannot resample fairly')
        simulations=[]
        for block in graphs:
            x=block.gain.to_numpy(float)
            idx=rng.integers(0,len(x),size=(B,len(x)))
            simulations.append(x[idx].mean(axis=1))
            graph_effects.append(float(x.mean()))
        graph_boot=np.column_stack(simulations)
        select=rng.integers(0,G,size=(B,G))
        topo_mean=np.take_along_axis(graph_boot,select,axis=1).mean(axis=1)
        boot += topo_mean*(len(topo)/n)
    low,high=np.quantile(boot,[.025,.975])
    scope=('graph_and_attack' if min(gcounts)>=2 else 'fixed_graphs_attack_only')
    # Graph-level paired randomization, not a sign-flip on correlated stress rows.
    p=float('nan')
    if min(gcounts)>=2:
        graph_effects=np.asarray(graph_effects,float)
        G=len(graph_effects)
        obs=abs(graph_effects.mean())
        extreme=0;P=10000
        for first in range(0,P,250):
            flips=rng.choice([-1.,1.],size=(min(250,P-first),G))
            extreme+=int((abs((flips*graph_effects).mean(axis=1))>=obs-1e-15).sum())
        p=(extreme+1)/(P+1)
    return float(low),float(high),p,scope


def audit_mapping(df, out):
    """Independently reconstruct assumed interrupted demand from source tables."""
    try:
        sys.path.insert(0,str(ROOT))
        import tnse_exp56_pglib_acopf as p56
        from pypower.idx_bus import PD
        case,meta = p56.get_case(118)
        assert meta['rate_coverage'] == 1.0
        pd_load = np.asarray(case['bus'][:,PD],float)
    except Exception as exc:
        raise RuntimeError('Cannot read pinned PGLib118 source; mapping audit cannot be skipped') from exc
    tables={}
    for pat in sorted(df.pattern.unique()):
        f=ROOT/f'outputs/tables/coupling_{pat}.csv'
        tab=pd.read_csv(f)
        if not {'comm_node','power_node'}.issubset(tab.columns):
            raise ValueError(f'Missing mapping columns {f}')
        comm=tab.comm_node.astype(int)
        power=tab.power_node.astype(int)
        if set(comm)!=set(range(118)) or set(power)!=set(range(118)):
            raise ValueError(f'Nonbijective cross-layer mapping {f}')
        mapping=dict(zip(comm,power))
        tables[pat]=(mapping,sourcehash(f))
    errors=[]
    for rec in df.itertuples(index=False):
        nodes=json.loads(rec.infected_comm_nodes)
        if len(nodes)!=int(rec.terminal_I) or len(set(nodes))!=len(nodes):
            raise ValueError('Terminal infection count inconsistent with stored nodes')
        if not all(0<=int(i)<118 for i in nodes):
            raise ValueError('Infected node outside IEEE118')
        mapping,expected_sha=tables[rec.pattern]
        if rec.coupling_sha256!=expected_sha:
            raise ValueError('Exp58 coupling SHA differs from present file')
        if rec.physical_case_sha256!=meta['source_sha256']:
            raise ValueError('Exp58 physical case SHA differs from pinned case')
        implied=float(pd_load[[mapping[int(i)] for i in nodes]].sum())*float(rec.load_factor)*float(rec.demand_interruption_fraction)
        errors.append(implied-float(rec.interrupted_demand_MW))
    err=np.array(errors)
    summary={'rows_verified':len(df),'max_abs_MW_reconstruction_error':float(np.max(abs(err))),
             'PGLib_118_source_sha256':meta['source_sha256'],
             'DDAC_mapping_sha256':tables.get('DDAC',(None,None))[1],
             'RC_mapping_sha256':tables.get('RC',(None,None))[1],
             'note':'This confirms arithmetic and archived crosswalk, NOT empirically validated cyber-to-bus causality.'}
    (out/'RECONSTRUCTION_AUDIT.json').write_text(json.dumps(summary,indent=2))
    if np.max(abs(err))>1e-5:
        raise RuntimeError(f'Physical MW reconstruction failed; maximum error {abs(err).max()}')
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',default='full_g1_s1',help='Exp58 archived folder name, e.g. full_g5_s2')
    a=ap.parse_args()
    src=R/'58_cyber_power_sensitivity'/a.source
    orig=src/'ALL_PHYSICAL_REPLAYS.csv'
    if not orig.is_file():raise FileNotFoundError(orig)
    df=pd.read_csv(orig)
    out=R/'60_physical_replication_audit'/a.source
    out.mkdir(parents=True,exist_ok=True)
    cols=BASE+CONTEXT+STRESS+['method','original_PAS','infected_comm_nodes','terminal_I',
         'interrupted_demand_MW','success','constraint_pass','coupling_sha256','physical_case_sha256']
    missing=set(cols)-set(df.columns)
    if missing:raise ValueError(f'Missing Exp58 columns: {missing}')
    if df.duplicated(BASE+CONTEXT+STRESS+['method']).any():raise ValueError('Duplicate physical rows')
    if not np.isfinite(df.interrupted_demand_MW).all():raise ValueError('Nonfinite demand loss')
    distinct_methods=set(df.method.unique())
    if distinct_methods!=set(METHODS):raise ValueError(f'Unexpected methods {distinct_methods}')
    wide=df.pivot(index=BASE+CONTEXT+STRESS,columns='method',values='interrupted_demand_MW')
    if wide.isna().any().any():raise ValueError('Missing paired physical results')
    group_count=wide.reset_index()[BASE].drop_duplicates().shape[0]
    trajectory_count=wide.reset_index()[BASE+CONTEXT].drop_duplicates().shape[0]
    stress_count=len(wide)
    for keys,grp in df.groupby(BASE+CONTEXT+['method']):
        if len(grp)!=4:raise ValueError(f'Expected four stress settings per trajectory: {keys}')
        if grp.original_PAS.max()-grp.original_PAS.min()>1e-10:raise ValueError('PAS changed across identical trajectory stress copies')
    audit_mapping(df,out)
    # Archived PAS must match Exp54 for exactly the same case/method.
    f54=R/'54_graph_observation_delay/full_g5_s10_start17000/raw_outcomes.csv'
    raw=pd.read_csv(f54)
    raw=raw[raw.method.isin(METHODS)][BASE+CONTEXT+['method','PAS']]
    distinct=df[BASE+CONTEXT+['method','original_PAS']].drop_duplicates()
    merged=distinct.merge(raw,on=BASE+CONTEXT+['method'],validate='one_to_one',how='left',indicator=True)
    if (merged._merge!='both').any() or not np.allclose(merged.original_PAS,merged.PAS,atol=1e-10,rtol=0):
        raise RuntimeError('Exp58 PAS differs from archived Exp54')
    # Two estimands: (i) pooled stresses, (ii) fixed 10% / 30% physical dependency.
    rows=[]; effects=[]
    for a,b in CONTRASTS:
        item=wide[a].sub(wide[b]).rename('gain').reset_index()
        for severity in [None,.10,.30]:
            current=item if severity is None else item[np.isclose(item.demand_interruption_fraction,severity)]
            # Average all repeated sensing, delay, and load perturbations of same *base attack*.
            cluster=current.groupby(BASE,sort=True).gain.mean().reset_index()
            gain=cluster.gain.to_numpy(float)
            lo,hi,p,scope=confidence(cluster,seed=60123+len(rows))
            label='all_predeclared' if severity is None else f'interruption_{severity:.2f}'
            rows.append({'comparison':f'{b} vs {a}','subset':label,
                 'n_physical_rows':len(current),'n_trajectories':current[BASE+CONTEXT].drop_duplicates().shape[0],
                 'n_base_attack_clusters':len(cluster),'n_graph_realizations':cluster[['topology','graph_seed']].drop_duplicates().shape[0],
                 'mean_gain_MW':float(gain.mean()),'cluster_bootstrap_CI_low':lo,'cluster_bootstrap_CI_high':hi,
                 'graph_signflip_p':p,'CI_scope':scope,'positive_base_cluster_fraction':float((gain>1e-10).mean()),
                 'negative_base_cluster_fraction':float((gain< -1e-10).mean())})
            if severity is None:
                effects.append((f'{b} vs {a}',cluster))
        # Per-topology descriptive subgroup estimates.
        for topo,g in item.groupby('topology'):
            cluster=g.groupby(BASE,sort=True).gain.mean().reset_index()
            lo,hi,p,scope=confidence(cluster,seed=62000+len(rows))
            rows.append({'comparison':f'{b} vs {a}','subset':f'topology_{topo}',
                 'n_physical_rows':len(g),'n_trajectories':g[BASE+CONTEXT].drop_duplicates().shape[0],
                 'n_base_attack_clusters':len(cluster),'n_graph_realizations':cluster[['topology','graph_seed']].drop_duplicates().shape[0],
                 'mean_gain_MW':float(cluster.gain.mean()),'cluster_bootstrap_CI_low':lo,'cluster_bootstrap_CI_high':hi,
                 'graph_signflip_p':p,'CI_scope':scope,'positive_base_cluster_fraction':float((cluster.gain>1e-10).mean()),
                 'negative_base_cluster_fraction':float((cluster.gain< -1e-10).mean())})
    stats=pd.DataFrame(rows)
    # Holm across exactly the two prespecified aggregate comparisons (subgroups exploratory).
    main=stats[stats.subset=='all_predeclared'].copy()
    pvals=main.graph_signflip_p.to_numpy()
    stats['holm_p_main_only']=np.nan
    if np.isfinite(pvals).all():
        order=np.argsort(pvals)
        adjusted=np.empty(2)
        adjusted[order]=np.minimum(1.,np.maximum.accumulate(pvals[order]*np.array([2,1])))
        stats.loc[main.index,'holm_p_main_only']=adjusted
    stats.to_csv(out/'CLUSTERED_PAIRED_EFFECTS.csv',index=False)
    # Do not equate solver convergence to physical soundness or closed-loop control.
    solve_rows=[]
    for (m,load,sev),g in df.groupby(['method','load_factor','demand_interruption_fraction']):
        valid=g['constraint_pass'].fillna(False).astype(bool)
        solved=g.success.fillna(False).astype(bool)
        solve_rows.append({'method':m,'load_factor':load,'interruption_fraction':sev,'n':len(g),
              'opf_solved':int(solved.sum()),'fully_checked':int(valid.sum()),
              'mean_interrupted_MW':float(g.interrupted_demand_MW.mean())})
    pd.DataFrame(solve_rows).to_csv(out/'FEASIBILITY_BY_STRESS.csv',index=False)
    # Visualization: interval forest. One row per primary effect and sensitivity level.
    selected=stats[stats.subset.isin(['all_predeclared','interruption_0.10','interruption_0.30'])]
    fig,ax=plt.subplots(figsize=(10,4.6))
    ys=np.arange(len(selected))
    for i,row in enumerate(selected.itertuples()):
        x=row.mean_gain_MW
        ax.errorbar(x,i,xerr=[[max(0,x-row.cluster_bootstrap_CI_low)],[max(0,row.cluster_bootstrap_CI_high-x)]],
                    fmt='o',ms=5,color='#1D7D83',ecolor='#1D7D83',capsize=3)
    ax.axvline(0,color='#B35C5C',lw=1,linestyle='--')
    ax.set_yticks(ys,[f'{r.comparison} | {r.subset}' for r in selected.itertuples()])
    ax.invert_yaxis();ax.set_xlabel('Assumed interrupted-demand reduction (MW; paired)')
    ax.set_title('Exp58: cluster-aware sensitivity across assumed physical perturbations',loc='left',pad=15)
    ax.grid(axis='x',alpha=.15);fig.subplots_adjust(left=.42,right=.97,top=.87,bottom=.16)
    for ext in ('pdf','svg','png'):fig.savefig(out/f'F60_physical_cluster_effects.{ext}',dpi=250,bbox_inches='tight')
    plt.close(fig)
    info={'Exp58_source':str(orig.relative_to(ROOT)),'Exp58_sha256':sourcehash(orig),
          'n_physical_paired_settings':stress_count,'n_distinct_trajectories':trajectory_count,
          'n_base_attack_clusters':group_count,
          'n_graph_instances':df[['topology','graph_seed']].drop_duplicates().shape[0],
          'assumptions':'Cyber infection implies 10%/30% load interruption at mapped PGLib bus; OPF redispatch allowed.',
          'limits':['Do not treat 4 stress settings as independent attack seeds',
                    'When one graph per topology, CI is conditional and graph-level p-value is not computed',
                    'This is an assumption-driven sensitivity replay, not physical closed-loop MPC']}
    (out/'METHOD_NOTE.json').write_text(json.dumps(info,indent=2))
    zipfile=out/'EXP60_EVIDENCE.zip'
    with ZipFile(zipfile,'w',ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(out.iterdir()):
            if f.is_file() and f!=zipfile:z.write(f,f.name)
        z.write(Path(__file__),'SOURCE/tnse_exp60.py')
    with ZipFile(zipfile) as z:
        if z.testzip() is not None:raise RuntimeError('Bad ZIP')
    print('\nEXP60 COMPLETE -- no original result overwritten')
    print(json.dumps(info,indent=2))
    print('\nPRIMARY ESTIMANDS\n',stats[stats.subset=='all_predeclared'].to_string(index=False))
    print('\nOUTPUT',out)

if __name__=='__main__':main()
