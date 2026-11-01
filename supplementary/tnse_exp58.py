#!/usr/bin/env python3
"""Exp58: archived Exp54 cyber-to-PGLib118 physical stress bridge.
Assumption-based coupling sensitivity; NOT live utility measurements.
"""
import argparse, copy, hashlib, importlib.util, inspect, json, sys, time
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
import pandas as pd
from pypower.idx_bus import BUS_I, PD, QD

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import tnse_exp56_pglib_acopf as opf
S54=ROOT/'tnse_exp54_final.py'
spec=importlib.util.spec_from_file_location('exp54_bridge',S54)
exp54=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=exp54
spec.loader.exec_module(exp54)
SRC=ROOT/'revision_experiments/results/54_graph_observation_delay/full_g5_s10_start17000'
OUT=ROOT/'revision_experiments/results/58_cyber_power_sensitivity'
OUT.mkdir(parents=True,exist_ok=True)
MET=['CDP','CF_STATIC','CF_LOCAL']
SEVERITIES=[.10,.30]
LOAD_FACTORS=[1.,1.08]
KEYS=list(exp54.KEYS)

def sha(file):return hashlib.sha256(Path(file).read_bytes()).hexdigest()

def coupling(pattern,ppc):
    path=ROOT/f'outputs/tables/coupling_{pattern}.csv'
    tab=pd.read_csv(path)
    if not {'comm_node','power_node'}.issubset(tab.columns):raise ValueError(f'Invalid coupling {path}')
    c=tab[['comm_node','power_node']].astype(int)
    if len(c)!=118 or set(c.comm_node)!=set(range(118)) or set(c.power_node)!=set(range(118)):
        raise ValueError(f'Nonbijective IEEE118 cyber-power mapping: {path}')
    # Original pandapower layer indexes power nodes 0..117.
    # PGLib physical bus IDs MUST agree with the index-order contract.
    busids=ppc['bus'][:,BUS_I].astype(int)
    if not np.array_equal(busids,np.arange(1,119)):
        raise RuntimeError('PGLib bus ordering differs from 1..118. Require reviewed crosswalk.')
    return c.set_index('comm_node').power_node.to_dict(),sha(path)

def recover_states(case,graphs,weights,archived):
    caught=[]
    old=exp54.ref.pas
    def tap(state,weight):
        frame=inspect.currentframe().f_back
        if frame.f_code.co_name=='run_case' and frame.f_globals.get('__name__')==exp54.__name__:
            caught.append(np.asarray(state,dtype=np.int8).copy())
        return old(state,weight)
    exp54.ref.pas=tap
    try:
        rows,events=exp54.run_case(case,weights,graphs)
    finally:
        exp54.ref.pas=old
    if len(caught)!=len(exp54.METHODS):
        raise RuntimeError(f'State capture failed: {len(caught)} vs {len(exp54.METHODS)}')
    mapping={m:state for m,state in zip(exp54.METHODS,caught)}
    lookup={r['method']:r for r in rows}
    original=archived.set_index('method')
    for method in exp54.METHODS:
        delta=abs(float(lookup[method]['PAS'])-float(original.loc[method,'PAS']))
        if delta>1e-10:
            raise RuntimeError(f'Archived replay mismatch {method}: {delta}; do not use this sample')
    return mapping,lookup

def select_cases(archived,phase,ngraph,nseed):
    d=archived[(archived['mode'].isin(['partial80','partial50'])) &
               (archived['method']=='CDP') &
               (archived['topology'].isin(['BA','ER','WS']))]
    graphs=sorted(d.graph_seed.unique())[:ngraph]
    seeds=sorted(d.seed.unique())[:nseed]
    d=d[d.graph_seed.isin(graphs) & d.seed.isin(seeds)]
    if phase=='pilot':
        d=d[(d.topology=='BA') & (d['mode']=='partial80') & (d.delay==0) & (d.pattern=='DDAC')]
        d=d.head(1)
    cases=[tuple(r[k] for k in KEYS) for r in d.to_dict('records')]
    if not cases:raise RuntimeError('No archived cases for requested design')
    return cases

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--phase',choices=['pilot','full'],required=True)
    a.add_argument('--graphs',type=int,default=1)
    a.add_argument('--seeds',type=int,default=1)
    opt=a.parse_args()
    target=OUT/f'{opt.phase}_g{opt.graphs}_s{opt.seeds}'
    target.mkdir(parents=True,exist_ok=True)
    if opt.graphs not in range(1,6) or opt.seeds not in range(1,11):
        raise ValueError('Choose graphs=1..5, seeds=1..10; subset only of archived Exp54')
    original=pd.read_csv(SRC/'raw_outcomes.csv')
    if original.duplicated(KEYS+['method']).any():raise RuntimeError('Duplicate archived outcomes')
    cases=select_cases(original,opt.phase,opt.graphs,opt.seeds)
    hashes={str(p.relative_to(ROOT)):sha(p) for p in [
        SRC/'raw_outcomes.csv',SRC/'FROZEN_PROTOCOL.json',S54,
        ROOT/'tnse_exp56_pglib_acopf.py',Path(__file__)]}
    cfg={'phase':opt.phase,'graphs':opt.graphs,'seeds':opt.seeds,'cases':cases,
         'fractions':SEVERITIES,'load_factors':LOAD_FACTORS,'methods':MET,
         'source_hashes':hashes,
         'assumption':'comm-node coupled to power-row, infected terminal nodes interrupt a fixed fraction of PD/QD'}
    lock=target/'PROTOCOL.json'
    if lock.exists() and json.loads(lock.read_text())!=json.loads(json.dumps(cfg)):
        raise RuntimeError('Protocol changed, do not reuse old checkpoints')
    if not lock.exists():lock.write_text(json.dumps(cfg,indent=2))
    ppc,meta=opf.get_case(118)
    if meta['rate_coverage']<1.:raise RuntimeError('IEEE118 thermal coverage incomplete')
    maps={}
    for pattern in ['DDAC','RC']:
        maps[pattern]=coupling(pattern,ppc)
    keys={(str(c[0]),int(c[1])) for c in cases}
    graphs={}
    for topo,gseed in keys:
        g=exp54.graph_of(topo,gseed)
        adj=exp54.ref.adjacency(g)
        degree=np.array([g.degree(i) for i in range(118)],float)
        graphs[(topo,gseed)]=(g,adj,degree)
    weights={p:exp54.ref.severity_from_pattern(p,118) for p in ['DDAC','RC']}
    checkpoints=target/'checkpoints';checkpoints.mkdir(exist_ok=True)
    for j,case in enumerate(cases,1):
        identifier='_'.join(map(str,case))
        file=checkpoints/f'{identifier}.json'
        if not file.exists():
            filters=np.ones(len(original),dtype=bool)
            for col,x in zip(KEYS,case):filters &= original[col].astype(str).to_numpy()==str(x)
            block=original.loc[filters].copy()
            if len(block)!=len(exp54.METHODS):raise RuntimeError(f'Incomplete archive: {case}')
            terminal,metrics=recover_states(case,graphs,weights,block)
            comm_to_power=maps[case[2]][0]
            outputs=[]
            for method in MET:
                state=terminal[method]
                infected=np.flatnonzero(state==2)
                bus_rows=np.array([comm_to_power[int(i)] for i in infected],dtype=int)
                for load_factor in LOAD_FACTORS:
                    for severity in SEVERITIES:
                        affected=copy.deepcopy(ppc)
                        affected['bus'][:,PD]*=load_factor
                        affected['bus'][:,QD]*=load_factor
                        gross=float(affected['bus'][:,PD].sum())
                        lost=float((affected['bus'][bus_rows,PD]*severity).sum()) if len(bus_rows) else 0.
                        affected['bus'][bus_rows,PD]*=(1.-severity)
                        affected['bus'][bus_rows,QD]*=(1.-severity)
                        result,_=opf.solve(affected,1.,meta,-1)
                        result.update({**dict(zip(KEYS,case)),
                            'method':method,'original_PAS':metrics[method]['PAS'],
                            'terminal_I':len(infected),'infected_comm_nodes':infected.astype(int).tolist(),
                            'load_factor':load_factor,'demand_interruption_fraction':severity,
                            'gross_demand_MW':gross,'interrupted_demand_MW':lost,
                            'interrupted_fraction_gross':lost/max(gross,1e-12),
                            'coupling_sha256':maps[case[2]][1],
                            'physical_case_sha256':meta['source_sha256']})
                        outputs.append(result)
            temp=file.with_suffix('.json.part')
            temp.write_text(json.dumps(outputs,allow_nan=True))
            temp.replace(file)
        if j%5==0 or j==len(cases):print(f'[{j}/{len(cases)}] coupled cyber/physical scenarios',flush=True)
    allrows=[]
    for case in cases:
        allrows.extend(json.loads((checkpoints/('_'.join(map(str,case))+'.json')).read_text()))
    df=pd.DataFrame(allrows)
    df.to_csv(target/'ALL_PHYSICAL_REPLAYS.csv',index=False)
    key=KEYS+['load_factor','demand_interruption_fraction']
    wide=df.pivot(index=key,columns='method',values='interrupted_demand_MW')
    if wide.isna().any().any():raise RuntimeError('Incomplete physical pairing')
    comparisons=[]
    for a,b in [('CDP','CF_LOCAL'),('CF_STATIC','CF_LOCAL')]:
        dif=(wide[a]-wide[b]).to_numpy(float)
        comparisons.append({'comparison':f'{b} vs {a}', 'n':len(dif),
            'mean_interrupted_demand_reduction_MW':float(dif.mean()),
            'fraction_better':float(np.mean(dif>1e-10)),
            'fraction_worse':float(np.mean(dif< -1e-10)),
            'fraction_tied':float(np.mean(abs(dif)<=1e-10))})
    pd.DataFrame(comparisons).to_csv(target/'PHYSICAL_PAIRED_CONTRASTS.csv',index=False)
    df.groupby(['method','load_factor','demand_interruption_fraction'],dropna=False).agg(
        n=('success','size'),opf_solved=('success','sum'),
        mean_interrupted_demand_MW=('interrupted_demand_MW','mean'),
        mean_max_thermal_utilization=('max_thermal_utilization','mean')
        ).reset_index().to_csv(target/'PHYSICAL_SUMMARY.csv',index=False)
    (target/'SCOPE.txt').write_text(
        'ASSUMPTION-BASED sensitivity only: coupling_*.csv maps original pandapower row IDs to PGLib IEEE118 bus row IDs.\n'
        'Terminal infectious cyber nodes lead to load INTERRUPTION proportional to a predefined fraction (10% or 30%).\n'
        'No measured physical failures; no empirically identified dependence of a bus on a cyber node.\n'
        'Controller trajectory replay is checked against archived Exp54 PAS on every case.\n'
        'Lower interrupted MW means less assumed load interruption; OPF feasibility is separately recorded.\n'
        'AC OPF allows redispatch; does not prove original fixed-dispatch security or closed-loop physical MPC.\n')
    pack=target/'EXP58_EVIDENCE.zip'
    with ZipFile(pack,'w',ZIP_DEFLATED) as z:
        for f in target.rglob('*'):
            if f.is_file() and f!=pack and not f.name.endswith('.part'):z.write(f,f.relative_to(ROOT))
        z.write(Path(__file__),'SOURCE/tnse_exp58.py')
    with ZipFile(pack) as z:
        if z.testzip():raise RuntimeError('ZIP verification failed')
    print('\nEXP58 COMPLETE\n',pd.DataFrame(comparisons).to_string(index=False),'\n',target)

if __name__=='__main__':main()
