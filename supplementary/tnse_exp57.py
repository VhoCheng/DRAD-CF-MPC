#!/usr/bin/env python3
"""Exp57: independent PGLib AC-OPF operating envelope and selected N-1 tests."""
import argparse, copy, hashlib, json, sys
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd
from pypower.idx_brch import BR_STATUS, RATE_A, PF, QF, PT, QT

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import tnse_exp56_pglib_acopf as src
OUT=ROOT/'revision_experiments/results/57_pglib_stress_n1'
OUT.mkdir(parents=True,exist_ok=True)
SIZES=[14,57,118,300]
LOADS=[.85,.90,.95,1.00,1.05,1.10]

def freeze(path, content):
    if path.exists() and json.loads(path.read_text())!=content:
        raise RuntimeError('Protocol changed; preserve old results and create a new version')
    if not path.exists(): path.write_text(json.dumps(content,indent=2,sort_keys=True))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['pilot','full'],required=True)
    p.add_argument('--n1-top',type=int,default=8)
    p.add_argument('--n1-random',type=int,default=4)
    a=p.parse_args()
    phase_dir=OUT/a.phase
    phase_dir.mkdir(parents=True,exist_ok=True)
    args={'phase':a.phase,'sizes':SIZES if a.phase=='full' else [14,118],
          'load_scales':LOADS if a.phase=='full' else [1.0],
          'N1_top':a.n1_top if a.phase=='full' else 1,
          'N1_random':a.n1_random if a.phase=='full' else 0,
          'source_hashes':{x:hashlib.sha256((ROOT/x).read_bytes()).hexdigest()
             for x in ['tnse_exp56_pglib_acopf.py',Path(__file__).name]}}
    freeze(phase_dir/'PROTOCOL.json',args)
    tasks=[]; metas=[]
    for n in args['sizes']:
        case,meta=src.get_case(n)
        metas.append(meta)
        print(f'[PGLIB] IEEE{n}: thermal coverage {meta["rated_branch_count"]}/{meta["active_branch_count"]}',flush=True)
        for scale in args['load_scales']:
            tasks.append((n,'load',float(scale),-1))
        # Contingencies ranked BEFORE outages, using basecase AC-OPF solution.
        base,solved=src.solve(case,1.0,meta,-1)
        if solved is None:
            print(f'[WARN] IEEE{n} base OPF failed; N-1 ranking skipped',flush=True)
            continue
        br=solved['branch']; active=(br[:,BR_STATUS]>0); rated=active&(br[:,RATE_A]>0)&(br[:,RATE_A]<9000)
        apparent=np.maximum(np.hypot(br[:,PF],br[:,QF]),np.hypot(br[:,PT],br[:,QT]))
        utilization=np.full(len(br),-np.inf)
        utilization[rated]=apparent[rated]/br[rated,RATE_A]
        top=[int(x) for x in np.argsort(-utilization,kind='stable')[:args['N1_top']] if rated[x]]
        candidates=[int(x) for x in np.flatnonzero(active) if x not in top]
        rng=np.random.default_rng(57000+n)
        random_picks=rng.choice(candidates,size=min(len(candidates),args['N1_random']),replace=False).tolist()
        for j in top: tasks.append((n,'N1_high_loading',1.0,int(j)))
        for j in random_picks: tasks.append((n,'N1_random',1.0,int(j)))
    pd.DataFrame(metas).to_csv(phase_dir/'SOURCE_COVERAGE.csv',index=False)
    rows=[]
    for ix,(n,kind,scale,outage) in enumerate(tasks,1):
        case,meta=src.get_case(n)
        result,_=src.solve(case,scale,meta,outage)
        result.update({'kind':kind,'expected_case':n,'source_used':'PGLib v23.07',
                       'thermal_rating_complete':meta['rate_coverage']>=.999999})
        rows.append(result)
        if ix%5==0 or ix==len(tasks):
            print(f'[{ix}/{len(tasks)}] IEEE{n} {kind} load={scale} branch={outage} '
                  f'OPF={result["success"]} full_constraints={result.get("constraint_pass")}',flush=True)
        pd.DataFrame(rows).to_csv(phase_dir/'ALL_RUNS.csv',index=False)
    df=pd.DataFrame(rows)
    df['classified']=np.select([
        ~df['connected'].fillna(False).astype(bool),
        ~df['success'].fillna(False).astype(bool),
        ~df.get('constraint_pass',pd.Series(False,index=df.index)).fillna(False).astype(bool)],
        ['ISLAND','SOLVER_FAILURE_OR_INFEASIBLE','SUCCESS_WITH_UNVERIFIED_CONSTRAINTS'],
        default='SUCCESS_CHECKED')
    df.to_csv(phase_dir/'ALL_RUNS.csv',index=False)
    summary=df.groupby(['expected_case','kind','classified'],dropna=False).size().reset_index(name='n')
    summary.to_csv(phase_dir/'CLASSIFICATION.csv',index=False)
    note=('AC OPF study with PGLib v23.07 source data. OPF solver failure is NOT proof of physical infeasibility.\n'
          'IEEE300 has 8 in-service branches with unspecified RATE_A in current benchmark, so complete thermal safety is unverified.\n'
          'No cyber-to-power closed-loop claim follows from these standalone OPF results.\n'
          'Outages are selected high-loading and random, not exhaustive N-1.\n')
    (phase_dir/'LIMITATIONS.txt').write_text(note)
    zpath=phase_dir/'EXP57_EVIDENCE.zip'
    with ZipFile(zpath,'w',ZIP_DEFLATED) as z:
        for file in sorted(phase_dir.iterdir()):
            if file.is_file() and file!=zpath:z.write(file,file.name)
        z.write(Path(__file__),'SOURCE/tnse_exp57.py')
    with ZipFile(zpath) as z:
        if z.testzip():raise RuntimeError('zip corrupt')
    print('\nEXP57 COMPLETE\n',summary.to_string(index=False),'\nOUTPUT',phase_dir,flush=True)

if __name__=='__main__':main()
