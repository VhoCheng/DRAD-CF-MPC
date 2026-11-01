#!/usr/bin/env python3
"""Exp59: evidence release gate; never promotes a failed/untested claim."""
import hashlib,json,sys
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parent
R=ROOT/'revision_experiments/results'
OUT=R/'59_submission_evidence_release';OUT.mkdir(parents=True,exist_ok=True)
BASE=R/'54_graph_observation_delay/full_g5_s10_start17000'
checks=[]
def check(name,passflag,details):
    checks.append({'test':name,'pass':bool(passflag),'details':str(details)})
    print(('[PASS]' if passflag else '[BLOCK]'),name,details,flush=True)

def main():
    raw=pd.read_csv(BASE/'raw_outcomes.csv')
    keys=['topology','graph_seed','pattern','budget','onset','seed','mode','delay']
    wide=raw.pivot(index=keys,columns='method',values='PAS')
    check('Exp54 full pairing', len(raw)==57600 and len(wide)==9600 and not wide.isna().any().any(),f'{len(raw)} method rows, {len(wide)} cases')
    p56=pd.read_csv(R/'56_pglib_v2307_acopf/ALL_ACOPF_RUNS.csv')
    physical=p56.groupby('case_size',dropna=False).agg(n=('success','size'),solver_success=('success','sum'))
    check('Exp56 4 network sizes',set(physical.index)=={14,57,118,300},physical.to_dict())
    check('Exp56 only validated thermal limits', bool(p56.loc[p56.case_size.isin([14,57,118]),'constraint_pass'].eq(True).sum()>0),
          'IEEE300 may remain partially unrated; do not include it in full thermal guarantee')
    ratings=pd.read_csv(R/'56_pglib_v2307_acopf/PGLIB_SOURCE_COVERAGE.csv')
    incomplete=ratings.loc[ratings.rate_coverage<.999999,['case_size','rate_coverage']]
    check('Complete thermal ratings all sizes',incomplete.empty,incomplete.to_dict('records'))
    status57=R/'57_pglib_stress_n1/full/ALL_RUNS.csv'
    if status57.exists():
        data57=pd.read_csv(status57)
        check('Exp57 extended physical stress present',len(data57)>20,f'{len(data57)} total solved/failed N1 and stress runs')
        check('Exp57 archived failures explicitly',{'success','classified'}.issubset(data57.columns),'classification columns available')
    else:check('Exp57 extended physical stress present',False,'Run python tnse_exp57.py --phase full')
    releases=sorted((R/'58_cyber_power_sensitivity').glob('full_*/ALL_PHYSICAL_REPLAYS.csv')) if (R/'58_cyber_power_sensitivity').exists() else []
    if releases:
        p58=pd.read_csv(releases[-1]);req={'original_PAS','interrupted_demand_MW','source_sha256','success','method'}
        # PGLIB source hash stored as physical_case_sha256, not source_sha256
        req.remove('source_sha256');req.add('physical_case_sha256')
        check('Exp58 cyber-power mapping verified',req.issubset(p58.columns),f'{len(p58)} physical replays; {releases[-1]}')
        check('Exp58 AC OPF convergence proportion recorded',True,f'{int(p58.success.sum())}/{len(p58)}')
    else:check('Exp58 cyber-power mapping verified',False,'Run python tnse_exp58.py --phase full --graphs 1 --seeds 1')
    flags=R/'55_final_evidence_closeout/F_REVIEWER_CLOSEOUT_FLAGS.txt'
    check('Exp55 closeout flags retained',flags.exists(),str(flags))
    fullstate='Oracle observation' ; practical='Partial80 and Partial50 assumed sensors'
    mapping=[
       ('R1.6 / R2.8','Topology generalization', 'Exp54/55', 'Qualified: 5 graphs/topology and fixed physical mapping'),
       ('R1.8','Paired statistics / uncertainty','Exp54/55','Use graph-cluster intervals; distinguish descriptive vs inferential'),
       ('R1.10','Component ablation','Exp49/50/54','STATE_LOCAL is an algorithm change; original NO_DYNAMIC advantage unsupported'),
       ('R1.11 / R3.2','Observation and delay','Exp51/52/54/55','Simulated sensing/actuation, not field-measured latency'),
       ('R1.5 / R3.3','Guard safety tradeoff','Exp42/47/54/55','Guard cost can be negative, no universal guarantee'),
       ('R1.4 / R3.1','Scalability / timing','Exp25/37/54/55','Controller compute time only; not end-to-end'),
       ('Physical safety','Thermal AC OPF and N-1','Exp56/57','Solver failure not proof of infeasibility; IEEE300 rating gap'),
       ('Physical coupling','Cyber-to-physical interruption assumption','Exp58','Prescribed comm→bus mapping, modeled load interruption, not field effect'),
       ('Objective reproducibility','EP double-discount / Eq21 source alignment','Source audit required','Unresolved unless corrected and rerun')]
    pd.DataFrame(mapping,columns=['reviewer_point','claim','evidence','necessary_qualification']).to_csv(OUT/'REVIEWER_EVIDENCE_MAP.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'RELEASE_GATE.csv',index=False)
    mustfix=[c for c in checks if not c['pass']]
    (OUT/'STATUS.txt').write_text(
        'STATUS: '+('REVIEW READY WITH LIMITATIONS' if not mustfix else 'PENDING OR BLOCKED')+'\n'+
        '\n'.join(f"{x['test']}: {x['details']}" for x in mustfix)+'\n\n'+
        'Never equate standalone PGLib AC OPF with certified cyber-physical closed-loop security.\n'+
        'Never claim gate improvement if threshold 0 and QUALITY_GATE=STATE_LOCAL.\n'+
        'Report failed and infeasible tests, include complete code/protocol hashes.\n')
    files=[BASE/'FROZEN_PROTOCOL.json',BASE/'raw_outcomes.csv',ROOT/'tnse_exp57.py',ROOT/'tnse_exp58.py',ROOT/'tnse_exp59.py']
    hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.exists()}
    (OUT/'FILE_HASHES.json').write_text(json.dumps(hashes,indent=2))
    target=OUT/'EXP59_RELEASE_AUDIT.zip'
    with ZipFile(target,'w',ZIP_DEFLATED) as zip:
        for f in OUT.iterdir():
            if f.is_file() and f!=target: zip.write(f,f.name)
    with ZipFile(target) as zip:
        if zip.testzip():raise RuntimeError('archive checksum failed')
    print('\nRELEASE STATUS:',(OUT/'STATUS.txt').read_text(),'\nARTIFACT:',target)

if __name__=='__main__':main()
