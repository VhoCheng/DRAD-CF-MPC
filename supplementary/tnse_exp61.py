#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exp61: physical data provenance, IEEE300 thermal gaps, IEEE118 crosswalk.

Diagnostic audit only. It never invents line ratings or fills missing values.
"""
import hashlib, json, sys
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
R=ROOT/'revision_experiments/results'
OUT=R/'61_physical_provenance_audit'
OUT.mkdir(parents=True,exist_ok=True)
from pypower.idx_bus import BUS_I, PD, QD
from pypower.idx_brch import F_BUS,T_BUS,BR_STATUS,RATE_A,RATE_B,RATE_C,BR_R,BR_X
import tnse_exp56_pglib_acopf as p56


def main():
    source=[];missing=[]
    for n in (14,57,118,300):
        case,meta=p56.get_case(n)
        branch=case['branch']; active=branch[:,BR_STATUS]>0
        valid=active&(branch[:,RATE_A]>0)&(branch[:,RATE_A]<9000)
        for j in np.flatnonzero(active&~valid):
            b=branch[j]
            missing.append({'network':n,'branch_index_zero_based':int(j),
                 'from_bus':int(b[F_BUS]),'to_bus':int(b[T_BUS]),
                 'RATE_A':float(b[RATE_A]),'RATE_B':float(b[RATE_B]),'RATE_C':float(b[RATE_C]),
                 'R_pu':float(b[BR_R]),'X_pu':float(b[BR_X]),
                 'status':'UNVERIFIED_RATING_DO_NOT_IMPUTE'})
        source.append({'network':n,'active':int(active.sum()),'rateA_valid':int(valid.sum()),
             'unknown':int((active&~valid).sum()),'coverage':float(valid.sum()/active.sum()),
             'sha256':meta['source_sha256'],'source_url':meta['source_url']})
    source=pd.DataFrame(source)
    source.to_csv(OUT/'PGLIB_RATING_COVERAGE.csv',index=False)
    missed=pd.DataFrame(missing,columns=['network','branch_index_zero_based','from_bus','to_bus',
          'RATE_A','RATE_B','RATE_C','R_pu','X_pu','status'])
    missed.to_csv(OUT/'MISSING_LINE_RATINGS.csv',index=False)
    # Check identity/case differences between original pandapower IEEE118 and PGLib118.
    crosswalk={'availability':'unverified'}
    try:
        import pandapower.networks as pn
        pp=pn.case118()
        ppc,_=p56.get_case(118)
        busids=np.asarray(ppc['bus'][:,BUS_I],int)
        ppbuses=np.asarray(sorted(pp.bus.index),int)
        if len(ppbuses)!=118 or set(ppbuses)!=set(range(118)) or not np.array_equal(busids,np.arange(1,119)):
            raise ValueError('IEEE118 row-label mapping is not 0..117 ↔ 1..118')
        pp_P=pp.load.groupby('bus')['p_mw'].sum().reindex(range(118),fill_value=0).to_numpy(float)
        pp_Q=pp.load.groupby('bus')['q_mvar'].sum().reindex(range(118),fill_value=0).to_numpy(float)
        lib_P=np.asarray(ppc['bus'][:,PD],float)
        lib_Q=np.asarray(ppc['bus'][:,QD],float)
        table=pd.DataFrame({'original_power_row_0based':range(118),'PGLib_bus_id_1based':busids,
                            'pandapower_PD_MW':pp_P,'PGLib_PD_MW':lib_P,
                            'pandapower_QD_Mvar':pp_Q,'PGLib_QD_Mvar':lib_Q})
        table['abs_delta_PD_MW']=abs(table['pandapower_PD_MW']-table['PGLib_PD_MW'])
        table['abs_delta_QD_Mvar']=abs(table['pandapower_QD_Mvar']-table['PGLib_QD_Mvar'])
        table.to_csv(OUT/'IEEE118_PANDAPOWER_PGLIB_LOAD_CROSSWALK.csv',index=False)
        crosswalk={'availability':'verified_label_order_only','P_load_corr':float(np.corrcoef(pp_P,lib_P)[0,1]),
                   'max_bus_PD_delta_MW':float(max(abs(pp_P-lib_P))),
                   'sum_original_PD_MW':float(pp_P.sum()),'sum_PGLib_PD_MW':float(lib_P.sum()),
                   'total_absolute_bus_PD_difference_MW':float(abs(pp_P-lib_P).sum()),
                   'max_bus_QD_delta_Mvar':float(max(abs(pp_Q-lib_Q))),
                   'fully_equal_demands':bool(np.allclose(pp_P,lib_P,atol=1e-6,rtol=0) and
                                              np.allclose(pp_Q,lib_Q,atol=1e-6,rtol=0)),
                   'interpretation':'Bus label coincidence is not proof that all engineering model parameters agree.'}
    except Exception as exc:
        crosswalk={'availability':'FAILED_OR_UNAVAILABLE','exception':repr(exc),
                   'action':'Do not assert exact physical-model equivalence.'}
    (OUT/'CROSSWALK_FINDINGS.json').write_text(json.dumps(crosswalk,indent=2))
    # Published topology-specific physical benchmark solve outcomes, no rewriting.
    p=R/'57_pglib_stress_n1/full/ALL_RUNS.csv'
    if not p.exists():raise FileNotFoundError(p)
    df=pd.read_csv(p)
    counts=df.groupby(['expected_case','kind','classified'],dropna=False).size().reset_index(name='n')
    counts.to_csv(OUT/'ACOPF_SUCCESS_FAILURE_BREAKDOWN.csv',index=False)
    group=df.groupby('expected_case').agg(n=('success','size'),solved=('success','sum'))
    lines=[]
    lines.append('# Final physical-science claim boundaries')
    lines.append('')
    for r in source.itertuples():
        lines.append(f'- IEEE {r.network}: {r.rateA_valid}/{r.active} active branches with usable PGLib-v23.07 RATE_A; coverage {r.coverage:.2%}.')
    lines.append('')
    lines.append('**IEEE 300 full branch-thermal-limit verification remains blocked if coverage < 100%.**')
    lines.append('Do not replace zeros or 9900 with fabricated utility thermal limits.')
    lines.append('AC OPF solver success does not establish closed-loop CPPS safety; failure does not prove infeasibility.')
    lines.append('PGLib thermal ratings are reference benchmark data, not site-measured limits.')
    lines.append('Exp58 physical load interruption is an assumed cyber-to-power sensitivity mapping, not field-observed curtailment.')
    lines.append('')
    lines.append('## Original pandapower IEEE118 vs PGLib IEEE118 comparison')
    lines.append('```json')
    lines.append(json.dumps(crosswalk,indent=2))
    lines.append('```')
    lines.append('')
    lines.append('## All Exp57 statuses (not just feasible ones)')
    lines.append('```')
    lines.append(counts.to_string(index=False))
    lines.append('```')
    (OUT/'CLAIM_BOUNDARIES.md').write_text('\n'.join(lines))
    # Source coverage, no aesthetic manipulation.
    fig,ax=plt.subplots(figsize=(7.8,4.5))
    labels=[f'IEEE {i}' for i in source.network]
    pct=100*source.coverage.to_numpy(float)
    ax.barh(labels,pct,color=['#2C7786' if x>99.999 else '#C58B4C' for x in pct])
    ax.set_xlim(0,104)
    for i,x in enumerate(pct):ax.text(min(x+.65,102),i,f'{x:.2f}%',va='center',fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Usable rated in-service branches (%)')
    ax.set_title('Thermal-limit coverage of source-pinned PGLib cases',loc='left',pad=13)
    ax.grid(axis='x',alpha=.1)
    fig.tight_layout()
    for ext in ('pdf','svg','png'):fig.savefig(OUT/f'F61_thermal_rating_coverage.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    pack=OUT/'EXP61_PROVENANCE_AUDIT.zip'
    with ZipFile(pack,'w',ZIP_DEFLATED,compresslevel=6) as z:
        for f in OUT.iterdir():
            if f.is_file() and f!=pack:z.write(f,f.name)
        z.write(Path(__file__),'SOURCE/tnse_exp61.py')
    with ZipFile(pack) as z:
        if z.testzip():raise RuntimeError('ZIP corrupt')
    print('\nEXP61 physical rating provenance and crosswalk audit COMPLETE')
    print(source.to_string(index=False))
    print('\nCROSSWALK:\n',json.dumps(crosswalk,indent=2))
    print('\nExp57 statuses:\n',counts.to_string(index=False))
    print('\nOUTPUT:',OUT)

if __name__=='__main__':main()
