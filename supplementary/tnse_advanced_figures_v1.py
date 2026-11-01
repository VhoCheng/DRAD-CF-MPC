#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TNSE journal figure suite — verified-source, no fabricated outcomes.

Run in DRAD-CF-MPC-revision root:
    python tnse_advanced_figures_v1.py --all
    python tnse_advanced_figures_v1.py --only 01,02,03,12,15

Inputs: real Exp54, Exp57, Exp58, Exp60, Exp61 CSV/JSON.
Figures: vector PDF/SVG + high-resolution PNG. All numbers taken from inputs.
"""
import argparse
import hashlib
import json
import math
import sys
import warnings
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent
R = ROOT / 'revision_experiments' / 'results'
OUT = ROOT / 'revision_experiments' / 'figures' / 'TNSE_advanced_gallery'
F54 = R/'54_graph_observation_delay'/'full_g5_s10_start17000'
F57 = R/'57_pglib_stress_n1'/'full'
F58 = R/'58_cyber_power_sensitivity'/'full_g5_s2'
F60 = R/'60_physical_replication_audit'/'full_g5_s2'
F61 = R/'61_physical_provenance_audit'

TOP = ['BA', 'ER', 'WS']
MODES = ['oracle', 'partial80', 'partial50', 'blind']
METH = ['CDP','STATIC_RANK','LOCAL_RANK','CF_STATIC','CF_LOCAL','CF_LOCAL_NO_GUARD']
KEY = ['topology','graph_seed','pattern','budget','onset','seed','mode','delay']
NO_DELAY = [x for x in KEY if x!='delay']
COLOR = dict(navy='#234563', navy2='#476F8F', teal='#0B827D', cyan='#52AEB3',
             gold='#DA9C50', coral='#C66F67', lilac='#8E7BAF', gray='#97A6B0',
             ink='#23384A', light='#E9EFF2', line='#C6D1D7')
MCOLOR = dict(CDP=COLOR['gray'],STATIC_RANK=COLOR['lilac'],LOCAL_RANK=COLOR['cyan'],
              CF_STATIC=COLOR['gold'],CF_LOCAL=COLOR['teal'],CF_LOCAL_NO_GUARD=COLOR['coral'])
TCOLOR = dict(BA=COLOR['navy2'],ER=COLOR['teal'],WS=COLOR['gold'])
LABEL = dict(oracle='Oracle (100%)',partial80='Recall 80%',partial50='Recall 50%',blind='Blind (0%)')
DPI = 320
FIGS = []
DATA = {}


def style(theme):
    global COLOR, MCOLOR, TCOLOR
    if theme == 'grayscale':
        COLOR.update(navy='#292929',navy2='#4C4C4C',teal='#202020',cyan='#666666',
                     gold='#888888',coral='#AAAAAA',lilac='#999999',gray='#B1B1B1',
                     ink='#222222',light='#F3F3F3',line='#DDDDDD')
        MCOLOR.update(CDP='#B0B0B0',STATIC_RANK='#969696',LOCAL_RANK='#676767',
                      CF_STATIC='#767676',CF_LOCAL='#222222',CF_LOCAL_NO_GUARD='#4A4A4A')
        TCOLOR.update(BA='#202020', ER='#676767', WS='#A0A0A0')
    plt.rcParams.update({
        'font.family':'DejaVu Sans','font.size':8.6,'axes.labelsize':8.6,
        'axes.titlesize':10,'axes.titleweight':'medium','axes.edgecolor':COLOR['line'],
        'axes.labelcolor':COLOR['ink'],'text.color':COLOR['ink'],
        'xtick.color':COLOR['ink'],'ytick.color':COLOR['ink'],
        'axes.spines.right':False,'axes.spines.top':False,
        'axes.linewidth':.7,'xtick.major.width':.65,'ytick.major.width':.65,
        'grid.linewidth':.65,'figure.facecolor':'white','axes.facecolor':'white',
        'savefig.facecolor':'white','pdf.fonttype':42,'ps.fonttype':42,
        'svg.fonttype':'none','figure.dpi':120
    })


def read(path, required=None):
    if not path.is_file():
        raise FileNotFoundError(f'Required REAL source is missing: {path}')
    d = pd.read_csv(path)
    missing = set(required or [])-set(d.columns)
    if missing:
        raise ValueError(f'Missing columns in {path}: {sorted(missing)}')
    DATA[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return d


def finish(fig, idx, title, sources, note='', axes=None):
    """Subfigure letters go BELOW axes using explicit axes positions."""
    if axes is not None:
        fig.canvas.draw()
        axs = np.asarray(axes, dtype=object).ravel()
        for i, ax in enumerate(axs):
            if not ax.get_visible(): continue
            bb = ax.get_position()
            fig.text((bb.x0+bb.x1)/2, max(.018,bb.y0-.070),
                     f'({chr(97+i)})', ha='center',va='top',fontsize=9,
                     fontweight='bold',color=COLOR['ink'])
    name = f'F{idx:02d}_{title}'
    for ext in ('pdf','svg','png'):
        fig.savefig(OUT/f'{name}.{ext}',dpi=DPI,bbox_inches='tight',pad_inches=.19)
    plt.close(fig)
    FIGS.append(dict(id=f'{idx:02d}',figure=name,sources='; '.join(sources),note=note))
    print('[FIGURE]',name,flush=True)


def panel(ax, text):
    ax.set_title(text,loc='left',fontsize=9.5,pad=13,color=COLOR['ink'])
    ax.grid(axis='y',alpha=.14,zorder=0)
    ax.set_axisbelow(True)


def data54():
    if 'd54' in DATA: return DATA['d54']
    path = F54/'raw_outcomes.csv'
    d=read(path,KEY+['method','PAS','decision_compute_s','action_changes','terminal_I'])
    if d.duplicated(KEY+['method']).any(): raise ValueError('Duplicate Exp54 method-cases')
    x=d.pivot(index=KEY,columns='method',values='PAS')
    if set(METH)-set(x.columns) or x.isna().any().any(): raise ValueError('Exp54 pairing incomplete')
    if (d.PAS.isna().any() or not np.isfinite(d.PAS.to_numpy(float)).all()):
        raise ValueError('Invalid Exp54 PAS values')
    DATA['d54']=d
    DATA['w54']=x
    return d


def wide54():
    data54(); return DATA['w54']


def bootstrap_graph(g, nboot=2500, seed=54):
    """Uncertainty over GRAPH instances; each graph contributes mean of paired scenarios."""
    x=g.groupby('graph_seed',sort=True).delta.mean().to_numpy(float)
    mean=float(g.delta.mean())
    if len(x)<2: return mean,np.nan,np.nan,len(x)
    rng=np.random.default_rng(seed)
    means=x[rng.integers(0,len(x),(nboot,len(x)))].mean(axis=1)
    return mean,float(np.quantile(means,.025)),float(np.quantile(means,.975)),len(x)


def contrasts(a='CF_STATIC',b='CF_LOCAL'):
    w=wide54(); z=(w[a]-w[b]).rename('delta').reset_index()
    return z


def annotated_ci(ax, val,lo,hi,y,color,ms=5):
    ax.errorbar(val,y,xerr=[[max(0,val-lo)],[max(0,hi-val)]],fmt='o',
                ms=ms,lw=1.5,capsize=3,color=color,ecolor=color,zorder=3)


def raincloud(ax, arrays, names, palette, xlabel='', sample=200):
    """Half-violin + boxplot + deterministic subsampled raw dots.
    Dots are a visual subset; all arrays contribute to violin, median, and IQR.
    """
    rng=np.random.default_rng(3301)
    for j, values in enumerate(arrays):
        values=np.asarray(values,float)
        if not len(values):continue
        y=j+1
        if len(values)>2 and np.std(values)>1e-12:
            parts=ax.violinplot([values],positions=[y],vert=False,widths=.62,
                                showextrema=False,showmeans=False)
            for body in parts['bodies']:
                verts=body.get_paths()[0].vertices
                verts[:,1]=np.maximum(verts[:,1],y)
                body.set_facecolor(palette[j]);body.set_edgecolor('none');body.set_alpha(.28)
        picks=rng.choice(len(values),min(sample,len(values)),replace=False)
        ax.scatter(values[picks],y-.11-rng.random(len(picks))*.22,
                   s=6,color=palette[j],alpha=.23,rasterized=True,linewidth=0,zorder=2)
        q1,med,q3=np.quantile(values,[.25,.5,.75])
        ax.plot([q1,q3],[y+.01,y+.01],lw=6,color=palette[j],solid_capstyle='butt',zorder=4)
        ax.plot([med],[y+.01],marker='|',ms=14,mew=1.5,color='white',zorder=5)
        ax.plot([np.mean(values)],[y+.01],marker='D',ms=4,color=COLOR['ink'],zorder=6)
    ax.set_yticks(range(1,len(names)+1),names)
    ax.invert_yaxis();ax.set_xlabel(xlabel);ax.grid(axis='x',alpha=.16);ax.axvline(0,ls=':',lw=.8,color=COLOR['gray'])


def ecdf(ax,x,label,c):
    x=np.sort(np.asarray(x,float));ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',
        lw=1.75,label=label,color=c)


def f01():
    d=data54(); fig,axs=plt.subplots(3,2,figsize=(12.2,10.2),sharex=True,sharey=True)
    for i,top in enumerate(TOP):
        for j,delay in enumerate([0,3]):
            ax=axs[i,j]; grp=d[(d.topology==top)&(d.delay==delay)]
            for m in ('CDP','CF_STATIC','CF_LOCAL','CF_LOCAL_NO_GUARD'):
                s=grp[grp.method==m].groupby('mode').PAS.mean().reindex(MODES)
                ax.plot(range(4),s.values,'-o',ms=3.5,lw=1.45,label=m,color=MCOLOR[m])
            panel(ax,f'{top} network  |  delay = {delay} steps')
            if j==0: ax.set_ylabel('Mean PAS ↓')
            ax.set_xticks(range(4),['100%','80%','50%','0%'])
            ax.tick_params(axis='x',labelbottom=True)
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center',ncol=4,
               bbox_to_anchor=(.5,.027),frameon=False,fontsize=9)
    fig.text(.5,.093,'Infection-state recall (simulated)  |  five graph realizations per topology',
             ha='center',fontsize=8.4,color=COLOR['ink'])
    fig.subplots_adjust(bottom=.16,top=.96,left=.09,right=.98,hspace=.42,wspace=.18)
    finish(fig,1,'PAS_vs_observation_multifacet',['Exp54/raw_outcomes.csv'],axes=axs)


def f02():
    z=contrasts(); fig,axs=plt.subplots(1,2,figsize=(13.2,6.5),sharey=True)
    modes=['oracle','partial80','partial50']; labels=[f'{t} / {LABEL[m]}' for t in TOP for m in modes]
    for ax,delay in zip(axs,[0,3]):
        for i,t in enumerate(TOP):
            for j,m in enumerate(modes):
                s=z[(z.topology==t)&(z['mode']==m)&(z.delay==delay)]
                val,lo,hi,G=bootstrap_graph(s,seed=200+j+i*3+delay)
                annotated_ci(ax,val,lo,hi,i*3+j,TCOLOR[t])
        ax.set_yticks(np.arange(9),labels);ax.invert_yaxis();ax.axvline(0,ls='--',lw=1,color=COLOR['coral'])
        ax.set_xlabel('PAS(CF_STATIC) − PAS(CF_LOCAL) ↑');panel(ax,f'Actuation delay: {delay} simulation steps')
        ax.grid(axis='x',alpha=.16)
    fig.subplots_adjust(left=.23,right=.98,top=.9,bottom=.19,wspace=.24)
    finish(fig,2,'Forest_topology_observation',['Exp54/raw_outcomes.csv'],
           '95% graph-resampled descriptive CI, G=5 per topology',axes=axs)


def f03():
    z=contrasts();fig,axs=plt.subplots(1,2,figsize=(13.0,4.4),sharey=True)
    for ax,mode in zip(axs,['partial80','partial50']):
        sub=z[(z['mode']==mode)&(z.delay==0)]
        raincloud(ax,[sub[sub.topology==t].delta.to_numpy() for t in TOP],TOP,
                  [TCOLOR[t] for t in TOP],'Paired PAS reduction (positive favors CF_LOCAL)',250)
        ax.set_title(f'Observed infection recall: {mode[-2:]}%',loc='left',pad=14)
    fig.subplots_adjust(left=.1,right=.985,top=.84,bottom=.27,wspace=.25)
    finish(fig,3,'Raincloud_paired_PAS',['Exp54/raw_outcomes.csv'],
           'Dots are reproducible subsamples; violin and quartiles use all matched scenarios',axes=axs)


def f04():
    z=contrasts();fig,axs=plt.subplots(1,2,figsize=(12.3,4.35),sharey=True)
    for ax,delay in zip(axs,[0,3]):
        mat=np.zeros((3,3)); ci=np.zeros((3,3))
        for i,t in enumerate(TOP):
            for j,m in enumerate(['oracle','partial80','partial50']):
                s=z[(z.topology==t)&(z['mode']==m)&(z.delay==delay)]
                mu,lo,hi,_=bootstrap_graph(s,seed=541+i+j*3+delay)
                mat[i,j]=mu;ci[i,j]=lo
        maxvalue=.045
        for i in range(3):
            for j in range(3):
                val=mat[i,j];
                ax.scatter(j,i,s=250+16500*abs(val),color=COLOR['teal'] if val>=0 else COLOR['coral'],
                           alpha=.22+.65*min(1,abs(val)/maxvalue),
                           edgecolor=COLOR['ink'] if ci[i,j]>0 else COLOR['coral'],lw=1.1)
                ax.text(j,i,f'{val:+.3f}',ha='center',va='center',fontsize=8.4,fontweight='medium')
        ax.set_xlim(-.55,2.55);ax.set_ylim(2.55,-.55)
        ax.set_xticks(range(3),['Oracle','80%','50%']);ax.set_yticks(range(3),TOP)
        ax.grid(False);ax.set_aspect('equal');panel(ax,f'Delay: {delay} steps  |  bubble size reflects |gain|')
    fig.subplots_adjust(left=.09,right=.98,top=.87,bottom=.20,wspace=.32)
    finish(fig,4,'Bubble_effect_matrix',['Exp54/raw_outcomes.csv'],
           'Bubble size encodes effect magnitude, dark outline denotes positive CI lower bound',axes=axs)


def f05():
    w=wide54().reset_index();q=w[(w['mode']=='partial80')&(w.delay==0)]
    fig,axs=plt.subplots(1,3,figsize=(13.3,4.5),sharex=True,sharey=True)
    lo=min(q.CF_STATIC.min(),q.CF_LOCAL.min());hi=max(q.CF_STATIC.max(),q.CF_LOCAL.max())
    for ax,t in zip(axs,TOP):
        d=q[q.topology==t]
        h=ax.hexbin(d.CF_STATIC,d.CF_LOCAL,gridsize=25,cmap='Blues',mincnt=1,
                    linewidths=0.0,extent=(lo,hi,lo,hi))
        ax.plot([lo,hi],[lo,hi],'--',color=COLOR['coral'],lw=1)
        ax.set_title(f'{t}  |  n={len(d)} paired cases',loc='left',pad=14)
        ax.set_xlabel('CF_STATIC PAS');ax.set_xlim(lo,hi);ax.set_ylim(lo,hi);ax.set_aspect('equal',adjustable='box')
    axs[0].set_ylabel('CF_LOCAL PAS');fig.colorbar(h,ax=axs,shrink=.7,pad=.025,label='Scenario count per hexagon')
    fig.subplots_adjust(left=.09,right=.9,top=.85,bottom=.23,wspace=.22)
    finish(fig,5,'Hexbin_paired_outcome',['Exp54/raw_outcomes.csv'],
           'Each point is a matched scenario, below diagonal means CF_LOCAL has lower PAS',axes=axs)


def f06():
    z=contrasts();d=z[(z.delay==0)&(z['mode'].isin(['oracle','partial80','partial50']))]
    fig,ax=plt.subplots(figsize=(10.5,6.4));positions=[];ys=[];labs=[]
    for i,t in enumerate(TOP):
        for j,m in enumerate(['oracle','partial80','partial50']):
            vals=d[(d.topology==t)&(d['mode']==m)].delta.to_numpy(float)
            y=i*3+j
            if len(vals)>3 and np.std(vals)>1e-12:
                counts,edges=np.histogram(vals,bins=60,density=True)
                center=(edges[1:]+edges[:-1])/2
                smoothed=np.convolve(counts,np.ones(5)/5,mode='same')
                scale=smoothed/max(smoothed.max(),1e-12)*.72
                ax.fill_between(center,y,y+scale,color=TCOLOR[t],alpha=.16+(.09*j),edgecolor=TCOLOR[t],lw=1)
                ax.plot(center,y+scale,color=TCOLOR[t],lw=1.25)
            ax.plot(np.mean(vals),y,marker='D',color=TCOLOR[t],ms=4)
            ys.append(y);labs.append(f'{t} / {m}')
    ax.set_yticks(ys,labs);ax.invert_yaxis();ax.set_ylim(9,-.8)
    ax.axvline(0,color=COLOR['coral'],ls='--',lw=1);ax.set_xlabel('Matched PAS reduction: CF_STATIC − CF_LOCAL');
    ax.set_title('Ridgeline: paired improvement distributions  |  delay = 0',loc='left',pad=15)
    ax.grid(axis='x',alpha=.12)
    fig.subplots_adjust(left=.21,right=.97,top=.88,bottom=.15)
    finish(fig,6,'Ridgeline_gain_distributions',['Exp54/raw_outcomes.csv'],
           'Smoothed histograms; no parametric normality assumption')


def f07():
    d=data54();d=d[(d['mode']=='partial80')&(d.delay==0)]
    fig,axs=plt.subplots(1,3,figsize=(13.1,4.2),sharex=True,sharey=True)
    for ax,t in zip(axs,TOP):
        for m in ('CDP','CF_STATIC','CF_LOCAL'):
            ecdf(ax,d[(d.topology==t)&(d.method==m)].PAS,m,MCOLOR[m])
        ax.set_title(t,loc='left',pad=14);ax.set_xlabel('PAS (smaller is better)');ax.set_xlim(0,1)
        ax.grid(alpha=.12)
    axs[0].set_ylabel('Empirical CDF')
    fig.legend(*axs[0].get_legend_handles_labels(),bbox_to_anchor=(.5,.01),loc='lower center',
               ncol=3,frameon=False)
    fig.subplots_adjust(left=.085,right=.985,top=.85,bottom=.25,wspace=.20)
    finish(fig,7,'PAS_ECDF_three_topologies',['Exp54/raw_outcomes.csv'],axes=axs)


def f08():
    z=contrasts('CF_LOCAL_NO_GUARD','CF_LOCAL')
    fig,axs=plt.subplots(1,2,figsize=(12.3,5.8),sharey=True)
    cats=[(t,m) for t in TOP for m in ['oracle','partial80','partial50']]
    for ax,delay in zip(axs,[0,3]):
        for idx,(t,m) in enumerate(cats):
            g=z[(z.topology==t)&(z['mode']==m)&(z.delay==delay)]
            mu,lo,hi,_=bootstrap_graph(g,seed=831+idx*2+delay)
            annotated_ci(ax,mu,lo,hi,idx,TCOLOR[t])
        ax.set_yticks(range(len(cats)),[f'{t} / {m}' for t,m in cats]);ax.invert_yaxis()
        ax.axvline(0,color=COLOR['coral'],ls='--',lw=.9)
        ax.set_xlabel('PAS(No Guard) − PAS(Guard)');panel(ax,f'Delay {delay}  |  positive favors Guard')
        ax.grid(axis='x',alpha=.14)
    fig.subplots_adjust(left=.18,right=.98,top=.9,bottom=.2,wspace=.26)
    finish(fig,8,'Guard_effect_boundary',['Exp54/raw_outcomes.csv'],
           'Signed effects; negative results intentionally retained',axes=axs)


def f09():
    d=data54()
    dw=d.pivot(index=NO_DELAY+['method'],columns='delay',values='PAS')
    if not {0,3}.issubset(dw.columns): raise ValueError('Missing 0/3-delay pairs')
    u=(dw[3]-dw[0]).rename('delta').reset_index()
    fig,axs=plt.subplots(1,3,figsize=(14.4,4.85),sharey=True)
    for ax,t in zip(axs,TOP):
        v=u[(u.topology==t)&(u['mode']=='partial80')]
        arr=[v[v.method==m].delta.to_numpy() for m in ['CDP','CF_STATIC','CF_LOCAL','CF_LOCAL_NO_GUARD']]
        raincloud(ax,arr,['CDP','CF-STATIC','CF-LOCAL','CF-LOCAL no Guard'],
                  [MCOLOR[m] for m in ['CDP','CF_STATIC','CF_LOCAL','CF_LOCAL_NO_GUARD']],
                  'PAS(delay=3) − PAS(delay=0)',140)
        ax.set_title(t,loc='left',pad=13)
    fig.subplots_adjust(left=.15,right=.97,top=.84,bottom=.25,wspace=.33)
    finish(fig,9,'Execution_delay_raincloud',['Exp54/raw_outcomes.csv'],
           'Only modeled actuation steps, not measured communications latency',axes=axs)


def f10():
    z=contrasts();z=z[(z['mode']=='partial80')&(z.delay==0)]
    fig,axs=plt.subplots(2,2,figsize=(11.8,8.4),sharey=True)
    for ax,pattern in zip(axs.ravel(),['DDAC','RC','DDAC','RC']):
        row=0 if ax in axs[0] else 1
        budgets=[20,40]; q=z[(z.pattern==pattern)&(z.onset==[90,210][row])]
        xs=np.arange(len(budgets))
        for i,t in enumerate(TOP):
            v=q[q.topology==t].groupby('budget').delta.mean().reindex(budgets)
            ax.plot(xs,v.values,'-o',ms=5,color=TCOLOR[t],label=t,lw=1.8)
        ax.set_xticks(xs,[str(i) for i in budgets]);ax.axhline(0,color=COLOR['coral'],ls=':',lw=.9)
        ax.set_xlabel('Protection budget (nodes)');ax.set_ylabel('Mean paired PAS reduction')
        panel(ax,f'{pattern}  /  attack onset { [90,210][row] }')
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,
               frameon=False,bbox_to_anchor=(.5,.014))
    fig.subplots_adjust(left=.095,right=.985,top=.94,bottom=.15,wspace=.2,hspace=.41)
    finish(fig,10,'Budget_attack_pattern_sensitivity',['Exp54/raw_outcomes.csv'],axes=axs)


def f11():
    d=data54();q=d[(d['mode']=='partial80')&(d.delay==0)]
    fig,axs=plt.subplots(1,3,figsize=(13.4,4.5),sharex=True,sharey=True)
    names=['CDP','STATIC_RANK','LOCAL_RANK','CF_STATIC','CF_LOCAL','CF_LOCAL_NO_GUARD']
    for ax,t in zip(axs,TOP):
        g=q[q.topology==t].groupby('method').agg(meanPAS=('PAS','mean'),
                           runtime=('decision_compute_s','mean'),
                           p95runtime=('decision_compute_s',lambda x:x.quantile(.95))).reindex(names)
        for m,rec in g.iterrows():
            ax.scatter(rec.runtime+.00000001,rec.meanPAS,color=MCOLOR[m],s=58,
                       edgecolor='white',lw=.6,zorder=3)
        ax.set_title(t,loc='left',pad=14);ax.grid(alpha=.14)
        ax.set_xlabel('Controller compute time per scenario (s, log scale)')
        ax.set_xscale('symlog',linthresh=.0001)
    axs[0].set_ylabel('Mean PAS ↓')
    fig.legend([Line2D([0],[0],marker='o',color='none',markerfacecolor=MCOLOR[m],markersize=7)
                for m in names],names,loc='lower center',bbox_to_anchor=(.5,.01),ncol=3,frameon=False)
    fig.subplots_adjust(left=.09,right=.985,top=.85,bottom=.30,wspace=.18)
    finish(fig,11,'Runtime_effectiveness_Pareto',['Exp54/raw_outcomes.csv'],
           'Controller compute only; CDP timing includes Python overhead and is not zero comm latency',axes=axs)


def f12():
    d=data54();d=d[(d['mode'].isin(['partial80','partial50']))&(d.method.isin(['CDP','CF_STATIC','CF_LOCAL']))]
    fig,axs=plt.subplots(1,2,figsize=(11.8,4.8),sharey=True)
    for ax,mode in zip(axs,['partial80','partial50']):
        g=d[d['mode']==mode].groupby('method').action_changes
        names=['CDP','CF_STATIC','CF_LOCAL']
        vals=[g.get_group(m).to_numpy() for m in names]
        sns_size=350
        # Not a continuous density: discrete change counts are shown as jittered violins.
        raincloud(ax,vals,names,[MCOLOR[m] for m in names],'Number of stage-to-stage action changes',sns_size)
        ax.set_title(LABEL[mode],loc='left',pad=14)
    fig.subplots_adjust(left=.17,right=.985,top=.85,bottom=.25,wspace=.26)
    finish(fig,12,'Defense_action_change_distribution',['Exp54/raw_outcomes.csv'],
           'Discrete action changes; all observations used for quantiles',axes=axs)


def f13():
    d=read(F57/'ALL_RUNS.csv',['expected_case','kind','classified'])
    c=d.groupby(['expected_case','classified']).size().unstack(fill_value=0)
    categories=['SUCCESS_CHECKED','SUCCESS_WITH_UNVERIFIED_CONSTRAINTS',
               'SOLVER_FAILURE_OR_INFEASIBLE','ISLAND']
    for col in categories:
        if col not in c: c[col]=0
    c=c[categories].reindex([14,57,118,300],fill_value=0)
    fig,axs=plt.subplots(1,2,figsize=(12.4,4.5))
    ax=axs[0];bottom=np.zeros(4)
    cc=['#0F827E','#DBAA62','#C86D69','#697684']
    for k,col in enumerate(categories):
        x=c[col].to_numpy();ax.bar(np.arange(4),x,bottom=bottom,label=col,color=cc[k],width=.58)
        bottom+=x
    for j,total in enumerate(bottom):ax.text(j,total+.5,str(int(total)),ha='center',fontsize=9)
    ax.set_xticks(range(4),['IEEE 14','IEEE 57','IEEE 118','IEEE 300']);ax.set_ylim(0,max(bottom)*1.14)
    ax.set_ylabel('OPF/stress conditions');panel(ax,'Solver and constraint-check status')
    ax=axs[1];kind=d.groupby(['expected_case','kind']).size().unstack(fill_value=0).reindex([14,57,118,300],fill_value=0)
    ks=list(kind.columns);bottom=np.zeros(4)
    for j,k in enumerate(ks):
        ax.bar(np.arange(4),kind[k],bottom=bottom,width=.6,label=k,
               color=[COLOR['navy2'],COLOR['teal'],COLOR['gold'],COLOR['coral']][j%4]);bottom+=kind[k].to_numpy()
    ax.set_xticks(range(4),['IEEE 14','IEEE 57','IEEE 118','IEEE 300']);ax.set_ylabel('Test configurations');panel(ax,'Stress-design coverage')
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.02),ncol=2,frameon=False,fontsize=7.4)
    fig.subplots_adjust(left=.08,right=.98,top=.86,bottom=.28,wspace=.24)
    finish(fig,13,'Physical_OPF_outcome_stacks',['Exp57/full/ALL_RUNS.csv'],
           'Solver failure is not proof of infeasibility; IEEE300 thermal coverage incomplete',axes=axs)


def f14():
    p=F61/'PGLIB_RATING_COVERAGE.csv'
    d=read(p)
    fields=['network','active','rateA_valid','unknown','coverage']
    if set(fields)-set(d.columns): raise ValueError(f'Exp61 unexpected thermal schema: {d.columns.tolist()}')
    d=d.set_index('network').reindex([14,57,118,300])
    fig,axs=plt.subplots(1,2,figsize=(11.6,4))
    ax=axs[0]
    known=d.rateA_valid.to_numpy(float);unknown=d.unknown.to_numpy(float)
    ax.bar(range(4),known,color=COLOR['teal'],width=.58,label='Usable rating')
    ax.bar(range(4),unknown,bottom=known,color=COLOR['coral'],width=.58,label='Unverified rating')
    ax.set_xticks(range(4),['14','57','118','300']);ax.set_xlabel('IEEE bus count');ax.set_ylabel('Active branches')
    for j,(a,b) in enumerate(zip(known,unknown)):
        ax.text(j,a+b+6,f'{int(a)}/{int(a+b)}',ha='center',fontsize=8)
    ax.legend(frameon=False,fontsize=8);panel(ax,'Rating-source completeness')
    ax=axs[1];r=d.coverage.to_numpy(float)
    ax.hlines(range(4),0,r*100,color=COLOR['light'],lw=7)
    ax.scatter(r*100,range(4),s=75,color=[COLOR['teal'] if a>=1 else COLOR['coral'] for a in r],zorder=3)
    ax.set_yticks(range(4),['14','57','118','300']);ax.set_xlim(94,101.4);ax.set_xlabel('Usable thermal-rating coverage (%)');
    for j,a in enumerate(r):ax.text(a*100-.25,j-.23,f'{a*100:.2f}%',ha='right',fontsize=8)
    panel(ax,'Coverage lollipop')
    fig.subplots_adjust(left=.09,right=.98,top=.83,bottom=.23,wspace=.34)
    finish(fig,14,'Physical_thermal_rating_coverage',['Exp61/PGLIB_RATING_COVERAGE.csv'],
           'PGLib benchmark values are not measured utility field ratings',axes=axs)


def f15():
    d=read(F60/'CLUSTERED_PAIRED_EFFECTS.csv',['comparison','subset','mean_gain_MW',
                      'cluster_bootstrap_CI_low','cluster_bootstrap_CI_high'])
    shown=d[d['subset'].isin(['all_predeclared','interruption_0.10','interruption_0.30'])].copy()
    names=['all_predeclared','interruption_0.10','interruption_0.30']
    contrasts=['CF_LOCAL vs CDP','CF_LOCAL vs CF_STATIC']
    fig,axs=plt.subplots(1,2,figsize=(12.2,4.6),sharey=True)
    for ax,comparison,c in zip(axs,contrasts,[COLOR['teal'],COLOR['navy2']]):
        group=shown[shown.comparison==comparison].set_index('subset')
        for i,k in enumerate(names):
            if k not in group.index:continue
            r=group.loc[k];v=float(r.mean_gain_MW)
            lo=float(r.cluster_bootstrap_CI_low);hi=float(r.cluster_bootstrap_CI_high)
            annotated_ci(ax,v,lo,hi,i,c,ms=6)
            ax.text(hi+(.03*max(1,hi)),i,f'{v:+.1f} MW',va='center',fontsize=8)
        ax.set_yticks(range(3),['All prespecified','10% demand interruption','30% demand interruption'])
        ax.invert_yaxis();ax.axvline(0,ls='--',color=COLOR['coral'],lw=.8)
        ax.set_xlabel('Simulated interrupted-demand reduction (MW)')
        panel(ax,comparison)
        ax.grid(axis='x',alpha=.17)
    fig.subplots_adjust(left=.2,right=.96,top=.86,bottom=.24,wspace=.2)
    finish(fig,15,'Physical_MW_clustered_forest',['Exp60/CLUSTERED_PAIRED_EFFECTS.csv'],
           'Clustered CI; assumes prescribed infection-to-load interruption, not field measurements',axes=axs)


def f16():
    d=read(F58/'ALL_PHYSICAL_REPLAYS.csv',
        ['method','interrupted_demand_MW','demand_interruption_fraction','load_factor','topology'])
    d=d[d.method.isin(['CDP','CF_STATIC','CF_LOCAL'])]
    fig,axs=plt.subplots(1,2,figsize=(12.5,4.7),sharey=True)
    for ax,severity in zip(axs,[.1,.3]):
        w=d[np.isclose(d.demand_interruption_fraction,severity)]
        names=['CDP','CF_STATIC','CF_LOCAL'];arr=[w[w.method==m].interrupted_demand_MW.to_numpy() for m in names]
        raincloud(ax,arr,names,[MCOLOR[m] for m in names],
                  'Assumed interrupted demand (MW; lower better)',220)
        ax.set_title(f'Assumed interruption fraction: {int(severity*100)}%',loc='left',pad=12)
    fig.subplots_adjust(left=.16,right=.98,top=.85,bottom=.26,wspace=.27)
    finish(fig,16,'Physical_interrupted_demand_raincloud',['Exp58/ALL_PHYSICAL_REPLAYS.csv'],
           'Same attack trajectory repeated across load stress; dots visually subsampled',axes=axs)


def f17():
    d=read(F60/'CLUSTERED_PAIRED_EFFECTS.csv',
        ['comparison','subset','mean_gain_MW','cluster_bootstrap_CI_low','cluster_bootstrap_CI_high'])
    fig,axs=plt.subplots(1,2,figsize=(12.7,4.2),sharey=True)
    for ax,comparison in zip(axs,['CF_LOCAL vs CDP','CF_LOCAL vs CF_STATIC']):
        g=d[(d.comparison==comparison)&(d.subset.isin(['topology_BA','topology_ER','topology_WS']))].set_index('subset')
        for i,t in enumerate(TOP):
            if 'topology_'+t not in g.index:continue
            r=g.loc['topology_'+t];annotated_ci(ax,float(r.mean_gain_MW),
                float(r.cluster_bootstrap_CI_low),float(r.cluster_bootstrap_CI_high),i,TCOLOR[t],6)
        ax.set_yticks(range(3),TOP);ax.invert_yaxis();ax.axvline(0,ls='--',color=COLOR['coral'],lw=.8)
        ax.set_xlabel('Mean reduction in assumed interruption (MW)');panel(ax,comparison)
        ax.grid(axis='x',alpha=.15)
    fig.subplots_adjust(left=.10,right=.985,top=.86,bottom=.22,wspace=.24)
    finish(fig,17,'Physical_reduction_by_topology',['Exp60/CLUSTERED_PAIRED_EFFECTS.csv'],
           'Subgroup descriptions; no cross-model field guarantee',axes=axs)


def f18():
    """UpSet-style intersection of observable adverse events. Descriptive only."""
    d=data54();sel=d[(d['mode']=='partial80')&(d.delay==0)]
    ww=sel.pivot(index=KEY,columns='method',values='PAS')
    action=sel.pivot(index=KEY,columns='method',values='action_changes')
    terminal=sel.pivot(index=KEY,columns='method',values='terminal_I')
    # Case-level, mutually interpretable indicator conditions.
    sig=pd.DataFrame({
        'Local worse than static':ww.CF_LOCAL>ww.CF_STATIC+1e-12,
        'Local worse than CDP':ww.CF_LOCAL>ww.CDP+1e-12,
        'More switches than static':action.CF_LOCAL>action.CF_STATIC,
        'More infected than static':terminal.CF_LOCAL>terminal.CF_STATIC
    })
    names=list(sig.columns)
    codes=sig.astype(int).apply(lambda x:''.join(map(str,x)),axis=1)
    top=codes.value_counts().head(12)
    fig=plt.figure(figsize=(12.0,6.5))
    gs=GridSpec(2,1,height_ratios=[2.4,1.3],figure=fig,hspace=.07,left=.24,right=.97,top=.94,bottom=.16)
    ax=fig.add_subplot(gs[0]);bot=fig.add_subplot(gs[1],sharex=ax)
    ax.bar(range(len(top)),top.to_numpy(),color=COLOR['navy2'],width=.68)
    for i,n in enumerate(top):ax.text(i,n+max(top)*.02,str(n),ha='center',fontsize=7.5)
    ax.set_ylabel('Paired cases');ax.tick_params(axis='x',labelbottom=False)
    ax.set_title('UpSet-style intersections of adverse events | 80% recall, zero delay',loc='left',pad=13)
    bot.set_yticks(range(4),names);bot.set_ylim(3.5,-.5)
    for j,code in enumerate(top.index):
        marked=[i for i,ch in enumerate(code) if ch=='1']
        bot.scatter([j]*4,range(4),s=21,color=COLOR['light'],zorder=2)
        if marked:
            bot.plot([j,j],[min(marked),max(marked)],lw=1.6,color=COLOR['ink'])
            bot.scatter([j]*len(marked),marked,s=27,color=COLOR['teal'],zorder=4)
    bot.set_xticks(range(len(top)),[str(i+1) for i in range(len(top))]);bot.set_xlabel('Most frequent event combinations (ranked)')
    for spine in ('top','right','bottom'):bot.spines[spine].set_visible(False)
    ax.grid(axis='y',alpha=.13)
    finish(fig,18,'UpSet_adverse_event_intersections',['Exp54/raw_outcomes.csv'],
           'Four pre-defined descriptive binary events; no statistical significance claimed')


def f19():
    d=read(F58/'ALL_PHYSICAL_REPLAYS.csv',
           ['topology','graph_seed','pattern','budget','onset','seed','mode','delay',
            'method','load_factor','demand_interruption_fraction','interrupted_demand_MW'])
    index=['topology','graph_seed','pattern','budget','onset','seed','mode','delay',
           'load_factor','demand_interruption_fraction']
    w=d.pivot(index=index,columns='method',values='interrupted_demand_MW')
    if w.isna().any().any():raise ValueError('Exp58 physical pairing is incomplete')
    z=(w.CDP-w.CF_LOCAL).rename('gain_MW').reset_index()
    g=z.groupby(['topology','mode','demand_interruption_fraction'],as_index=False).gain_MW.mean()
    fig,axs=plt.subplots(1,3,figsize=(13.7,4.3),sharey=True)
    for ax,t in zip(axs,TOP):
        s=g[g.topology==t]
        for frac,ls in [(.1,'-o'),(.3,'--s')]:
            v=s[np.isclose(s.demand_interruption_fraction,frac)].set_index('mode').reindex(['partial80','partial50'])
            ax.plot(range(2),v.gain_MW,ls,label=f'{int(frac*100)}% interruption',
                    lw=2,color=COLOR['teal'] if frac==.1 else COLOR['gold'],ms=5)
        ax.axhline(0,ls=':',color=COLOR['coral']);ax.set_xticks(range(2),['80%','50%'])
        ax.set_xlabel('Observation recall');ax.set_title(t,loc='left',pad=12);ax.grid(axis='y',alpha=.16)
    axs[0].set_ylabel('Mean assumed interruption reduction (MW)')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.015),ncol=2,frameon=False)
    fig.subplots_adjust(left=.085,right=.985,top=.86,bottom=.29,wspace=.21)
    finish(fig,19,'Physical_stress_response_slopes',['Exp58/ALL_PHYSICAL_REPLAYS.csv'],
           'Descriptive means only; repeated load/interruption settings are correlated',axes=axs)


def f20():
    import networkx as nx
    # Rebuild exactly the topology definitions in Exp54, graph seed 42;
    # this depicts input networks, not optimized/recovered cyber-physical topology.
    n=118;gs=42
    G={
      'BA': nx.barabasi_albert_graph(n,2,seed=gs),
      'WS': nx.watts_strogatz_graph(n,4,.1,seed=gs)
    }
    if not nx.is_connected(G['WS']):
        for k in range(1,200):
            trial=nx.watts_strogatz_graph(n,4,.1,seed=gs+10007*k)
            if nx.is_connected(trial):G['WS']=trial;break
    for k in range(2000):
        trial=nx.gnm_random_graph(n,232,seed=gs+10007*k)
        if nx.is_connected(trial):G['ER']=trial;break
    if 'ER' not in G: raise RuntimeError('Cannot rebuild Exp54 ER network')
    fig,axs=plt.subplots(1,3,figsize=(13.5,4.5))
    for ax,t in zip(axs,TOP):
        gg=G[t];pos=nx.spring_layout(gg,seed=54,iterations=80)
        degree=np.array([gg.degree(i) for i in range(n)])
        nx.draw_networkx_edges(gg,pos,ax=ax,alpha=.16,width=.6,edge_color=COLOR['navy2'])
        nx.draw_networkx_nodes(gg,pos,ax=ax,nodelist=list(range(n)),
               node_size=8+degree*3,node_color=degree,cmap='viridis',
               vmin=0,vmax=max(18,degree.max()),edgecolors='white',linewidths=.2)
        ax.set_axis_off()
        ax.set_title(f'{t}: N={n}, E={gg.number_of_edges()}, clustering={nx.average_clustering(gg):.3f}',
                     loc='left',fontsize=9.3,pad=12)
    fig.subplots_adjust(left=.02,right=.98,top=.89,bottom=.07,wspace=.12)
    finish(fig,20,'Actual_graph_family_structures',['Exp54/tnse_exp54_final.py protocol'],
           'Graph families recreated using exact Exp54 generator/seed, positions are illustrative',axes=axs)


FIGFUN={i:globals()[f'f{i:02d}'] for i in range(1,21)}


def contact_sheet():
    """Contact sheet is a preview ONLY. Each individual PDF/SVG is print-ready."""
    try:
        from PIL import Image,ImageOps,ImageDraw,ImageFont
    except ImportError:
        print('[SKIP] Pillow missing: contact sheet not created');return
    files=[OUT/(x['figure']+'.png') for x in FIGS]
    if not files:return
    W=1100; H=620; pad=32; head=68; cols=3
    rows=math.ceil(len(files)/cols)
    canvas=Image.new('RGB',(cols*W,rows*(H+head)),(243,246,247))
    draw=ImageDraw.Draw(canvas)
    for j,path in enumerate(files):
        im=Image.open(path).convert('RGB');im.thumbnail((W-2*pad,H-2*pad))
        bx=(j%cols)*W;by=(j//cols)*(H+head)
        px=bx+(W-im.width)//2;py=by+head+(H-im.height)//2
        canvas.paste(im,(px,py));draw.text((bx+pad,by+19),FIGS[j]['figure'][:46],fill=(34,54,67))
    canvas.save(OUT/'00_ALL_20_FIGURES_CONTACT_SHEET.jpg',quality=90,optimize=True)


def manifest():
    pd.DataFrame(FIGS).to_csv(OUT/'FIGURE_MANIFEST.csv',index=False)
    (OUT/'SOURCE_SHA256.json').write_text(json.dumps({k:v for k,v in DATA.items() if k!='d54' and k!='w54'},indent=2))
    (OUT/'SUBMISSION_NOTES.txt').write_text(
      'REAL DATA ONLY. Source hashes in SOURCE_SHA256.json.\n'
      'Do not report bootstrap CI as power-grid-wide certification.\n'
      'Exp54 has five graph instances per topology; uncertainty at graph level is limited.\n'
      'Exp57 solver failure is not necessarily mathematical infeasibility.\n'
      'Exp58 load interruption is assumed (10%/30%), NOT measured utility outages.\n'
      'Exp61 pandapower-to-PGLib crosswalk FAILED_OR_UNAVAILABLE; equivalence is NOT proven.\n'
      'Some figures depict overlapping statistics; choose a nonredundant set for the main text.\n'
      'Plotting jitter or visual dot subsampling never changes computed statistical quantities.\n'
      'No synthetic model outcomes, stars, or data were generated.\n',encoding='utf-8')
    zipname=OUT/'TNSE_ADVANCED_FIGURES_ALL.zip'
    with ZipFile(zipname,'w',ZIP_DEFLATED,compresslevel=5,allowZip64=True) as z:
        for f in sorted(OUT.iterdir()):
            if f.is_file() and f!=zipname:
                z.write(f,f.name)
        z.write(Path(__file__),'SOURCE/tnse_advanced_figures_v1.py')
    with ZipFile(zipname) as z:
        problem=z.testzip()
        if problem:raise RuntimeError(f'Corrupted package member: {problem}')
    print('[PACKAGE]',zipname)


def main():
    global DPI
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--all',action='store_true')
    p.add_argument('--only',default='')
    p.add_argument('--theme',choices=['ocean','grayscale'],default='ocean')
    p.add_argument('--dpi',type=int,default=320)
    a=p.parse_args()
    DPI=a.dpi;style(a.theme);OUT.mkdir(parents=True,exist_ok=True)
    chosen=range(1,21) if (a.all or not a.only) else [int(v) for v in a.only.split(',')]
    if not chosen:raise ValueError('No figures selected')
    data54()
    errors=[]
    for i in chosen:
        if i not in FIGFUN:raise ValueError('Unknown figure number '+str(i))
        try:FIGFUN[i]()
        except Exception as ex:
            import traceback
            traceback.print_exc()
            errors.append(f'F{i:02d}: {ex}')
            print(f'[FAIL] F{i:02d}: {ex}',flush=True)
    contact_sheet()
    manifest()
    (OUT/'FAILED_FIGURES.txt').write_text('\n'.join(errors)+'\n' if errors else 'NONE\n')
    print(f'\nGENERATED: {len(FIGS)} / {len(chosen)}')
    print('OUTPUT:',OUT)
    if errors:
        print('WARNINGS:',errors)
        sys.exit(2)


if __name__=='__main__':main()
