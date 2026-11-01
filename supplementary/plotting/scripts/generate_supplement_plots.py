from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
D=R/'data/54_graph_observation_delay/full_g5_s10_start17000'
OUT=R/'figures/new';OUT.mkdir(exist_ok=True,parents=True)
df=pd.read_csv(D/'raw_outcomes.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelsize':12,'legend.fontsize':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.14,'pdf.fonttype':42})
methods=['CDP','LOCAL_RANK','CF_STATIC','CF_LOCAL']
short={'CDP':'CDP','LOCAL_RANK':'Local rank','CF_STATIC':'CF static','CF_LOCAL':'CF local'}
colors=['#7C8A95','#208B82','#A188AD','#B95370']
keys=['topology','graph_seed','pattern','budget','onset','seed','mode','method']
wide=df.pivot(index=keys,columns='delay',values='PAS');wide['delta']=wide[3]-wide[0];wide=wide.reset_index()
fig,axes=plt.subplots(1,3,figsize=(11.4,4.7),sharex=True,sharey=True);rng=np.random.default_rng(54001)
for ax,top in zip(axes,['BA','ER','WS']):
 for i,m in enumerate(methods):
  values=wide.loc[(wide.topology==top)&(wide.method==m),'delta'].to_numpy();assert len(values)==1600
  v=ax.violinplot(values,positions=[i],vert=False,widths=.65,showextrema=False)
  for body in v['bodies']:body.set_facecolor(colors[i]);body.set_alpha(.25);body.set_edgecolor('none')
  ax.scatter(values[::8],i+rng.uniform(-.12,.12,len(values[::8])),s=5,color=colors[i],alpha=.18,edgecolors='none')
  ax.boxplot(values,positions=[i],vert=False,widths=.18,showfliers=False,patch_artist=True,boxprops={'facecolor':'white','edgecolor':colors[i]},medianprops={'color':colors[i]},whiskerprops={'color':colors[i]},capprops={'color':colors[i]})
 ax.set(title=top,yticks=range(4),yticklabels=[short[m] for m in methods],xlabel='PAS(delay 3) minus PAS(delay 0)');ax.axvline(0,c='#777',ls='--',lw=.8)
for i,ax in enumerate(axes):ax.text(.5,-.28,'('+chr(97+i)+')',transform=ax.transAxes,ha='center',weight='bold')
axes[0].invert_yaxis();fig.subplots_adjust(left=.11,right=.98,bottom=.25,top=.90,wspace=.20)
fig.savefig(OUT/'execution_delay.pdf');fig.savefig(OUT/'execution_delay.png',dpi=300);plt.close(fig)
methods=['CDP','STATIC_RANK','LOCAL_RANK','CF_STATIC','CF_LOCAL','CF_LOCAL_NO_GUARD']
labels=['CDP','Static rank','Local rank','CF static','CF local','CF local, no guard']
fig,axes=plt.subplots(1,2,figsize=(10.5,4.8),sharey=True)
for ax,field,title in zip(axes,['non_CDP_actions','action_changes'],['Non-CDP allocations per case','Allocation changes\nincluding first deployment']):
 bottom=np.zeros(6)
 for count,c in zip(range(4),['#DBE1E7','#93B4BD','#4E879A','#244D69']):
  fraction=np.array([(df.loc[df.method==m,field]==count).mean() for m in methods]);ax.barh(range(6),fraction,left=bottom,height=.67,label=str(count),color=c);bottom+=fraction
 assert np.allclose(bottom,1)
 ax.set(title=title,yticks=range(6),yticklabels=labels,xlabel='Fraction of 9600 paired contexts',xlim=(0,1))
axes[0].invert_yaxis()
for i,ax in enumerate(axes):ax.text(.5,-.25,'('+chr(97+i)+')',transform=ax.transAxes,ha='center',weight='bold')
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,title='Event count over three stages',loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.56,0))
fig.subplots_adjust(left=.20,right=.98,bottom=.30,top=.88,wspace=.20)
fig.savefig(OUT/'action_changes.pdf');fig.savefig(OUT/'action_changes.png',dpi=300);plt.close(fig)
print('Rebuilt two supplementary figures from exact matched raw rows; no outcome changes.')

