from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
d=pd.read_csv(R/'data/reliability/calibration_profiles.csv').set_index('method')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(7.0,4.1));x=np.arange(2);w=.32
ax.bar(x-w/2,d.loc[['Original','Balanced'],'K5_predicted_gain'],w,color='#3F6D91',label='Selection preview (5 rollouts)')
ax.bar(x+w/2,d.loc[['Original','Balanced'],'K20_independent_gain'],w,color='#C28A55',label='Independent check (20 rollouts)')
ax.axhline(0,color='#697B86',lw=.8)
ax.set(xticks=x,xticklabels=['Degree-prioritized\nfeature profile','Distributed-feature\nprofile'],ylabel='Mean paired PAS gain')
ax.legend(frameon=False,loc='lower center',bbox_to_anchor=(.5,1.02),fontsize=10);ax.grid(axis='y',alpha=.14);ax.set_axisbelow(True)
fig.tight_layout()
for ext in ['png','pdf','svg']:fig.savefig(R/('figures/supp_calibration.'+ext),dpi=300,bbox_inches='tight',pad_inches=.06)
plt.close(fig)
print('Calibration labels updated; both profile outcomes retained unchanged.')
