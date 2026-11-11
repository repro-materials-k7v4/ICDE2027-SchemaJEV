from pathlib import Path
from figure_data import rows
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
O = Path(__file__).resolve().parents[1] / 'appendix'
O.mkdir(exist_ok=True)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':6.5,'axes.titlesize':7,'axes.labelsize':6.5,'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':5.5,'axes.linewidth':.55,'lines.linewidth':1,'lines.markersize':3,'errorbar.capsize':1.6,'grid.linewidth':.4,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
colors=['#336CA8','#BD7130','#328977'];types=['choice','binary','score'];labels=['Choice (%)','Binary comp. (%)','Score NMAE']
def base():
 fig,axs=plt.subplots(2,2,figsize=(3.5,2.85),layout='constrained');return fig,axs.ravel()
def finish(fig,name):
 for ext in ['png','svg','pdf']:fig.savefig(O / f'{name}.{ext}',dpi=420,facecolor='white')
 plt.close(fig)
def deco(ax,xlabel,ylabel):
 ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.yaxis.set_major_locator(MaxNLocator(4));ax.grid(axis='y',alpha=.25);ax.tick_params(length=2,width=.5)
rs=rows('ratio_primary_and_NLL_summary.csv');fig,axs=base()
for i,t in enumerate(types):
 z=sorted([r for r in rs if r['cohort']=='ALL18' and r['type']==t],key=lambda r:int(r['R_updates']));new=[r for r in z if int(r['n_seed'])==3];scale=1 if t=='score' else 100
 axs[i].errorbar([int(r['R_updates'])/1200 for r in new],[float(r['mean_primary'])*scale for r in new],yerr=[float(r['sample_SD_primary'])*scale for r in new],marker='o',color=colors[i],label='n=3')
 main=next(r for r in z if int(r['n_seed'])==2);axs[i].errorbar([5/6],[float(main['mean_primary'])*scale],yerr=[float(main['sample_SD_primary'])*scale],marker='o',mfc='white',mec='#555555',color='#555555',ls='none',label='Main, n=2');deco(axs[i],'R / 1200',labels[i]);axs[i].set_xticks([0,1/3,2/3,1],['0','1/3','2/3','1']);axs[i].set_title(['(a) Choice','(b) Binary','(c) Score'][i])
axs[0].legend(frameon=False,loc='best')
for i,t in enumerate(types):
 z=sorted([r for r in rs if r['cohort']=='ALL18' and r['type']==t and int(r['n_seed'])==3],key=lambda r:int(r['R_updates']));axs[3].plot([int(r['R_updates'])/1200 for r in z],[float(r['mean_NLL']) for r in z],marker='o',color=colors[i],label=t.title())
axs[3].set_xticks([0,1/3,2/3,1],['0','1/3','2/3','1']);deco(axs[3],'R / 1200','Native NLL');axs[3].set_title('(d) NLL tradeoffs');axs[3].legend(frameon=False);finish(fig,'ratio_paper_2x2')
rs=rows('K_dimension_macro.csv');fig,axs=base()
for i,t in enumerate(types):
 z=sorted([r for r in rs if r['task_type']==t],key=lambda r:int(r['K']));a=[r for r in z if int(r['n_student_seeds'])==3];b=[r for r in z if int(r['n_student_seeds'])==2];sc=1 if t=='score' else 100
 axs[i].errorbar([int(r['K']) for r in a],[float(r['mean'])*sc for r in a],yerr=[float(r['sample_SD'])*sc for r in a],marker='o',color=colors[i])
 axs[i].errorbar([int(r['K']) for r in b],[float(r['mean'])*sc for r in b],yerr=[float(r['sample_SD'])*sc for r in b],marker='o',mfc='white',mec='#555555',color='#555555',ls='none');axs[i].set_xscale('log',base=2);axs[i].set_xticks([1,4,16,60],['1','4','16','60']);deco(axs[i],'PCA dimensions K',labels[i]);axs[i].set_title(['(a) Choice','(b) Binary','(c) Score'][i])
axs[3].axis('off');axs[3].text(0,1,'Fixed N=60 teacher basis\nR1000 + CE200; rank64\nK1-32: 3 student seeds\nK60: 2 seeds (43/44)\nHollow marker: K60\nWhiskers: sample SD\nFixed18 panel',va='top',fontsize=6.5,linespacing=1.55);finish(fig,'K_dimension_paper_2x2')
