"""Rebuild adapter-strength, injection-site, and runtime/resource figures."""
import argparse,csv,math
from pathlib import Path
from statistics import mean
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plot_style as S

D=Path(__file__).resolve().parent
TYPES=('choice','binary','score')
SCALES=(0,.25,.5,.75,1,1.25,1.5,2)
MODES=('backbone','encoder_only','head_only','full')

def rows(p):return list(csv.DictReader(open(p))) if p.exists() else []
def number(v):
 if v in ('',None):return None
 x=float(v);return x if math.isfinite(x) else None
def save(fig,out,name):
 with matplotlib.rc_context({'savefig.bbox':None,'savefig.pad_inches':0}):
  fig.savefig(out/(name+'.pdf'),bbox_inches=None,pad_inches=0);fig.savefig(out/(name+'.png'),dpi=300,bbox_inches=None,pad_inches=0)
 plt.close(fig)
def label(t):return 'Choice Acc. (%)' if t=='choice' else 'Binary Comp. (%)' if t=='binary' else 'Score NMAE ×100'
def value(r,key):return np.nan if r is None or number(r.get(key)) is None else 100*number(r[key])
def get(data,experiment,mode,cohort,t):
 b=[r for r in data if r['experiment']==experiment and (number(r['mode'])==mode if isinstance(mode,(float,int)) else r['mode']==mode) and r['cohort']==cohort and r['task_type']==t];assert len(b)==1;return b[0]

def adapter_plot(data,out,experiment,cohort):
 modes=SCALES if experiment=='strength' else MODES;x=np.array(modes,dtype=float) if experiment=='strength' else np.arange(4);fig,axes=plt.subplots(3,1,sharex=True,figsize=(S.COL_W,4.55),gridspec_kw={'hspace':.24})
 for ax,t in zip(axes,TYPES):
  b=[get(data,experiment,m,cohort,t) for m in modes];y=np.array([value(r,'primary_mean') for r in b]);sd=np.array([value(r,'primary_sample_SD') for r in b]);color=S.TYPE_COLOR[t]
  ax.plot(x,y,marker=S.TYPE_MARKER[t],color=color,linestyle='-' if experiment=='strength' else 'none');valid=np.isfinite(sd)&np.isfinite(y)
  if valid.any():ax.errorbar(x[valid],y[valid],yerr=sd[valid],fmt='none',color=color,elinewidth=.8,capsize=1.6)
  ax.set_ylabel(label(t));ax.grid(axis='y',color=S.GRID,linewidth=.5)
  for xx,r in zip(x,b):
   if int(r['primary_n'])==0:ax.text(xx,.02,'pending',rotation=90,transform=ax.get_xaxis_transform(),fontsize=6,color=S.GREY,ha='center')
 if experiment=='strength':
  b=[get(data,experiment,m,cohort,'choice') for m in modes];axes[-1].set_xticks(x,[('%g'%m)+'\nn='+r['n'] for m,r in zip(modes,b)],fontsize=6.5);axes[-1].set_xlabel('Effective adapter strength $s$');axes[-1].set_xlim(-.04,2.04)
 else:
  b=[get(data,experiment,m,cohort,'choice') for m in modes];axes[-1].set_xticks(x,[name+'\nn='+r['n'] for name,r in zip(('Backbone','Encoder','Decision','Full'),b)],fontsize=7);axes[-1].set_xlabel('Adapted components');axes[-1].set_xlim(-.25,3.25)
 fig.subplots_adjust(left=.185,right=.985,top=.985,bottom=.15);save(fig,out,'fig_'+('adapter_strength' if experiment=='strength' else 'injection_components')+'_'+cohort)

def efficiency(data,out):
 cold=rows(data/'efficiency_cold_stage_summary.csv');resident=rows(data/'efficiency_resident_timings.csv');storage=rows(data/'efficiency_cost_components.csv')
 if cold:
  names={'import_and_CUDA_context_wall_ms':'Import/CUDA','static_source_binding_wall_ms':'Source binding','exact_restore_SHA_IO_and_model_load_wall_ms':'Restore/SHA I/O','original_TRAIN_NUM_wall_ms':'TRAIN NUM','native_batch_preparation_wall_ms':'Batch preparation','adapter_generation_wall_ms':'Adapter generation','native_query_wall_ms':'Native query','cold_validated_response_wall_ms':'Complete cold response'};fig,ax=plt.subplots(figsize=(S.COL_W,2.85));y=np.arange(len(cold));v=[number(r['median'])/1000 for r in cold];p=[number(r['p90'])/1000 for r in cold];ax.barh(y,v,height=.6,color=[S.OURS if r['stage']=='cold_validated_response_wall_ms' else S.GREY for r in cold]);ax.plot(p,y,'|',color=S.INK,markersize=7);ax.set_yticks(y,[names[r['stage']] for r in cold],fontsize=7);ax.invert_yaxis();ax.set_xscale('log');ax.set_xlabel('Cold wall time (s)\nmedian and p90; n=3');ax.grid(axis='x',color=S.GRID);fig.subplots_adjust(left=.42,right=.98,top=.98,bottom=.17);save(fig,out,'fig_efficiency_cold')
 if resident:
  modes=('uncached','cached_feature','cached_coordinates','cached_materialized_adapter');b=[r for r in resident if r['section']=='resident_service_timings'];datasets=list(dict.fromkeys(r['dataset'] for r in b));fig,ax=plt.subplots(figsize=(S.COL_W,2.45));values=[]
  for ds in datasets:
   x=[number(next(r for r in b if r['dataset']==ds and r['stage']==mode)['wall_ms_median']) for mode in modes];ratio=np.array(x)/x[0];values.append(ratio);ax.plot(range(4),ratio,color=S.GREY_LIGHT,linewidth=.55,marker='.',markersize=2)
  ax.plot(range(4),np.mean(np.array(values),axis=0),color=S.OURS,marker='o',linewidth=1.5);ax.set_xticks(range(4),['Uncached','Feature','Coordinate','Adapter'],fontsize=7);ax.set_ylabel('Latency / same-task uncached');ax.set_xlabel('Resident cache mode');ax.grid(axis='y',color=S.GRID);fig.subplots_adjust(left=.19,right=.98,top=.97,bottom=.22);save(fig,out,'fig_efficiency_resident_cache')
  stage=[r for r in resident if r['section']=='stage_timings'];names=list(dict.fromkeys(r['stage'] for r in stage));fig,ax=plt.subplots(figsize=(S.COL_W,2.9))
  for i,name in enumerate(names):
   v=[number(r['wall_ms_median']) for r in stage if r['stage']==name];ax.scatter(v,[i]*len(v),s=8,color=S.GREY_LIGHT);ax.plot(mean(v),i,'D',color=S.OURS,markersize=3.6)
  labels={'description_tokenization_CPU':'Tokenization','description_encoder':'Description encoder','trunk_and_coordinate_heads':'Coordinate network','PCA_factor_decode':'PCA decode','factor_document_views_CPU':'Factor views','one_controller_context_entry_exit_CPU':'Controller entry/exit','native_query_with_materialized_adapters':'Native query'};ax.set_yticks(range(len(names)),[labels[n] for n in names],fontsize=7);ax.invert_yaxis();ax.set_xscale('log');ax.set_xlabel('Standalone stage wall time (ms)');ax.grid(axis='x',color=S.GRID);fig.subplots_adjust(left=.43,right=.98,top=.98,bottom=.17);save(fig,out,'fig_efficiency_stages')
 positive=[r for r in storage if number(r.get('physical_bytes')) is not None and number(r['physical_bytes'])>0]
 if positive:
  labels={'base':'Backbone','shared_PCA_basis':'Shared\nbasis','teacher_library_checkpoints':'Teacher\nlibrary','student_checkpoint':'Student'};fig,ax=plt.subplots(figsize=(S.COL_W,2.45));ax.bar(range(len(positive)),[number(r['physical_bytes'])/1024**2 for r in positive],color=[S.OURS if r['item']=='student_checkpoint' else S.GREY for r in positive]);ax.set_xticks(range(len(positive)),[labels[r['item']] for r in positive],fontsize=7);ax.set_yscale('log');ax.set_ylabel('Existing physical storage (MiB)');ax.grid(axis='y',color=S.GRID);fig.subplots_adjust(left=.19,right=.98,top=.97,bottom=.23);save(fig,out,'fig_efficiency_storage')

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=D/'data');p.add_argument('--output',type=Path,default=D/'figures');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);S.apply();adapter=rows(a.data/'adapter_cohort_type_mean_SD.csv')
 for cohort in ('ALL18','ID12','OOD6'):
  for experiment in ('component','strength'):adapter_plot(adapter,a.output,experiment,cohort)
 efficiency(a.data,a.output)

if __name__=='__main__':main()
