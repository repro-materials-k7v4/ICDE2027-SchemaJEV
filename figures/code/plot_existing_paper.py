from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev
import argparse
import csv
import hashlib
import json
import math

import numpy as np
import style as S
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
FIG = ROOT / 'figures'
TABLE = ROOT / 'tables'
APP = ROOT / 'appendix'
TYPES = ('choice', 'binary', 'score')
KS = (1, 2, 4, 8, 16, 32, 60)
RS = (0, 200, 400, 600, 800, 1000, 1200)
MAIN = ('PROJECTED_B_sSEED', 'canonical_projected_repeat', 'CE200')
DATASETS = (
    ('scienceqa__choice', 'SciQA', 'ScienceQA', 'ID12', 'choice'),
    ('cosmosqa__choice', 'Cosmos', 'CosmosQA', 'ID12', 'choice'),
    ('social_iqa__choice', 'SIQA', 'SocialIQA', 'ID12', 'choice'),
    ('sick__choice', 'SICK', 'SICK', 'ID12', 'choice'),
    ('news_headlines_sarcasm__binary', 'Sarc.', 'Sarcasm', 'ID12', 'binary'),
    ('liar__binary', 'LIAR', 'LIAR', 'ID12', 'binary'),
    ('semeval2018_task3_irony__binary', 'Irony', 'Irony', 'ID12', 'binary'),
    ('measuring_hate_speech__binary', 'Hate', 'HateSpeech', 'ID12', 'binary'),
    ('holistic_dialogue__score', 'HDial.', 'HolDialog', 'ID12', 'score'),
    ('epitome__score', 'EPIT.', 'EPITOME', 'ID12', 'score'),
    ('cefr_sp__score', 'CEFR', 'CEFR-SP', 'ID12', 'score'),
    ('realsumm__score', 'RSum.', 'RealSumm', 'ID12', 'score'),
    ('scotus__choice', 'SCOT.', 'SCOTUS', 'OOD6', 'choice'),
    ('snips__choice', 'SNIPS', 'SNIPS', 'OOD6', 'choice'),
    ('mrpc__binary', 'MRPC', 'MRPC', 'OOD6', 'binary'),
    ('amazon_reviews2023__binary', 'Amz.-B', 'Amazon-B', 'OOD6', 'binary'),
    ('cmu_mosei__score', 'MOSEI', 'MOSEI', 'OOD6', 'score'),
    ('amazon_reviews2023__score', 'Amz.-S', 'Amazon-S', 'OOD6', 'score'),
)
IDS = [x[0] for x in DATASETS]
PANEL = {x[0]: dict(header=x[1], label=x[2], cohort=x[3], task_type=x[4]) for x in DATASETS}
METHODS = (
    ('Released bare', 'Backbone', 'Frozen backbone'),
    ('Bare + description', 'Backbone + desc.', 'Frozen backbone'),
    ('Mean delta W (60)', r'Mean $\Delta W$', 'Teacher-library reuse (60 teachers)'),
    ('Arrow routing (60, top 4)', 'Arrow routing', 'Teacher-library reuse (60 teachers)'),
    ('Ordinary LoRA (1200, 96k)', 'LoRA (1,200 updates)', 'Unified multi-task LoRA'),
    ('Ordinary LoRA (1x)', 'LoRA (1,591 updates)', 'Unified multi-task LoRA'),
    ('Direct T2L (Indep-128)', 'Indep-128', 'Direct T2L, raw-factor targets'),
    ('Direct T2L (FiLM-512)', 'FiLM-512', 'Direct T2L, raw-factor targets'),
    ('Direct T2L (Indep-512)', 'Indep-512', 'Direct T2L, raw-factor targets'),
    ('Direct T2L (rank64, 0.50M)', 'Canonical, 0.50M', 'Direct T2L, canonical targets'),
    ('Direct T2L (rank64, 0.93M)', 'Canonical, 0.93M', 'Direct T2L, canonical targets'),
    ('Direct T2L (rank64, 2.00M)', 'Canonical, 2.00M', 'Direct T2L, canonical targets'),
    ('PCA-T2L (K60, 0.93M)', r'\method{}', 'PCA-T2L'),
)


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def save_csv(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (DATA / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def stats(values):
    return dict(mean=mean(values), sd=stdev(values) if len(values) > 1 else None, n=len(values))


def save(fig, name, folder=FIG):
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / (name+'.pdf'), facecolor='white')
    fig.savefig(folder / (name+'.png'), dpi=300, facecolor='white')
    plt.close(fig)


def recipe(row):
    return row['recipe_key'], row['family'], row['stage']


def macro(rows, key, cohort, typ):
    subset = [r for r in rows if recipe(r) == key and r['task_type'] == typ and
              (cohort == 'ALL18' or r['cohort'] == cohort)]
    out = {}
    for seed in (42, 43, 44):
        selected = [r for r in subset if r['seed'] == seed]
        assert len(selected) == (6 if cohort == 'ALL18' else 4 if cohort == 'ID12' else 2)
        out[seed] = mean(r['primary'] for r in selected)
    return out


def band(ax, x, values, color, marker, linestyle='-', face=None):
    centers = np.array([mean(v.values()) for v in values])*100
    sd = np.array([stdev(v.values()) for v in values])*100
    ax.fill_between(x, centers-sd, centers+sd, color=color, alpha=.14, linewidth=0)
    ax.plot(x, centers, color=color, marker=marker, markersize=3,
        markerfacecolor=color if face is None else face, linestyle=linestyle,
        markeredgewidth=.7, zorder=3)
    return centers, sd


def frame(ax):
    ax.grid(axis='y', color=S.GRID, linewidth=.5)


def dimension(rows, inventory):
    keys = {int(r['active_K']): recipe(r) for r in inventory if r['family'] == 'canonical_projected_kgrid'}
    keys[60] = MAIN
    fig, axes = plt.subplots(1, 2, figsize=(S.COL_W, 2.10), gridspec_kw=dict(wspace=.32))
    records = []
    for ax, cohort, title in zip(axes, ('ID12', 'OOD6'), ('(a) ID held-out', '(b) OOD task views')):
        for typ in TYPES:
            anchor = macro(rows, keys[1], cohort, typ)
            values = [macro(rows, keys[k], cohort, typ) for k in KS]
            sign = -1 if typ == 'score' else 1
            changes = [{s: sign*(v[s]-anchor[s]) for s in v} for v in values]
            band(ax, KS, changes, S.TYPE_COLOR[typ], S.TYPE_MARKER[typ],
                '-' if cohort == 'ID12' else '--', None if cohort == 'ID12' else 'white')
            for k, v, delta in zip(KS, values, changes):
                for seed in v:
                    records.append(dict(K=k, cohort=cohort, task_type=typ, seed=seed,
                        raw_primary=v[seed], signed_change_from_K1=delta[seed]))
        ax.set_xscale('log', base=2)
        ax.set_xticks([1, 4, 16, 60], ['1', '4', '16', '60'])
        ax.set_xticks([2, 8, 32], minor=True)
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlim(.8, 75)
        ax.set_xlabel('Dimension $K$')
        ax.set_ylabel('Change from $K=1$ (pp)' if cohort == 'ID12' else '')
        ax.set_title(title, loc='left')
        ax.set_box_aspect(1)
        frame(ax)
    handles = [Line2D([], [], color=S.TYPE_COLOR[t], marker=S.TYPE_MARKER[t],
        label=S.TYPE_NAME[t] if t != 'score' else 'NMAE reduction') for t in TYPES]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(.5, .005), fontsize=6.5)
    fig.subplots_adjust(left=.15, right=.99, top=.97, bottom=.24)
    save(fig, 'fig_capacity')
    save_csv('K_dimension_per_seed_plot_data.csv', records)
    fig, axes = plt.subplots(1, 3, figsize=(S.COL_W, 1.44), gridspec_kw=dict(wspace=.55))
    for ax, typ, letter in zip(axes, TYPES, 'abc'):
        for cohort, ls, face in [('ID12', '-', None), ('OOD6', '--', 'white')]:
            band(ax, KS, [macro(rows, keys[k], cohort, typ) for k in KS],
                S.TYPE_COLOR[typ], S.TYPE_MARKER[typ], ls, face)
        ax.set_xscale('log', base=2)
        ax.set_xticks([1, 4, 16, 60], ['1', '4', '16', '60'])
        ax.set_xlim(.8, 75)
        ax.set_xlabel('Dimension $K$')
        ax.set_ylabel(r'NMAE $\times100$' if typ == 'score' else 'Accuracy (%)' if typ == 'choice' else 'Composite (%)')
        ax.set_title('('+letter+') '+S.TYPE_NAME[typ], loc='left')
        frame(ax)
    fig.legend(handles=[Line2D([], [], color=S.GREY_DARK, marker='o', label='ID12'),
        Line2D([], [], color=S.GREY_DARK, marker='o', markerfacecolor='white', linestyle='--', label='OOD6')],
        ncol=2, loc='lower center', bbox_to_anchor=(.5, -.20))
    fig.subplots_adjust(left=.12, right=.99, top=.90, bottom=.30)
    save(fig, 'fig_capacity_raw_type_panels', APP)


def allocation(rows, inventory, baselines):
    keys = {int(r['R_updates']): recipe(r) for r in inventory if r['family'] == 'canonical_projected_ratio'}
    keys[1000] = MAIN
    fig, axes = plt.subplots(2, 2, figsize=(S.COL_W, 2.58), gridspec_kw=dict(hspace=.65, wspace=.45))
    records = []
    for ax, typ, letter in zip(axes.flat, TYPES, 'abc'):
        values = [macro(rows, keys[r], 'ALL18', typ) for r in RS]
        band(ax, RS, values, S.TYPE_COLOR[typ], S.TYPE_MARKER[typ])
        current = mean(values[5].values())*100
        ax.plot(1000, current, '*', color=S.OURS, markersize=6, markeredgecolor='white', markeredgewidth=.4, zorder=4)
        base = [r for r in baselines if r['method'] == 'Ordinary LoRA (1200, 96k)' and r['task_type'] == typ]
        reference = mean(mean(float(r['primary']) for r in base if int(r['seed']) == s) for s in (42,43,44))*100
        ax.axhline(reference, color=S.GREY, linestyle=(0,(3,1.5)), linewidth=.8)
        ax.set_ylabel(r'NMAE $\times100$' if typ == 'score' else 'Accuracy (%)' if typ == 'choice' else 'Composite (%)')
        ax.set_title('('+letter+') '+S.TYPE_NAME[typ], loc='left')
        for r, v in zip(RS, values):
            records += [dict(R_updates=r, CE_updates=1200-r, task_type=typ, cohort='ALL18', seed=s, primary=v[s]) for s in v]
        frame(ax)
    ax = axes[1,1]
    times = read_csv(DATA/'training_phase_minutes.csv')
    rm = [mean(float(z['R_minutes']) for z in times if int(z['R_updates']) == r) for r in RS]
    cm = [mean(float(z['CE_minutes']) for z in times if int(z['R_updates']) == r) for r in RS]
    total = [[float(z['student_minutes']) for z in times if int(z['R_updates']) == r] for r in RS]
    ax.bar(RS, cm, width=140, color='#5B6B7F', label='CE phase')
    ax.bar(RS, rm, width=140, bottom=cm, color='#B9C4D1', label='R phase')
    ax.errorbar(RS, list(map(mean,total)), yerr=list(map(stdev,total)), fmt='none', ecolor=S.INK, elinewidth=.6, capsize=1.3)
    ax.set_ylabel('Student phase time (min)')
    ax.set_title('(d) Training time', loc='left')
    ax.legend(loc='upper right', fontsize=6, handlelength=.8)
    frame(ax)
    for ax in axes.flat:
        ax.set_xticks([0,400,800,1200])
        ax.set_xticks([200,600,1000], minor=True)
        ax.set_xlim(-80,1280)
        ax.set_xlabel('Reconstruction updates $R$')
    fig.subplots_adjust(left=.16, right=.99, top=.95, bottom=.13)
    save(fig,'fig_allocation')
    save_csv('allocation_ALL18_per_seed_plot_data.csv', records)


def capacity(rows, inventory):
    settings = {'PCA-T2L': {int(r['trainable_parameters']): recipe(r) for r in inventory if r['family'] == 'pca_capacity'},
        'Direct T2L': {int(r['trainable_parameters']): recipe(r) for r in inventory if r['family'] == 'direct_capacity' and r['stage'] == 'CE200'}}
    settings['PCA-T2L'][929984] = MAIN
    fig, axes = plt.subplots(2,2,figsize=(S.COL_W,2.63),gridspec_kw=dict(hspace=.60,wspace=.45))
    records = []
    for ax, typ, letter in zip(axes.flat,TYPES,'abc'):
        for name in ('PCA-T2L','Direct T2L'):
            pp = sorted(settings[name])
            values = [macro(rows,settings[name][p],'ALL18',typ) for p in pp]
            color = S.OURS if name == 'PCA-T2L' else S.GREY_DARK
            band(ax,np.array(pp)/1e6,values,color,S.TYPE_MARKER[typ],'-' if name == 'PCA-T2L' else '--',None if name == 'PCA-T2L' else 'white')
            for p,v in zip(pp,values):records += [dict(method=name,parameters=p,task_type=typ,seed=s,primary=v[s]) for s in v]
        ax.set_xticks([.5,.93,2],['0.50','0.93','2.00'])
        ax.set_xlabel('Generator parameters (M)')
        ax.set_ylabel(r'NMAE $\times100$' if typ == 'score' else 'Accuracy (%)' if typ == 'choice' else 'Composite (%)')
        ax.set_title('('+letter+') '+S.TYPE_NAME[typ],loc='left')
        frame(ax)
    axes[1,1].axis('off')
    axes[1,1].legend(handles=[Line2D([],[],color=S.OURS,marker='o',label='PCA-T2L'),
        Line2D([],[],color=S.GREY_DARK,marker='o',markerfacecolor='white',linestyle='--',label='Direct T2L')],loc='upper left',fontsize=7)
    fig.subplots_adjust(left=.16,right=.99,top=.95,bottom=.13)
    save(fig,'fig_network_capacity')
    save_csv('network_capacity_per_seed_plot_data.csv',records)


def library(rows, inventory):
    fig, axes = plt.subplots(1,3,figsize=(S.TEXT_W,1.75),gridspec_kw=dict(wspace=.35))
    records=[]
    for ax,typ in zip(axes,TYPES):
        for n,marker,color in [(64,'o',S.TYPE_COLOR[typ]),(128,'s',S.GREY_DARK)]:
            settings={int(r['active_K']):recipe(r) for r in inventory if r['family']=='canonical_projected_N'+str(n)}
            kk=sorted(settings)
            vv=[macro(rows,settings[k],'ALL18',typ)for k in kk]
            band(ax,kk,vv,color,marker,'-' if n==64 else '--')
            for k,v in zip(kk,vv):records += [dict(teachers=n,K=k,task_type=typ,seed=s,primary=v[s])for s in v]
        ax.set_xticks([60,64,128],['60','64','128'])
        labels=ax.get_xticklabels();labels[0].set_ha('right');labels[1].set_ha('left')
        ax.set_xlim(55,133)
        ax.set_xlabel('Retained dimension $K$')
        ax.set_title(S.TYPE_NAME[typ],loc='left')
        ax.set_ylabel(r'NMAE $\times100$'if typ=='score'else 'Accuracy (%)'if typ=='choice'else 'Composite (%)')
        frame(ax)
    fig.legend(handles=[Line2D([],[],color=S.GREY_DARK,marker='o',label='64-teacher basis'),Line2D([],[],color=S.GREY_DARK,marker='s',linestyle='--',label='128-teacher basis')],loc='lower center',ncol=2,bbox_to_anchor=(.5,-.08))
    fig.subplots_adjust(left=.07,right=.99,top=.91,bottom=.28)
    save(fig,'teacher_library_dimension_ALL18',APP)
    save_csv('teacher_library_dimension_per_seed_plot_data.csv',records)


def geometry():
    rows=read_csv(DATA/'main_geometry_seed_cells.csv')
    assert {int(r['seed'])for r in rows}=={42,43,44}
    matrices={}
    records=[]
    for space in ('description_feature','coordinates','effective_deltaW','factor_A','factor_B'):
        mat=np.zeros((18,18))
        for i,di in enumerate(IDS):
            for j,dj in enumerate(IDS):
                a=[float(r['cosine'])for r in rows if r['space']==space and r['dataset_i']==di and r['dataset_j']==dj]
                assert len(a)==3
                mat[i,j]=mean(a)
                records.append(dict(space=space,dataset_i=di,dataset_j=dj,n=3,mean_cosine=mean(a),sample_SD=stdev(a)))
        matrices[space]=mat
    cmap=LinearSegmentedColormap.from_list('paper_cosine',['#2F6CAB','#9DBCDD','#F7F7F7','#E7A6A0','#B2182B'])
    def draw(spaces,name):
        fig,axes=plt.subplots(1,len(spaces),figsize=(S.TEXT_W,2.50),gridspec_kw=dict(wspace=.18))
        for ax,(space,title)in zip(axes,spaces):
            mat=matrices[space];mask=np.triu(np.ones_like(mat,dtype=bool),k=1)
            im=ax.imshow(np.ma.array(mat,mask=mask),vmin=-1,vmax=1,cmap=cmap,aspect='equal',interpolation='nearest')
            ax.set_xticks(range(18),[PANEL[d]['header']for d in IDS],rotation=90,fontsize=5.4)
            ax.set_yticks(range(18),[PANEL[d]['header']for d in IDS],fontsize=5.4)
            ax.tick_params(length=0,pad=1)
            ax.axhline(11.5,color=S.GREY_DARK,linewidth=.5);ax.axvline(11.5,color=S.GREY_DARK,linewidth=.5)
            ax.set_title(title,loc='left')
            for spine in ax.spines.values():spine.set_visible(False)
        fig.subplots_adjust(left=.06,right=.94,top=.89,bottom=.22)
        cb=fig.colorbar(im,ax=list(axes),fraction=.017,pad=.015,shrink=.80)
        cb.set_label('Cosine similarity',fontsize=7)
        save(fig,name)
    draw([('description_feature','(a) Description embeddings'),('coordinates','(b) Generated coordinates'),('effective_deltaW',r'(c) Effective updates $\Delta W$')],'fig_geometry')
    draw([('effective_deltaW',r'(a) Effective updates $\Delta W$'),('factor_A','(b) Factor $A$'),('factor_B','(c) Factor $B$')],'fig_factor_geometry')
    save_csv('geometry_mean_sample_SD_plot_data.csv',records)


def tables(baselines):
    per={}
    for method,_,_ in METHODS:
        b=[r for r in baselines if r['method']==method]
        seeds=sorted({int(r['seed'])for r in b})
        assert len(seeds)==(1 if method in [m[0]for m in METHODS[:4]]else 3)
        for ds in IDS:
            a=[float(r['primary'])for r in b if r['dataset']==ds];per[method,ds]=stats(a)
        for typ in TYPES:
            a=[mean(float(r['primary'])for r in b if int(r['seed'])==s and r['task_type']==typ)for s in seeds]
            per[method,typ]=stats(a)
    columns=IDS+list(TYPES)
    marks={}
    for col in columns:
        typ=PANEL[col]['task_type']if col in PANEL else col
        vals=sorted(set(round(per[m,col]['mean']*100,2)for m,_,_ in METHODS),reverse=typ!='score')
        for m,_,_ in METHODS:
            value=round(per[m,col]['mean']*100,2)
            marks[m,col]='best'if value==vals[0]else 'second'if value==vals[1]else None
    def cell(m,col,sd=False):
        v=per[m,col];s=f"{100*v['mean']:.2f}"
        if marks[m,col]=='best':s=r'\textbf{'+s+'}'
        elif marks[m,col]=='second':s=r'\underline{'+s+'}'
        if sd and v['sd']is not None:s=r'\shortstack{'+s+r'\\[-1pt]{\tiny$\pm$'+f"{100*v['sd']:.2f}"+'}}'
        return s
    def main(sd=False):
        label='tab:main-sd'if sd else 'tab:main'
        caption=('Per-view mean and sample SD of the fixed checkpoints. 'if sd else 'Per-view comparison of the fixed checkpoints. ')+r'Choice: accuracy (\%); Binary: $(\mathrm{Acc}+\mathrm{F1}_{\mathrm{Yes}}+\mathrm{BAcc})/3$ (\%); Score: NMAE ($\times100$, lower is better). Trained methods use seeds 42, 43 and 44; frozen and library-reuse controls have $n=1$ and no defined sample SD. ID12 has four held-out views per type; OOD6 has two task-transfer views per type and does not imply corpus-disjoint transfer. LoRA uses 1,200 or 1,591 CE updates; the default PCA and canonical Direct schedule is $R1000+CE200$. All methods use the same per-view masks and native option ordering. \textbf{Bold} and \underline{underline} indicate the best and second-best displayed means.'
        lines=[r'\begin{table*}[!t]',r'\centering',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\scriptsize',r'\setlength{\tabcolsep}{1.4pt}',r'\renewcommand{\arraystretch}{1.08}',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{@{}l'+'c'*21+r'@{}}',r'\toprule',r' & \multicolumn{12}{c}{ID held-out views (ID12)} & \multicolumn{6}{c}{OOD task-transfer views (OOD6)} & \multicolumn{3}{c}{ALL18 macro} \\',r'\cmidrule(lr){2-13}\cmidrule(lr){14-19}\cmidrule(l){20-22}',r' & \multicolumn{4}{c}{Choice $\uparrow$} & \multicolumn{4}{c}{Binary $\uparrow$} & \multicolumn{4}{c}{Score $\downarrow$} & \multicolumn{2}{c}{Ch.$\uparrow$} & \multicolumn{2}{c}{Bin.$\uparrow$} & \multicolumn{2}{c}{Sc.$\downarrow$} & Ch.$\uparrow$ & Bin.$\uparrow$ & Sc.$\downarrow$ \\',r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}\cmidrule(lr){14-15}\cmidrule(lr){16-17}\cmidrule(lr){18-19}',r'Method & '+' & '.join(PANEL[d]['header']for d in IDS)+r' & & & \\']
        counts={r['dataset']:int(r['records'])for r in baselines}
        lines += [r'{\color{gray}\# eval} & '+' & '.join(str(counts[d]).replace(',','{,}')for d in IDS)+r' & & & \\']
        group=None
        for m,name,g in METHODS:
            if group!=g:
                lines += [r'\midrule',r'\multicolumn{22}{@{}l}{\textit{'+g+r'}} \\'];group=g
            prefix=r'\rowcolor{oursbg}'if m==METHODS[-1][0]else ''
            lines += [prefix+name+' & '+' & '.join(cell(m,col,sd)for col in columns)+r' \\']
        lines += [r'\bottomrule',r'\end{tabular}}',r'\end{table*}']
        (TABLE/('tab_main_sd.tex'if sd else 'tab_main.tex')).write_text('\n'.join(lines)+'\n')
    main();main(True)
    abl=read_csv(DATA/'component_ablation_mean_SD.csv');order=list(dict.fromkeys(r['method']for r in abl))
    config=[('Backbone','',''),('Unified LoRA','0','1200'),('PCA','1200','0'),('PCA','0','1200'),('PCA','1000','200'),('Direct','1200','0'),('Direct','0','1200'),('Direct','1000','200')]
    for sd in (False,True):
        caption=r'Component ablation with 1,200 nominal student updates. R is teacher-parameter reconstruction and CE is task cross-entropy. PCA and Direct use approximately 0.93M generator parameters. Values are dataset-equal macro means; trained methods have $n=3$, and Backbone has $n=1$. Score is NMAE $\times100$.'
        lines=[r'\begin{table*}[!t]'if sd else r'\begin{table}[!t]',r'\centering',r'\caption{'+caption+'}',r'\label{tab:ablation'+('-sd'if sd else '')+'}',r'\scriptsize',r'\setlength{\tabcolsep}{1.6pt}',r'\resizebox{'+(r'\textwidth'if sd else r'\columnwidth')+r'}{!}{%',r'\begin{tabular}{@{}lcc'+'c'*9+r'@{}}',r'\toprule',r' & & & \multicolumn{3}{c}{ALL18} & \multicolumn{3}{c}{ID12} & \multicolumn{3}{c}{OOD6} \\',r'\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(l){10-12}',r'Generator & R & CE & '+' & '.join([r'Ch.$\uparrow$',r'Bin.$\uparrow$',r'Sc.$\downarrow$']*3)+r' \\',r'\midrule']
        for index,(old,(gen,r,ce))in enumerate(zip(order,config)):
            vals=[]
            for cohort in ('ALL18','ID12','OOD6'):
                for typ in TYPES:
                    z=next(x for x in abl if x['method']==old and x['cohort']==cohort and x['task_type']==typ)
                    s=f"{100*float(z['mean']):.2f}"
                    rank=sorted(set(round(100*float(x['mean']),2)for x in abl if x['cohort']==cohort and x['task_type']==typ),reverse=typ!='score')
                    value=round(100*float(z['mean']),2)
                    if value==rank[0]:s=r'\textbf{'+s+'}'
                    elif value==rank[1]:s=r'\underline{'+s+'}'
                    if sd and z['sd']:s += r'\,$\pm$\,'+f"{100*float(z['sd']):.2f}"
                    vals.append(s)
            if index in (2,5):lines.append(r'\midrule')
            lines.append((r'\rowcolor{oursbg}'if index==4 else '')+gen+' & '+r+' & '+ce+' & '+' & '.join(vals)+r' \\')
        lines += [r'\bottomrule',r'\end{tabular}}',r'\end{table*}'if sd else r'\end{table}']
        (TABLE/('tab_ablation_sd.tex'if sd else 'tab_ablation.tex')).write_text('\n'.join(lines)+'\n')


def gain(baselines):
    fig,axes=plt.subplots(1,3,figsize=(S.TEXT_W,1.85),gridspec_kw=dict(wspace=.40))
    records=[]
    for ax,typ in zip(axes,TYPES):
        ids=[d for d in IDS if PANEL[d]['task_type']==typ]
        for i,ds in enumerate(ids):
            p={int(r['seed']):float(r['primary'])for r in baselines if r['method']==METHODS[-1][0]and r['dataset']==ds}
            l={int(r['seed']):float(r['primary'])for r in baselines if r['method']=='Ordinary LoRA (1200, 96k)'and r['dataset']==ds}
            assert set(p)==set(l)=={42,43,44}
            values=[(p[s]-l[s])*(-1 if typ=='score'else 1)*100 for s in (42,43,44)]
            m,sd=mean(values),stdev(values);color=S.TYPE_COLOR[typ]
            ax.barh(i,m,height=.62,facecolor='white'if PANEL[ds]['cohort']=='OOD6'else color,edgecolor=color,linewidth=.8)
            ax.errorbar(m,i,xerr=sd,fmt='none',color=S.GREY_DARK,elinewidth=.7,capsize=1.5)
            records.append(dict(dataset=ds,cohort=PANEL[ds]['cohort'],task_type=typ,n=3,signed_gain_mean_pp=m,sample_SD_pp=sd))
        ax.set_yticks(range(6),[PANEL[d]['label']for d in ids],fontsize=6)
        ax.invert_yaxis();ax.axvline(0,color=S.GREY_DARK,linewidth=.6);ax.axhline(3.5,color=S.GREY_LIGHT,linestyle='--',linewidth=.5)
        ax.set_title(S.TYPE_NAME[typ],loc='left');ax.set_xlabel('NMAE reduction (pp)'if typ=='score'else 'Primary-metric change (pp)');ax.grid(axis='x',color=S.GRID,linewidth=.5)
    fig.legend(handles=[Patch(facecolor=S.GREY_DARK,edgecolor=S.GREY_DARK,label='ID held-out'),Patch(facecolor='white',edgecolor=S.GREY_DARK,label='OOD task views')],loc='lower center',ncol=2,bbox_to_anchor=(.5,-.06))
    fig.subplots_adjust(left=.10,right=.99,top=.90,bottom=.25)
    save(fig,'fig_gain_over_lora');save_csv('paired_gain_vs_unified_LoRA1200.csv',records)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT);a=ap.parse_args()
    assert a.output.resolve()==ROOT
    S.apply();FIG.mkdir(exist_ok=True);TABLE.mkdir(exist_ok=True)
    rows=read_csv(DATA/'all_family_seed_dataset_results.csv');inventory=read_csv(DATA/'inventory_completion.csv')
    assert len(rows)==1458 and len(inventory)==81 and all(r['status']=='complete'for r in inventory)
    for r in rows:r['seed']=int(r['seed']);r['primary']=float(r['primary'])
    baselines=read_csv(DATA/'main_baseline_admitted_seed_dataset.csv')
    assert len(baselines)==558 and len({r['method']for r in baselines})==13
    tables(baselines);dimension(rows,inventory);allocation(rows,inventory,baselines);capacity(rows,inventory);library(rows,inventory);geometry();gain(baselines)


if __name__=='__main__':
    main()
