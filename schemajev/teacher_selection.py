"""Independent CPU selector over complete fixed-mask DEV prediction rows.
Inputs are stored predictions with explicit canonical gold/option mappings. No GPU or model imports.
"""
import math, statistics
def require(value, message):
    if not value:
        raise ValueError(message)
GRID=tuple(range(25,201,25))
def pearson(a,b):
    ma=statistics.fmean(a); mb=statistics.fmean(b)
    x=[v-ma for v in a]; y=[v-mb for v in b]; den=math.sqrt(sum(v*v for v in x)*sum(v*v for v in y))
    return sum(i*j for i,j in zip(x,y))/den if den else None
def ranks(a):
    order=sorted(range(len(a)),key=a.__getitem__); out=[0.]*len(a); i=0
    while i<len(a):
        j=i+1
        while j<len(a) and a[order[j]]==a[order[i]]: j+=1
        for k in range(i,j): out[order[k]]=(i+j-1)/2
        i=j
    return out
def metrics(rows, kind):
    require(rows and len({r['row_id'] for r in rows})==len(rows), 'nonempty unique fixed mask')
    nll=[]; correct=[]; pred=[]; gold=[]
    for r in rows:
        p=r['probabilities']; target=r['target_probability']; require(len(p)==len(target)>1 and all(math.isfinite(x) and x>=0 for x in p+target) and abs(sum(p)-1)<1e-5 and abs(sum(target)-1)<1e-5,'valid probability distributions')
        require(all(v>0 or t==0 for v,t in zip(p,target)), 'undefined infinite NLL: no probability clipping'); nll.append(-math.fsum(t*math.log(v) for v,t in zip(p,target) if t>0))
        if kind=='score':
            s=r['score_values']; require(len(s)==len(p) and all(math.isfinite(v) for v in s),'fixed numeric support'); pred.append(math.fsum(x*y for x,y in zip(p,s))); gold.append(float(r['gold_score'])); require(math.isfinite(gold[-1]) and min(s)<=gold[-1]<=max(s) and abs(math.fsum(x*y for x,y in zip(target,s))-gold[-1])<=1e-6*max(1.,abs(min(s)),abs(max(s)),max(s)-min(s)), 'continuous gold and soft grid mismatch')
        else:
            opts=r['option_ids']; require(len(opts)==len(p) and len(set(opts))==len(opts),'explicit option IDs'); pred.append(opts[max(range(len(p)),key=p.__getitem__)]); gold.append(r['gold_option_id']); correct.append(pred[-1]==gold[-1])
    out={'count':len(rows),'nll':statistics.fmean(nll)}
    if kind=='choice': out['accuracy']=statistics.fmean(correct);out['utility']=out['accuracy']
    elif kind=='binary':
        require(all(set(r['option_ids'])=={'Yes','No'} for r in rows) and set(gold)=={'Yes','No'},'binary fixed support and both gold classes required')
        tp=sum(p=='Yes' and g=='Yes' for p,g in zip(pred,gold)); fp=sum(p=='Yes' and g=='No' for p,g in zip(pred,gold)); fn=sum(p=='No' and g=='Yes' for p,g in zip(pred,gold)); tn=sum(p=='No' and g=='No' for p,g in zip(pred,gold))
        f1=2*tp/(2*tp+fp+fn); ba=(tp/(tp+fn)+tn/(tn+fp))/2; acc=statistics.fmean(correct)
        out.update(accuracy=acc,yes_f1=f1,balanced_accuracy=ba,utility=(acc+f1+ba)/3)
    elif kind=='score':
        bounds={(min(r['score_values']),max(r['score_values'])) for r in rows}; require(len(bounds)==1,'fixed task score support'); lo,hi=next(iter(bounds)); require(hi>lo,'nonzero fixed score range'); mae=statistics.fmean(abs(x-y) for x,y in zip(pred,gold)); nm=mae/(hi-lo)
        out.update(mae=mae,range_nmae=nm,utility=1-nm,pearson=pearson(pred,gold),spearman=pearson(ranks(pred),ranks(gold)))
    else: raise ValueError('unknown task type')
    return out
def select(checkpoints, kind, baseline):
    require(set(checkpoints)==set(GRID), 'complete checkpoint grid')
    mask={(r['row_id']) for r in baseline}; require(len(mask)==len(baseline),'unique baseline mask')
    b=metrics(baseline,kind); all_metrics={}
    for step in GRID:
        rows=checkpoints[step]; require({r['row_id'] for r in rows}==mask and len(rows)==len(mask),'same fixed complete DEV mask')
        # Check gold and support identity as well as row identity.
        byid={r['row_id']:r for r in rows}
        for r in baseline:
            q=byid[r['row_id']]
            for k in ('target_probability','gold_score','gold_option_id','option_ids','score_values'):
                require(r.get(k)==q.get(k),'fixed DEV target/support mismatch')
        all_metrics[step]=metrics(rows,kind)
    best=min(GRID,key=lambda s:(-all_metrics[s]['utility'],all_metrics[s]['nll'],s)); gain=all_metrics[best]['utility']-b['utility']
    return {'selected_step':best,'selected_prefix_presentations':best*80,'completed_updates':200,'completed_presentations':16000,'baseline':b,'checkpoints':all_metrics,'gain':gain,'eligible':gain>0}
