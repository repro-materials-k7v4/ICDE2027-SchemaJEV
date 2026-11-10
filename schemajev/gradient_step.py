"""One three-type window, accumulating each type mean without retained graphs."""
import math
TYPES=('choice','binary','score')
def need(x,m):
 if not x:raise ValueError(m)
def compute(torch,mathmod,params,rows,loss_for_chunk,method,step,ema_state):
 need(len(rows)==80 and len(params)>0,'exact 80 rows and trainable params');counts={t:sum(r['task_type']==t for r in rows) for t in TYPES};need(min(counts.values())>0 and max(counts.values())-min(counts.values())<=1,'balanced fixed batch');grads={};losses={}
 grads={t:[None for _ in params] for t in TYPES};losses={t:0. for t in TYPES}
 for p in params:p.grad=None
 # Consecutive same-task rows share one original descriptor and native
 # query. Preserve exact course order and cap original microbatch at four.
 at=0;micro_calls=0
 while at<len(rows):
  t=rows[at]['task_type'];dataset=rows[at]['dataset_id'];end=at+1
  while end<len(rows) and end-at<4 and rows[end]['task_type']==t and rows[end]['dataset_id']==dataset:end+=1
  chunk=rows[at:end];values=loss_for_chunk(chunk);need(isinstance(values,list) and len(values)==len(chunk),'one original loss per row');need(all(v.ndim==0 and bool(torch.isfinite(v).item()) for v in values),'finite own scalar losses');losses[t]+=sum(float(v.detach().cpu()) for v in values)/counts[t]
  aggregate=sum(values)/counts[t];sample=torch.autograd.grad(aggregate,params,allow_unused=True)
  for i,g in enumerate(sample):
   if g is not None:
    v=g.detach().float().cpu().clone();grads[t][i]=v if grads[t][i] is None else grads[t][i].add_(v)
  sample=None;aggregate=None;values=None;micro_calls+=1;at=end
 norms={t:mathmod.full_gradient_norm(torch,grads[t]) for t in TYPES};cos={}
 for i,a in enumerate(TYPES):
  for b in TYPES[i+1:]:
   dot=sum(float((x.double()*y.double()).sum()) for x,y in zip(grads[a],grads[b]) if x is not None and y is not None);den=norms[a]*norms[b];cos[a+'__'+b]=dot/den if den else None
 weights={t:1. for t in TYPES};updated=False;projected=None
 if method=='ema_gradnorm':ema_state,updated=mathmod.ema_weights(norms,ema_state,step);weights=ema_state['weights']
 if method=='pcgrad':combined,projected=mathmod.pcgrad_project(torch,grads,seed=42,step=step)
 else:
  combined=[]
  for i in range(len(params)):
   present=[(grads[t][i],weights[t]) for t in TYPES if grads[t][i] is not None];combined.append(None if not present else sum(x*w for x,w in present)/3.)
 for p,g in zip(params,combined):p.grad=None if g is None else g.to(device=p.device,dtype=p.dtype)
 need(all(p.grad is None or bool(torch.isfinite(p.grad).all().item()) for p in params),'finite combined gradients')
 return ema_state,{'type_mean_losses':losses,'type_gradient_norms':norms,'type_gradient_cosines':cos,'detached_weights':weights,'ema_updated':updated,'pcgrad':projected,'counts':counts,'native_forward_backward_micro_calls':micro_calls}
