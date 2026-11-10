"""Three-type loss alignment primitives. No sampling, model, CUDA setup or I/O.

PCGrad implements Yu et al. NeurIPS 2020, Algorithm 1; the returned mean
is a projected update, not the gradient of the original scalar objective.
Each type's input gradient must be the gradient of that type's mean loss.
"""
import hashlib,math,random
TYPES=('choice','binary','score')
METHODS=('type_mean_ce','ce_div_log_options','ema_gradnorm','pcgrad','cdf_01','cdf_1')
PAPER='https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf'
def need(v,m):
 if not v:raise ValueError(m)
def option_ce(torch,logits,target_probs,*,divide_log_options=False):
 need(logits.ndim==1 and logits.numel()>=2 and target_probs.shape==logits.shape,'one option vector with >=2 candidates')
 z=logits.to(dtype=torch.promote_types(logits.dtype,torch.float32));q=target_probs.detach().to(device=logits.device,dtype=z.dtype)
 need(bool(torch.isfinite(z).all()) and bool(torch.isfinite(q).all()) and bool((q>=0).all()) and abs(float(q.sum())-1.)<1e-5,'finite normalized nonnegative soft target')
 val=-(q*torch.log_softmax(z,dim=0)).sum()
 return val/math.log(logits.numel()) if divide_log_options else val

def score_cdf(torch,logits,target_probs,candidate_values):
 need(logits.ndim==1 and logits.numel()>=2 and logits.shape==target_probs.shape,'one score option vector')
 z=logits.to(dtype=torch.promote_types(logits.dtype,torch.float32));c=torch.as_tensor(candidate_values,device=logits.device,dtype=z.dtype).detach();q=target_probs.detach().to(device=logits.device,dtype=z.dtype)
 need(c.shape==logits.shape and bool(torch.isfinite(c).all()) and bool(torch.isfinite(logits).all()) and bool(torch.isfinite(q).all()) and bool((q>=0).all()) and abs(float(q.sum())-1.)<1e-5,'finite candidates and normalized nonnegative targets')
 c,order=torch.sort(c);need(bool((c[1:]>c[:-1]).all()),'distinct score values required');z=z[order];q=q[order]
 dx=c[1:]-c[:-1]
 delta=(torch.softmax(z,dim=0)-q).cumsum(0)[:-1]
 return (delta.square()*dx).sum()/(c[-1]-c[0])

def type_mean(torch,losses_by_type):
 need(set(losses_by_type)==set(TYPES) and all(isinstance(losses_by_type[t],(tuple,list)) and len(losses_by_type[t])>0 for t in TYPES),'all three nonempty types')
 means={t:torch.stack(losses_by_type[t]).mean() for t in TYPES}
 need(all(bool(torch.isfinite(x)) for x in means.values()),'finite type means')
 return sum(means.values())/3.,means

def ema_weights(norms,state,step,*,decay=.9,interval=10):
 need(type(step) is int and step>=1 and type(interval) is int and interval>=1 and 0<=decay<1,'valid step/cadence/decay')
 need(isinstance(state,dict) and set(state)=={'ema','weights','last_update_step'},'explicit detached EMA state')
 old=state['ema'];weights=state['weights'];last=state['last_update_step']
 need(type(last) is int and 0<=last<step and set(weights)==set(TYPES) and (old is None or set(old)==set(TYPES)),'strict causal EMA state')
 need(all(math.isfinite(float(weights[t])) and .25-1e-9<=float(weights[t])<=4+1e-9 for t in TYPES) and abs(sum(weights.values())-3)<1e-7,'bounded unit-mean weights')
 if step%interval:
  return dict(ema=None if old is None else dict(old),weights=dict(weights),last_update_step=last),False
 need(isinstance(norms,dict) and set(norms)==set(TYPES) and all(math.isfinite(float(norms[t])) and float(norms[t])>=0 for t in TYPES),'finite detached full-gradient norms')
 e={t:float(norms[t]) if old is None else decay*float(old[t])+(1-decay)*float(norms[t]) for t in TYPES}
 need(all(math.isfinite(v) and v>=0 for v in e.values()),'finite EMA norms')
 if all(v==0 for v in e.values()):w={t:1. for t in TYPES}
 else:
  # positive epsilon prevents undefined inverse at a zero-norm task; solve
  # for scale so clipping and mean-one hold simultaneously.
  raw={t:1./max(e[t],1e-12) for t in TYPES};lo,hi=0.,1.
  def total(scale):return sum(min(4.,max(.25,scale*raw[t])) for t in TYPES)
  while total(hi)<3.:hi*=2
  for _ in range(100):
   mid=(lo+hi)/2
   if total(mid)<3.:lo=mid
   else:hi=mid
  w={t:min(4.,max(.25,((lo+hi)/2)*raw[t])) for t in TYPES}
 return {'ema':e,'weights':w,'last_update_step':step},True

def initial_ema_state():return {'ema':None,'weights':{t:1. for t in TYPES},'last_update_step':0}

def pcgrad_project(torch,task_grads,*,seed,step):
 """Project against immutable original task gradients, then average tasks.

 task_grads: dict type -> same-length list[Tensor|None]; None = zero in the
 full trainable parameter vector. Output is None only if all tasks are None.
 The local Python PRNG does not touch global Python, NumPy, torch or CUDA RNG.
 """
 need(set(task_grads)==set(TYPES) and type(seed) is int and type(step) is int and step>=1,'three tasks and valid local seed/step');n=len(task_grads[TYPES[0]])
 need(n>0 and all(len(task_grads[t])==n for t in TYPES),'shared parameter layout');refs=[]
 for i in range(n):
  present=[task_grads[t][i] for t in TYPES if task_grads[t][i] is not None];refs.append(present[0] if present else None)
  for v in present:need(v.shape==present[0].shape and v.device==present[0].device and v.dtype==present[0].dtype and bool(torch.isfinite(v).all()),'finite matching parameter gradients')
 originals={t:[None if refs[i] is None else (torch.zeros_like(refs[i]) if task_grads[t][i] is None else task_grads[t][i].detach().clone()) for i in range(n)] for t in TYPES}
 def dot(a,b):
  terms=[(x.double()*y.double()).sum() for x,y in zip(a,b) if x is not None]
  return sum(terms) if terms else torch.tensor(0.,dtype=torch.float64)
 norms={t:float(dot(originals[t],originals[t]).sqrt()) for t in TYPES};cos={}
 for i,a in enumerate(TYPES):
  for b in TYPES[i+1:]:
   denom=norms[a]*norms[b];cos[a+'__'+b]=float(dot(originals[a],originals[b]))/denom if denom else None
 out={};order={};projections=0
 for t in TYPES:
  g=[None if x is None else x.clone() for x in originals[t]];others=[x for x in TYPES if x!=t];seedbytes=hashlib.sha256(f'pcgrad-v1:{seed}:{step}:{t}'.encode()).digest();random.Random(int.from_bytes(seedbytes,'big')).shuffle(others);order[t]=others
  for j in others:
   d=dot(g,originals[j]);den=dot(originals[j],originals[j]);need(bool(torch.isfinite(d)) and bool(torch.isfinite(den)),'finite projection')
   if float(d)<0 and float(den)>0:
    scale=d/den;g=[None if a is None else a-scale.to(dtype=a.dtype,device=a.device)*b for a,b in zip(g,originals[j])];projections+=1
  out[t]=g
 result=[None if refs[i] is None else sum(out[t][i] for t in TYPES)/3. for i in range(n)]
 need(all(x is None or bool(torch.isfinite(x).all()) for x in result),'finite projected update')
 return result,{'type_grad_norms':norms,'type_grad_cosines':cos,'projection_order':order,'projection_count':projections,'aggregation':'equal_type_mean','is_original_objective_gradient':False}

def install_grads(params,grads):
 need(len(params)==len(grads),'same parameter ordering')
 for p,g in zip(params,grads):
  need(g is None or (g.shape==p.shape and g.device==p.device),'matching parameter layout');p.grad=None if g is None else g.detach().to(dtype=p.dtype).clone()

def row_loss(torch,method,task_type,logits,target_probs,candidate_values=None):
 need(method in METHODS and task_type in TYPES,'declared recipe/type')
 ce=option_ce(torch,logits,target_probs,divide_log_options=method=='ce_div_log_options')
 if task_type=='score' and method in ('cdf_01','cdf_1'):
  need(candidate_values is not None,'original score candidate values required')
  cdf=score_cdf(torch,logits,target_probs,candidate_values)
  return ce+(.1 if method=='cdf_01' else 1.)*cdf,{'ce':ce.detach(),'cdf':cdf.detach()}
 return ce,{'ce':ce.detach(),'cdf':None}

def weighted_type_mean(torch,means,weights):
 need(set(means)==set(weights)==set(TYPES),'exact three type means and detached weights')
 vals={t:float(weights[t]) for t in TYPES};need(all(math.isfinite(v) and .25-1e-9<=v<=4+1e-9 for v in vals.values()) and abs(sum(vals.values())-3)<1e-7,'bounded detached unit-mean weights')
 return sum(means[t]*vals[t] for t in TYPES)/3.

def full_gradient_norm(torch,grads):
 terms=[]
 for g in grads:
  if g is not None:
   need(bool(torch.isfinite(g).all()),'finite full gradient');terms.append(g.detach().double().square().sum())
 return float(sum(terms).sqrt()) if terms else 0.
