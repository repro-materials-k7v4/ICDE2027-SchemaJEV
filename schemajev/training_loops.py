"""Native reconstruction and task CE; descriptions, not teacher ID, enter forward."""
import hashlib,json,math,random,time

def need(value,message):
 if not value:raise ValueError(message)

def groups(torch,hyper,stage):
 named=[(n,p) for n,p in hyper.named_parameters() if p.requires_grad]
 heads=[p for n,p in named if n.startswith('coordinate_heads.')]
 trunk=[p for n,p in named if not n.startswith('coordinate_heads.')]
 need(len(heads)==48 and bool(trunk) and sum(p.numel() for _,p in named)==929984 and all(p.dtype==torch.float32 for _,p in named),'exact small model trainables and 24 heads')
 rates={'RECON':(.01,.0003),'CE':(.0003,.00003)};need(stage in rates,'declared phase')
 h,t=rates[stage];return named,torch.optim.AdamW([{'params':heads,'peak_lr':h},{'params':trunk,'peak_lr':t}],weight_decay=0.)

def schedule(ids):
 ids=sorted(ids);need(len(ids)==len(set(ids))==60,'exact TRAIN60');rng=random.Random(42);order=[]
 while len(order)<4000:
  epoch=ids.copy();rng.shuffle(epoch);order.extend(epoch)
 return order[:4000]

def reconstruct(torch,ctx,phase,factor_loss):
 need(phase in ('NUM','RECON') and len(ctx.teacher_order)==60 and set(ctx.teacher_order)==set(ctx.conditions) and tuple(ctx.coordinates.shape)==(24,60,60),'qualified teacher order and coordinates')
 named,opt=groups(torch,ctx.model.hypernet,'RECON');need(not opt.state,'fresh optimizer');order=schedule(ctx.teacher_order);index={x:i for i,x in enumerate(ctx.teacher_order)};steps=1 if phase=='NUM' else 1000;history=[];ctx.model.train();torch.cuda.synchronize();started=time.monotonic()
 try:
  for step in range(1,steps+1):
   ctx.lease.boundary();opt.zero_grad(set_to_none=True);batch=order[(step-1)*4:step*4];need(len(batch)==4,'fixed reconstruction batch')
   loss_a=loss_b=0.
   for dataset in batch:
    ctx.lease.boundary();predicted,_=ctx.model.generate_adapters(ctx.conditions[dataset]);i=index[dataset]
    with torch.no_grad(),ctx.model._autocast():
     z={f'{l:03d}':ctx.coordinates[l,i].reshape(1,60).to(device=ctx.device,dtype=torch.float32) for l in range(24)}
     target=ctx.model.hypernet.adapter_for_document(ctx.model.hypernet.decoder(z),0)
    value,info=factor_loss.loss(torch,predicted,target);(value/4.).backward();loss_a+=info['A_relative_squared_mean']/4.;loss_b+=info['B_relative_squared_mean']/4.
   need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for _,p in named),'all actual reconstruction gradients present/finite')
   norm=float(torch.nn.utils.clip_grad_norm_([p for _,p in named],1.,error_if_nonfinite=True));need(math.isfinite(norm) and (phase!='NUM' or norm>0),'finite actual reconstruction gradient')
   for group in opt.param_groups:group['lr']=group['peak_lr']*min(step/10.,1.)
   opt.step();history.append({'step':step,'A_loss':loss_a,'B_loss':loss_b,'preclip_grad_norm':norm})
  ctx.assert_frozen();torch.cuda.synchronize();elapsed=time.monotonic()-started;return {'elapsed_seconds':elapsed,'CUDA_synchronized':True,'complete':True,'steps':steps,'presentations':steps*4,'schedule_sha256':hashlib.sha256(json.dumps(order,separators=(',',':')).encode()).hexdigest(),'fresh_optimizer':True,'head_peak_lr':.01,'trunk_peak_lr':.0003,'warmup_steps':10,'KD_alpha':0.,'TEST_reads':0,'teacher_target_used_only_in_loss':True,'history':history}
 finally:
  opt.zero_grad(set_to_none=True);opt.state.clear()
  for group in opt.param_groups:group['params'].clear()

def task_ce(torch,ctx,course,proof,gradient_step,loss_math,native_row,phase):
 need(phase in ('NUM','CE') and proof.get('complete') is True and proof.get('course_canonical_sha256')=='0b8cf639a7c2eceace84ae4fbd43e343f802b3d53fcec627015c6ed458fe3c94' and len(course)==16000 and len({(r['dataset_id'],r['row_id']) for r in course})==16000,'qualified original exact 16k course')
 named,opt=groups(torch,ctx.model.hypernet,'CE');need(not opt.state,'fresh CE optimizer');state=loss_math.initial_ema_state();steps=1 if phase=='NUM' else 200;history=[];ctx.model.train();torch.cuda.synchronize();started=time.monotonic()
 try:
  for step in range(1,steps+1):
   ctx.lease.boundary();opt.zero_grad(set_to_none=True);window=course[(step-1)*80:step*80];need(len(window)==80,'exact 80 records')
   def losses(chunk):
    ctx.lease.boundary();need(1<=len(chunk)<=4 and all(x['dataset_id']==chunk[0]['dataset_id'] and x['task_type']==chunk[0]['task_type'] for x in chunk),'original type and dataset microbatch');d=chunk[0]['dataset_id'];u=ctx.units[d];need(d in ctx.conditions and ctx.fit_types[d]==chunk[0]['task_type'] and all(x['row_id'] in u['by_id'] for x in chunk),'exact FIT rows');indices=[u['by_id'][x['row_id']] for x in chunk];factors,prefix=ctx.model.generate_adapters(ctx.conditions[d]);answer=ctx.queries[d].query_logits([u['contracts'][i] for i in indices],factor_adapters=factors,prefix_tokens=prefix);need(answer.get('targets_in_model_payload') is False and len(answer['logits_record_order'])==len(chunk),'native query record count and no targets');return [native_row.calculate(torch,loss_math,'type_mean_ce',r['task_type'],z.float(),u['records'][i])[0] for r,i,z in zip(chunk,indices,answer['logits_record_order'])]
   state,diag=gradient_step.compute(torch,loss_math,[p for _,p in named],window,losses,'type_mean_ce',step,state)
   need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for _,p in named),'all actual CE gradients present/finite')
   norm=float(torch.nn.utils.clip_grad_norm_([p for _,p in named],1.,error_if_nonfinite=True));need(math.isfinite(norm) and (phase!='NUM' or norm>0),'finite actual CE gradient')
   for group in opt.param_groups:group['lr']=group['peak_lr']*min(step/10.,1.)
   opt.step();history.append({'step':step,'preclip_grad_norm':norm,'type_mean_losses':diag['type_mean_losses']})
  ctx.assert_frozen();torch.cuda.synchronize();elapsed=time.monotonic()-started;return {'elapsed_seconds':elapsed,'CUDA_synchronized':True,'complete':True,'steps':steps,'presentations':steps*80,'fresh_optimizer':True,'head_peak_lr':.0003,'trunk_peak_lr':.00003,'warmup_steps':10,'KD_alpha':0.,'TEST_reads':0,'teacher_target_used_in_forward':False,'history':history}
 finally:
  opt.zero_grad(set_to_none=True);opt.state.clear()
  for group in opt.param_groups:group['params'].clear()
