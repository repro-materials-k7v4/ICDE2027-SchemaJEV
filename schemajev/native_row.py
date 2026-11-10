"""Preserve original native option ordering and exact soft-label input contract."""
import math
def need(v,m):
 if not v:raise ValueError(m)
def calculate(torch,mathmod,method,task_type,logits,record):
 y=record['target_probability'];need(isinstance(y,list) and len(y)==logits.numel() and all(isinstance(v,(int,float)) and math.isfinite(v) and v>=0 for v in y) and abs(sum(y)-1)<=1e-6,'original native human soft CE list/shape/sum contract');target=torch.tensor(y,device=logits.device,dtype=torch.float32);values=record.get('score_values') if task_type=='score' and method in ('cdf_01','cdf_1') else None;loss,parts=mathmod.row_loss(torch,method,task_type,logits,target,values);return loss,parts,target
def native_parity(torch,mathmod,kernel,logits,record,params):
 native=kernel.human_soft_CE(torch,[logits],[record],target_field='target_probability');mean,_,target=calculate(torch,mathmod,'type_mean_ce','choice',logits,record);need(bool(torch.equal(native.detach(),mean.detach())),'exact native vs mechanism CE loss');a=torch.autograd.grad(native,params,allow_unused=True,retain_graph=True);b=torch.autograd.grad(mean,params,allow_unused=True,retain_graph=True);need(all((x is None and y is None) or (x is not None and y is not None and bool(torch.equal(x,y))) for x,y in zip(a,b)),'exact native vs mechanism CE gradients');return True
