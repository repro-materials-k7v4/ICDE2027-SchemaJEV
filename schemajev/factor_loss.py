"""Differentiable equal A/B relative-Frobenius teacher reconstruction.

Teacher factors are targets only. Forward factors must come from the actual
HNet(description) and the original native PCA decoder. Zero target factors use
squared absolute Frobenius error, with no hidden denominator floor.
"""
import math

def need(x,m):
 if not x:raise ValueError(m)

def loss(torch,predicted,target):
 need(set(predicted)==set(target)==set(range(24)),'same 24 original layers');totals={'A':None,'B':None};zero={'A':0,'B':0};count={'A':0,'B':0}
 for layer in range(24):
  need(set(predicted[layer])==set(target[layer]) and len(target[layer])==4,'same four original native targets')
  for name in sorted(target[layer]):
   need(len(predicted[layer][name])==len(target[layer][name])==2,'exact native A/B pair')
   for i,side in enumerate(('A','B')):
    a=predicted[layer][name][i].float();b=target[layer][name][i].detach().float();need(a.shape==b.shape and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()),'same finite native factors')
    err=(a-b).square().sum();den=b.square().sum();need(bool(torch.isfinite(err)) and bool(torch.isfinite(den)),'finite factor energies')
    if bool(den>0):v=err/den
    else:v=err;zero[side]+=1
    totals[side]=v if totals[side] is None else totals[side]+v;count[side]+=1
 need(count=={'A':96,'B':96},'exact 96+96 native factors')
 av=totals['A']/count['A'];bv=totals['B']/count['B'];total=(av+bv)/2.
 need(bool(torch.isfinite(total)),'finite reconstruction loss')
 return total,{'A_relative_squared_mean':float(av.detach().cpu()),'B_relative_squared_mean':float(bv.detach().cpu()),'zero_target_factors_A':zero['A'],'zero_target_factors_B':zero['B'],'absolute_squared_frobenius_for_zero_target':True,'factor_count':192}
