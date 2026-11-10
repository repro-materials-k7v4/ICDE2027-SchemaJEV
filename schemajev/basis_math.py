"""Uncentered per-layer factor PCA and total-teacher energy; no data IO."""
import math
import numpy as np

N=60
K=60
TYPES=('choice','binary','score')

def prior_weights(types):
    if len(types)!=N or any(types.count(t)!=20 for t in TYPES):
        raise ValueError('exactly 20 teachers of each type required')
    rng=np.random.default_rng(42)
    beta=np.zeros(N,dtype=np.float64)
    for t in TYPES:
        ix=[i for i,x in enumerate(types) if x==t]
        beta[ix]=rng.dirichlet(np.ones(20))/3
    return beta

def fit_layer(rows, k=K):
    """Rows [teacher, flattened concatenated native factors] in fixed order.

    For X=USVᵀ, scaled directions D=(S/√N)Vᵀ; Xᵀ(U/√N)=Dᵀ.
    """
    x=np.asarray(rows,dtype=np.float64)
    if x.ndim!=2 or x.shape[0]!=N or not np.isfinite(x).all():
        raise ValueError('60 finite raw factor rows required')
    u,s,vt=np.linalg.svd(x,full_matrices=False)
    tol=max(x.shape)*np.finfo(x.dtype).eps*s[0] if len(s) else 0
    rank=int(np.count_nonzero(s>tol))
    if rank<k: raise ValueError(f'unsupported: numerical rank {rank} < {k}')
    # Deterministic sign convention fixes orientation, including teacher mapping.
    u=u[:,:k].copy(); vt=vt[:k].copy(); s=s[:k].copy()
    for j in range(k):
        i=np.argmax(np.abs(u[:,j]))
        if u[i,j]<0: u[:,j]*=-1; vt[j]*=-1
    return {'U':u,'singular_values':s,'directions':vt*(s/math.sqrt(N))[:,None],
            'rms':s/math.sqrt(N),'rank':rank,'shape':x.shape}

def decode_numpy(beta, u, directions, z, rows):
    beta=np.asarray(beta); u=np.asarray(u); z=np.asarray(z)
    c=beta+u@z/math.sqrt(N)
    direct=beta@rows+z@directions
    return c,direct

def total_balance(torch, zs, us, beta):
    """zs: [task, layer, K], us: [layer, 60, K]. Mean over TRAIN tasks/layers.

    Average squared total coefficients over the 60 TRAIN tasks, normalize over teachers per layer, then average the layer penalties.
    """
    c=beta[None,None,:]+torch.einsum('lnk,tlk->tln',us,zs)/math.sqrt(N)
    energy=c.square().mean(dim=0); denom=energy.sum(dim=-1,keepdim=True)
    if bool((denom<=0).any().item()): raise ValueError('zero total coefficient energy')
    p=energy/denom
    return (N*p.square().sum(dim=-1)-1).mean()

def objective(torch, task_loss, lam, condition_forward, us, beta):
    if lam not in (0.,.01,0.1,1.): raise ValueError('unsupported lambda')
    if lam==0.: return task_loss, None
    penalty=total_balance(torch,condition_forward(),us,beta)
    return task_loss+lam*penalty, penalty

def backward_streamed_balance(torch, coordinate_for, task_ids, us, beta, coefficient):
    """Exact two-pass VJP of the all-60 description loss without keeping 60 graphs."""
    ids=list(task_ids)
    if len(ids)!=60 or ids!=sorted(set(ids)): raise ValueError('canonical 60 TRAIN descriptions required')
    if coefficient not in (0.,.01,.1,1.): raise ValueError('bad lambda')
    if coefficient==0.: return {'forward_calls':0,'loss':0.}
    values=[]
    for task in ids:
        z=coordinate_for(task)
        if z.shape!=(us.shape[0],60): raise ValueError('coordinate shape')
        values.append(z.detach().clone())
    detached=torch.stack(values).requires_grad_(True)
    loss=total_balance(torch,detached,us,beta)
    grad,=torch.autograd.grad(loss*coefficient,detached)
    for i,task in enumerate(ids):
        z=coordinate_for(task)
        if not torch.equal(z.detach(),values[i]): raise ValueError('nondeterministic coordinate replay')
        torch.autograd.backward(z,grad[i])
    return {'forward_calls':120,'loss':float(loss.detach())}
