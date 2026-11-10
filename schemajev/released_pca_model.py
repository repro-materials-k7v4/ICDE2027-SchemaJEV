"""Released native PCA decoder: frozen prior and frozen per-layer directions.

The base rank64 Common/PCA modules are the reusable native architecture source,
not a historical model, factor, basis, RNG or checkpoint.
"""
import math

def classes(torch,rank,native_core):
    nn=torch.nn
    _,BasePCA,_=native_core.classes(torch,rank)
    class Decoder(nn.Module):
        def __init__(self,geometry,manifest,arrays):
            super().__init__(); self.geometry=geometry; self.k=60; self.layer_keys=[f'{i:03d}' for i in range(geometry.layers)]; self.layout={}
            if len(self.layer_keys)!=24 or sorted(manifest['layers'])!=self.layer_keys: raise ValueError('24 native layers required')
            for i,key in enumerate(self.layer_keys):
                row=manifest['layers'][key]; a=arrays[key]; directions=torch.as_tensor(a['directions'],dtype=torch.float32); prior=torch.as_tensor(a['prior'],dtype=torch.float32)
                if directions.shape!=(60,prior.numel()) or row['rank']<60: raise ValueError('invalid directions')
                expected={(t.path,side):(t.in_features,geometry.generated_rank) if side=='A' else (geometry.generated_rank,t.out_features) for t in geometry.targets(i) for side in ('A','B')}
                layout=row['layout']
                if {(r['target'],r['side']) for r in layout}!=set(expected) or [tuple(x) for x in row['shapes']]!=[expected[(r['target'],r['side'])] for r in layout]: raise ValueError('factor layout mismatch')
                self.layout[key]=list(zip(layout,row['shapes']))
                self.register_buffer('D_'+key,directions); self.register_buffer('P_'+key,prior)
        def forward(self,coordinates):
            out={}
            for key in self.layer_keys:
                z=coordinates[key]
                if z.ndim!=2 or z.shape[1]!=60: raise ValueError('bad coordinates')
                values=getattr(self,'P_'+key)+z.float()@getattr(self,'D_'+key); pos=0
                index=int(key); pairs={}
                for entry,shape in self.layout[key]:
                    n=math.prod(shape); target,side=entry['target'],entry['side']; pairs.setdefault(target,{})[side]=values[:,pos:pos+n].reshape(z.shape[0],*shape); pos+=n
                if pos!=values.shape[1]: raise ValueError('factor layout not exhaustive')
                out[index]={target:(pair['A'],pair['B']) for target,pair in pairs.items()}
            return out
    class PCA(BasePCA):
        def __init__(self,geometry,decoder):
            super().__init__(geometry,decoder)
            with torch.no_grad():
                for h in self.coordinate_heads.values(): h.weight.zero_(); h.bias.zero_()
        def initialization_protocol(self):
            return {'prior':'fixed task-agnostic native factor mixture','coordinate_heads':'zero weights and biases','fresh_HNet':True}
    return PCA,Decoder
