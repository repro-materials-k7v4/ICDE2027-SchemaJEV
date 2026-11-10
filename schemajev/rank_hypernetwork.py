"""Rank-sweep copy of original SHINE: only generated rank and shared512 memory change.
All direct factors retain the original sqrt(.001) multiplier, with no alpha/r.
The original meta rank remains128. No generated factor is zero padded.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F

EXPECTED_TRAINABLE = 49_499_136
EXPECTED_GENERATED = 1_886_208

@dataclass(frozen=True)
class ProjectionSpec:
    path: str
    in_features: int
    out_features: int
    meta_rank: int

@dataclass(frozen=True)
class ArchitectureSpec:
    hidden_size: int = 768
    encoder_layers: int = 22
    decision_layers: int = 2
    encoder_intermediate_size: int = 1152
    decision_intermediate_size: int = 3072
    memory_tokens: int = 256
    generated_rank: int = 8
    meta_rank: int = 128
    narrow_meta_cap: int = 16
    heads: int = 12
    feedforward_size: int = 1536
    fixture: bool = False

    @property
    def layers(self):
        return self.encoder_layers + self.decision_layers

    def targets(self, layer):
        if not 0 <= layer < self.layers:
            raise ValueError("Layer outside architecture")
        h = self.hidden_size
        if layer < self.encoder_layers:
            f = self.encoder_intermediate_size
            dims = {"attn.Wqkv": (h, 3*h), "attn.Wo": (h, h),
                    "mlp.Wi": (h, 2*f), "mlp.Wo": (f, h)}
        else:
            f = self.decision_intermediate_size
            dims = {"self_attn.in_proj_weight": (h, 3*h), "self_attn.out_proj": (h, h),
                    "linear1": (h, f), "linear2": (f, h)}
        return tuple(ProjectionSpec(name, i, o, min(self.meta_rank, i, o)) for name,(i,o) in dims.items())

    @classmethod
    def cpu_fixture(cls, memory_tokens=32, encoder_layers=22):
        return cls(hidden_size=8, encoder_layers=encoder_layers, decision_layers=2,
                   encoder_intermediate_size=12, decision_intermediate_size=32,
                   memory_tokens=memory_tokens, generated_rank=2, meta_rank=4,
                   heads=2, feedforward_size=16, fixture=True)

def _key(layer, path, factor):
    return f"layer{layer}_{path.replace('.', '_')}_{factor}"

class RankShineHypernetwork(nn.Module):
    def __init__(self, spec=None, generated_scale=0.001):
        super().__init__()
        self.spec = spec or ArchitectureSpec()
        if not self.spec.fixture:
            from dataclasses import replace
            if self.spec.generated_rank not in (4,8,16,32) or self.spec.memory_tokens != 512:
                raise ValueError("Rank sweep requires r4/8/16/32 and shared512 memory tokens")
            if self.spec != replace(ArchitectureSpec(), generated_rank=self.spec.generated_rank, memory_tokens=512):
                raise ValueError("Only output rank and shared memory token count may change")
        if generated_scale != 0.001:
            raise ValueError("SHINE generated factor scale is fixed at0.001")
        self.generated_scale = generated_scale
        self.initialization_seed = int(torch.initial_seed())
        self.memory_init_std = 0.02
        self.hidden_size = self.spec.hidden_size
        self.num_memory_tokens = self.spec.memory_tokens
        self.generated_rank = self.spec.generated_rank
        self.meta_rank = self.spec.meta_rank
        self.target_specs = {i:self.spec.targets(i) for i in range(self.spec.layers)}
        self.parameters_per_layer = {i:sum(self.generated_rank*(t.in_features+t.out_features) for t in row)
                                     for i,row in self.target_specs.items()}
        if max(self.parameters_per_layer.values()) > self.num_memory_tokens*self.hidden_size:
            raise ValueError("Memory tensor cannot serialize all generated LoRA factors")
        self.meta = nn.ParameterDict()
        for i,row in self.target_specs.items():
            for target in row:
                a = nn.Parameter(torch.empty(target.in_features,target.meta_rank))
                nn.init.normal_(a,std=math.sqrt(generated_scale))
                self.meta[_key(i,target.path,"A")] = a
                self.meta[_key(i,target.path,"B")] = nn.Parameter(torch.zeros(target.meta_rank,target.out_features))
        self.memory_embeddings = nn.Parameter(torch.empty(self.num_memory_tokens,self.hidden_size))
        nn.init.normal_(self.memory_embeddings,std=self.memory_init_std)
        self.layer_pe = nn.Parameter(torch.zeros(self.spec.layers,self.hidden_size))
        self.token_pe = nn.Parameter(torch.zeros(self.num_memory_tokens,self.hidden_size))
        self.transformer_layers = nn.ModuleList(nn.TransformerEncoderLayer(
            self.hidden_size,self.spec.heads,self.spec.feedforward_size,dropout=0.0,
            activation="gelu",layer_norm_eps=1e-5,batch_first=True,norm_first=False,bias=True) for _ in range(4))
        if not self.spec.fixture:
            inventory = self.parameter_inventory()
            if inventory["total_trainable"] != EXPECTED_TRAINABLE + 2*(512-256)*self.spec.hidden_size or inventory["generated_per_document"] != EXPECTED_GENERATED*self.generated_rank//8:
                raise RuntimeError(f"Production parameter geometry differs: {inventory}")

    def meta_adapters(self):
        return {i:{t.path:(self.meta[_key(i,t.path,"A")],self.meta[_key(i,t.path,"B")]) for t in row}
                for i,row in self.target_specs.items()}

    def append_memory(self, context_embeddings, attention_mask):
        if context_embeddings.ndim != 3 or context_embeddings.shape[-1] != self.hidden_size:
            raise ValueError("Context embeddings must be[B,T,H]")
        if tuple(attention_mask.shape) != tuple(context_embeddings.shape[:2]):
            raise ValueError("Context mask shape differs")
        m = self.memory_embeddings.to(context_embeddings.dtype)[None].expand(context_embeddings.shape[0],-1,-1)
        real = attention_mask.new_ones(context_embeddings.shape[0],self.num_memory_tokens)
        return torch.cat((context_embeddings,m),1),torch.cat((attention_mask,real),1)

    def transform_memory(self, memory_states):
        expected = (self.spec.layers,self.num_memory_tokens,self.hidden_size)
        if memory_states.ndim != 4 or tuple(memory_states.shape[1:]) != expected:
            raise ValueError(f"Memory states must be[B,{expected}]")
        b,l,m,h = memory_states.shape
        x = memory_states.to(self.layer_pe.dtype) + self.layer_pe[None,:,None,:] + self.token_pe[None,None,:,:]
        for i,block in enumerate(self.transformer_layers):
            if i%2 == 0:
                x = block(x.transpose(1,2).reshape(b*m,l,h)).reshape(b,m,l,h).transpose(1,2)
            else:
                x = block(x.reshape(b*l,m,h)).reshape(b,l,m,h)
        return x

    def factors_from_memory(self, transformed):
        if tuple(transformed.shape[1:]) != (self.spec.layers,self.num_memory_tokens,self.hidden_size):
            raise ValueError("Transformed memory geometry differs")
        b = transformed.shape[0];flat = transformed.reshape(b,self.spec.layers,-1)
        scale = math.sqrt(self.generated_scale);adapters = {}
        for i,row in self.target_specs.items():
            needed = self.parameters_per_layer[i]
            values = flat[:,i,:needed]
            if values.shape[-1] != needed:
                raise ValueError("Insufficient memory serialization capacity; zero padding is forbidden")
            offset = 0;targets = {}
            for target in row:
                na = target.in_features*self.generated_rank;nb = target.out_features*self.generated_rank
                a = values[:,offset:offset+na].reshape(b,target.in_features,self.generated_rank);offset += na
                factor_b = values[:,offset:offset+nb].reshape(b,self.generated_rank,target.out_features);offset += nb
                targets[target.path] = (a*scale,factor_b*scale)
            if offset != needed:
                raise RuntimeError("Adapter factor layout mismatch")
            adapters[i] = targets
        return adapters

    def forward(self,memory_states):
        return self.factors_from_memory(self.transform_memory(memory_states))

    @staticmethod
    def adapter_for_document(adapters,index=0):
        return {i:{name:(a[index],b[index]) for name,(a,b) in targets.items()} for i,targets in adapters.items()}

    def parameter_inventory(self):
        encoder_meta = sum(p.numel() for name,p in self.meta.items() if int(name.split('_',1)[0][5:]) < self.spec.encoder_layers)
        meta_total = sum(p.numel() for p in self.meta.parameters())
        return {"encoder_meta_lora":encoder_meta,"decision_meta_lora":meta_total-encoder_meta,
                "meta_lora":meta_total,"m2p_blocks":sum(p.numel() for p in self.transformer_layers.parameters()),
                "memory_embeddings":self.memory_embeddings.numel(),"token_pe":self.token_pe.numel(),"layer_pe":self.layer_pe.numel(),
                "total_trainable":sum(p.numel() for p in self.parameters() if p.requires_grad),
                "generated_per_document":sum(self.parameters_per_layer.values()),"projection_targets":sum(map(len,self.target_specs.values()))}

    def layout_inventory(self):
        return {str(i):{"kind":"encoder" if i<self.spec.encoder_layers else "decision_transformer",
                        "generated_numel":self.parameters_per_layer[i],
                        "targets":[{"path":t.path,"in":t.in_features,"out":t.out_features,"meta_rank":t.meta_rank,"generated_rank":self.generated_rank} for t in row]}
                for i,row in self.target_specs.items()}

    def initialization_protocol(self):
        return {"torch_initial_seed":self.initialization_seed,"memory_std":0.02,"meta_A_std":math.sqrt(self.generated_scale),
                "meta_B":"zeros","token_pe":"zeros","layer_pe":"zeros","generated_factor_multiplier":math.sqrt(self.generated_scale)}
