"""J4 dualbase selected100 PCA core, parameterized predeclared lambda only.
Original ced/808 decoder/head/condition VJP equations remain unchanged.
"""
"""Genuine100 joint candidate teachers; K grid is a fixed nested raw-PCA prefix.
All native decoder/HNet/math/initialization equations remain parent3108.
"""
"""Fresh native SHINE meta/memory/4blocks; rank64 Direct or predeclared K16 factor PCA.

No file I/O or model/basis loading on import. Direct uses two width1024 output
streams, necessary because rank64 cannot fit the old single512x768 serializer.
This is an explicit architecture extension, not a claim of equal r32 capacity.
"""
import math
from dataclasses import asdict


def need(v,message):
    if not v:raise ValueError(message)


K_GRID=(8,16,32,64,96)
HEAD_RECIPE='old_K16_prefix_preserving_seed4242_tail96_layerseed_v1'
def initialize_nested_heads(torch,heads):
    """Old K16 stream remains exact; per-layer tail uses isolated fork RNG.
    All layers consume exactly16 base rows irrespective of requestedK. Tail
    generation cannot move the old seed4242 stream or the caller's RNG state.
    """
    keys=sorted(heads);need(keys and all(1<=heads[k].out_features<=96 and heads[k].in_features==256 for k in keys),'Only fixed256 trunk and at most96 nested head rows')
    devices=sorted({h.weight.device.index or 0 for h in heads.values()if h.weight.device.type=='cuda'})
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(4242)
        for i,key in enumerate(keys):
            h=heads[key];base=torch.empty((16,256),device=h.weight.device,dtype=h.weight.dtype);torch.nn.init.normal_(base,std=1/math.sqrt(256))
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(5242+i);tail=torch.empty((80,256),device=h.weight.device,dtype=h.weight.dtype);torch.nn.init.normal_(tail,std=1/math.sqrt(256))
            with torch.no_grad():h.weight.copy_(torch.cat((base,tail),0)[:h.out_features]);h.bias.zero_()
    return {'head_recipe':HEAD_RECIPE,'old_K16_weights_prefix_exact':True,'base_head_seed':4242,'base_rows_per_layer_rng_advance':16,'tail_rows_per_layer':80,'tail_layer_seed':'5242+layer_index','tail_fork_does_not_advance_base_or_global_rng':True}

def classes(torch,rank):
    nn=torch.nn
    class Common(rank.RankShineHypernetwork):
        def __init__(self,g):
            nn.Module.__init__(self);self.spec=g
            if not g.fixture:
                need(g.base_id=='original_multilingual'and g.generated_rank==64 and g.memory_tokens==512 and
                     (g.hidden_size,g.encoder_layers,g.decision_layers,g.encoder_intermediate_size,g.decision_intermediate_size,g.meta_rank,g.heads,g.feedforward_size)==(768,22,2,1152,3072,128,12,1536),'Exact fresh rank64 original native common geometry')
            self.generated_scale=.001;self.initialization_seed=int(torch.initial_seed());self.memory_init_std=.02
            self.hidden_size=g.hidden_size;self.num_memory_tokens=g.memory_tokens;self.generated_rank=g.generated_rank;self.meta_rank=g.meta_rank
            self.target_specs={i:g.targets(i)for i in range(g.layers)}
            self.parameters_per_layer={i:sum(g.generated_rank*(t.in_features+t.out_features)for t in ts)for i,ts in self.target_specs.items()}
            self.meta=nn.ParameterDict()
            for i,targets in self.target_specs.items():
                for t in targets:
                    a=nn.Parameter(torch.empty(t.in_features,t.meta_rank));nn.init.normal_(a,std=math.sqrt(.001));self.meta[rank._key(i,t.path,'A')]=a;self.meta[rank._key(i,t.path,'B')]=nn.Parameter(torch.zeros(t.meta_rank,t.out_features))
            self.memory_embeddings=nn.Parameter(torch.empty(g.memory_tokens,g.hidden_size));nn.init.normal_(self.memory_embeddings,std=.02)
            self.layer_pe=nn.Parameter(torch.zeros(g.layers,g.hidden_size));self.token_pe=nn.Parameter(torch.zeros(g.memory_tokens,g.hidden_size))
            self.transformer_layers=nn.ModuleList(nn.TransformerEncoderLayer(g.hidden_size,g.heads,g.feedforward_size,dropout=0.,activation='gelu',layer_norm_eps=1e-5,batch_first=True,norm_first=False,bias=True)for _ in range(4))

    class Direct(Common):
        def __init__(self,g,output_width=1024):
            super().__init__(g);self.output_width=output_width
            need(all(sum(t.in_features*g.generated_rank for t in ts)<=g.memory_tokens*output_width and
                     sum(t.out_features*g.generated_rank for t in ts)<=g.memory_tokens*output_width for ts in self.target_specs.values()),'Both complete A/B streams fit without zero padding')
            self.factor_A_head=nn.Linear(g.hidden_size,output_width);self.factor_B_head=nn.Linear(g.hidden_size,output_width)
            nn.init.normal_(self.factor_A_head.weight,std=1/math.sqrt(g.hidden_size));nn.init.zeros_(self.factor_A_head.bias);nn.init.zeros_(self.factor_B_head.weight);nn.init.zeros_(self.factor_B_head.bias)
        def forward(self,memory_states):
            transformed=self.transform_memory(memory_states);batch=transformed.shape[0]
            av=self.factor_A_head(transformed).flatten(2);bv=self.factor_B_head(transformed).flatten(2);scale=math.sqrt(.001);out={}
            for i,targets in self.target_specs.items():
                ao=bo=0;result={}
                for t in targets:
                    na=t.in_features*self.generated_rank;nb=t.out_features*self.generated_rank
                    a=av[:,i,ao:ao+na].reshape(batch,t.in_features,self.generated_rank);b=bv[:,i,bo:bo+nb].reshape(batch,self.generated_rank,t.out_features)
                    result[t.path]=(a*scale,b*scale);ao+=na;bo+=nb
                out[i]=result
            return out
        def initialization_protocol(self):
            return {'common_seed':42,'fresh_random_HNet':True,'Direct_A_head':'random normal1/sqrt(hidden), bias0','Direct_B_head':'weight0,bias0','both_factors_zero':False,'output_stream_width':self.output_width,'generated_factor_multiplier':math.sqrt(.001),'old_student_or_teacher_CP_loaded':False,'shared_LoRA_or_A_or_anchor':False}

    class PCADecoder(nn.Module):
        def __init__(self,g,manifest,arrays,k=16):
            super().__init__();need(k in K_GRID and manifest['K']==k and manifest['schema']=='dualbase-selected100-r64-zero-origin-PCA-bank-v1'and manifest['teacher_count']==100 and manifest['generated_rank']==64 and manifest['zero_origin']is True and manifest['centered']is False and manifest['runtime_mean_added']is False and manifest['runtime_anchor']is False and manifest['SharedA']is False,'Only actual selected100 joint-candidate uncentered zero-origin factor bank')
            self.k=k;self.teacher_count=manifest['teacher_count'];self.layer_keys=[f'{i:03d}'for i in range(g.layers)];need(sorted(manifest['layers'])==self.layer_keys,'All native layer directions')
            self.dimensions={key:k for key in self.layer_keys};self.dimension=len(self.layer_keys)*k;self.layout={}
            for i,key in enumerate(self.layer_keys):
                layer=manifest['layers'][key];d=arrays[key]['directions'];rms=arrays[key]['component_RMS'];need(tuple(d.shape)==(k,layer['vector_numel'])and tuple(rms.shape)==(k,)and bool(torch.isfinite(rms).all())and bool((rms>0).all())and layer['dimensions']==k and layer['actual_factor_numerical_rank']>=k,'Actual unit nestedK directions and positive teacher-only component RMS, no padding')
                need(len(layer['layout'])==4,'All original four targets per layer');covered=[]
                expected={t.path:t for t in g.targets(i)}
                for row in layer['layout']:
                    t=expected[row['target']];need(row['A_shape']==[t.in_features,g.generated_rank]and row['B_shape']==[g.generated_rank,t.out_features],'Exact original rank64 target factor shapes')
                    for side in('A','B'):
                        start=row[side+'_offset'];width=math.prod(row[side+'_shape']);need(width==row[side+'_numel']and 0<=start<start+width<=layer['vector_numel'],'Valid explicit packed factor interval');covered.append((start,start+width))
                cursor=0
                for start,end in sorted(covered):need(start==cursor,'Disjoint gapless factor intervals');cursor=end
                need(cursor==layer['vector_numel']and {r['target']for r in layer['layout']}==set(expected),'No runtime mean/anchor coordinate or missing target/factor slice')
                self.layout[key]=layer['layout'];self.register_buffer('layer_'+key+'_directions',d.detach().float().clone());self.register_buffer('layer_'+key+'_component_RMS',rms.detach().float().clone())
        def forward(self,coordinates):
            out={}
            for key in self.layer_keys:
                z=coordinates[key];need(z.ndim==2 and z.shape[1]==self.k,'Exact fresh layer coordinates')
                with torch.autocast(z.device.type,enabled=False):values=(z.float()*getattr(self,'layer_'+key+'_component_RMS'))@getattr(self,'layer_'+key+'_directions')
                for row in self.layout[key]:
                    pair=[]
                    for side in('A','B'):
                        start=row[side+'_offset'];shape=row[side+'_shape'];pair.append(values[:,start:start+math.prod(shape)].reshape(z.shape[0],*shape))
                    out.setdefault(int(key),{})[row['target']]=tuple(pair)
            return out

    class PCA(Common):
        def __init__(self,g,decoder):
            super().__init__(g);self.decoder=decoder
            self.coordinate_trunk=nn.Sequential(nn.LayerNorm(g.hidden_size),nn.Linear(g.hidden_size,256),nn.GELU(),nn.LayerNorm(256))
            self.coordinate_heads=nn.ModuleDict({key:nn.Linear(256,decoder.k)for key in decoder.layer_keys})
            initialize_nested_heads(torch,self.coordinate_heads)
            self.last_coordinates={}
        def standardized_coordinates(self,memory_states):
            # Same unbounded standard-coordinate head, without RMS or factor decoding.
            transformed=self.transform_memory(memory_states);hidden=self.coordinate_trunk(transformed.mean(2));return {key:self.coordinate_heads[key](hidden[:,int(key),:])for key in self.decoder.layer_keys}
        def forward(self,memory_states):
            z=self.standardized_coordinates(memory_states);self.last_coordinates={key:v.detach()for key,v in z.items()};return self.decoder(z)
        def initialization_protocol(self):
            return {'common_seed':42,'head_seed':4242,'fresh_random_HNet':True,'PCA_head_weight':'nonzero nested normal1/sqrt256; original K16 exact prefix; final hidden LayerNorm','head_recipe':HEAD_RECIPE,'PCA_head_bias':0.,'zero_origin_decoder':True,'coordinate_units':'teacher-only uncentered component RMS sqrt(eigenvalue/actual selected teacher_count100)','coordinate_hard_cap':False,'runtime_teacher_mean':False,'SharedLoRA_or_SharedA_or_trained_anchor':False,'old_student_HNet_CP_loaded':False}
    return Direct,PCA,PCADecoder


def construct_student(torch,*,tokenizer,backbone,common,geometry,rank_module,original_models,
                      method,basis_manifest=None,basis_arrays=None,pca_k=16,device='cpu',gradient_checkpointing=False):
    need(method in('direct_t2l','pca_t2l'),'Explicit random description HNet method')
    D,P,Decoder=classes(torch,rank_module);device=torch.device(device);fork=[device.index or 0]if device.type=='cuda'else[]
    with torch.random.fork_rng(devices=fork):
        torch.manual_seed(42)
        if method=='direct_t2l':hyper=D(geometry)
        else:
            decoder=Decoder(geometry,basis_manifest,basis_arrays,k=pca_k);hyper=P(geometry,decoder)
            # Common construction had seed42, output head recipe is separately frozen.
            with torch.random.fork_rng(devices=fork):
                torch.manual_seed(4242)
                initialize_nested_heads(torch,hyper.coordinate_heads)
        hyper=hyper.to(device=device,dtype=torch.float32)
    class FreshDescriptionNativeStudent(original_models.FullT2LLaya):
        def parameter_inventory(self):
            named={n:p.numel()for n,p in self.named_parameters()if p.requires_grad};need(named and all(n.startswith('hypernet.')for n in named),'Only genuinely fresh HNet/meta/memory/heads train')
            return {'total_trainable':sum(named.values()),'parameters':named,'base_geometry':asdict(geometry),'actual_description_conditioning':True,'SharedLoRA_or_anchor_or_runtime_teacher_mean':False}
    model=FreshDescriptionNativeStudent.from_components(tokenizer,backbone,hyper,common,gradient_checkpointing=gradient_checkpointing,max_length=8192,head_max_length=512,amp_dtype=torch.bfloat16,amp_enabled=device.type=='cuda')
    model.mode='halfdata_fresh_'+method;model.architecture_config.update(fresh_random_HNet=True,new_native_base_frozen=True,generated_rank=64,memory_tokens=512,coordinate_dimension_per_layer=pca_k if method=='pca_t2l'else None,PCA_coordinate_units='teacher_component_RMS_unbounded'if method=='pca_t2l'else None,direct_output_stream_width=1024 if method=='direct_t2l'else None,shared_LoRA_branch=False,SharedA=False,trained_adapter_or_HNet_initialization=False,runtime_teacher_mean=False,runtime_anchor=False,not_equal_to_previous_r32_serializer_capacity=True)
    return model


DIRECTION_BALANCE_CONFIG={
 'schema':'dualbase-standardized-PCA-direction-usage-balance-v1',
 'coefficient':1.0,'epsilon':1e-8,
 'condition_scope':'all60_H2_TRAIN_task_descriptions_equal_dataset_vote',
 'condition_count':60,'layer_count':24,
 'energy':'mean_d float32(z_std[d,l,k]^2)',
 'probability':'(e_k+epsilon)/(sum_k e_k+K*epsilon)',
 'loss':'mean_24layers(K*sum_k p_k^2-1)',
 'RMS_weighted':False,'once_per_optimizer_update':True,
 'query_or_gold_used':False,'condition_hidden_cache':False,
 'coordinate_decoder_used_for_balance':False,
 'DDP_normalization':'rank_local_description_VJP_times_world_over_global_condition_count_then_original_gradient_average',
 'zero_energy_policy':'no_extra_energy_objective; explicit_collapse_flag_and_effective_dims_None_at_or_below_Keps'
}

def direction_balance_config(coefficient):
    need(type(coefficient)is float and coefficient in(0.,.1,1.,3.),'Only predeclared lambda grid')
    return {**DIRECTION_BALANCE_CONFIG,'coefficient':coefficient}

def validate_direction_balance_config(config):
    need(config==direction_balance_config(config.get('coefficient')),'Exact parameterized lambda/eps/all60 equal standard-coordinate objective');return True

def direction_balance_objective(torch,energy,epsilon=1e-8):
    """Energy is [layers,K], computed in FP32; PCA RMS never enters this loss."""
    need(energy.ndim==2 and energy.dtype==torch.float32 and energy.shape[0]>0 and energy.shape[1]>1 and bool(torch.isfinite(energy).all())and bool((energy>=0).all()),'Finite nonnegative FP32 standard-coordinate energy')
    need(epsilon==1e-8,'One predeclared energy epsilon')
    k=energy.shape[1];total=energy.sum(1);p=(energy+epsilon)/(total[:,None]+k*epsilon);perlayer=k*p.square().sum(1)-1.;loss=perlayer.mean()
    with torch.no_grad():
        good=total>k*epsilon;eff=1./p.square().sum(1);share=p.max(1).values;layers=[]
        for i in range(energy.shape[0]):
            layers.append({'layer':i,'total_energy':float(total[i]),'effective_dims':float(eff[i])if bool(good[i])else None,'max_share':float(share[i]),'balance_loss':float(perlayer[i]),'energy_at_or_below_Keps':not bool(good[i])})
        diagnostics={'direction_balance_loss':float(loss.detach()),'direction_total_energy_mean':float(total.mean()),'direction_total_energy_min':float(total.min()),'direction_effective_dims_mean':float(eff[good].mean())if bool(good.any())else None,'direction_max_share_mean':float(share.mean()),'direction_max_share_max':float(share.max()),'direction_energy_collapsed_layers':int((~good).sum()),'direction_usage_effective_dims_valid_layers':int(good.sum()),'direction_usage_per_layer':layers}
    return loss,diagnostics

def backward_direction_usage_balance(torch,coordinates_for_condition,condition_ids,*,rank=0,world_size=1,reduce_energy_sum=None,coefficient=1.0,epsilon=1e-8,expected_layers=24):
    """One global-description objective and exact streamed coordinate VJP.

    Both passes use grad-enabled, identical original condition forwards. The
    first graph is immediately discarded per description; only tiny detached
    coordinates are retained. No factor decoder, query, gold or teacher is used.
    stop-gradient on dL/de is the chain rule, never on trainable condition paths.
    """
    ids=list(condition_ids);need(ids==sorted(ids)and len(ids)==len(set(ids))and ids,'Canonical unique condition ID axis')
    need(world_size in(1,2)and 0<=rank<world_size and len(ids)%world_size==0,'Complete equal deterministic condition shards')
    need(type(coefficient)is float and coefficient in(0.,.1,1.,3.),'Only predeclared lambda0/.1/1/3; original chain scaling')
    if coefficient==0.:
        return {'direction_balance_coefficient':0.,'direction_balance_loss':0.,'direction_objective_evaluations':0,'direction_condition_forward_calls':0,'lambda_zero_original_CE_path_untouched':True}
    local_ids=ids[rank::world_size];energy_sum=None;first_values={};keys=None
    for identity in local_ids:
        with torch.enable_grad():
            coords=coordinates_for_condition(identity);current=sorted(coords)
            if keys is None:keys=current
            need(current==keys and len(keys)==expected_layers,'All exact native coordinate layers')
            value=torch.stack([coords[key].float().squeeze(0)for key in keys]);need(value.ndim==2 and value.shape[0]==expected_layers and value.shape[1]in K_GRID,'Only standard [layer,K] coordinates, never decoded factors')
            saved=value.detach().clone();first_values[identity]=saved
            if energy_sum is None:energy_sum=torch.zeros_like(saved,dtype=torch.float32)
            energy_sum.add_(saved.square());del value,coords,saved
    if world_size>1:
        need(callable(reduce_energy_sum),'Original distributed energy SUM required')
        reduce_energy_sum(energy_sum)
    energy=(energy_sum/len(ids)).detach().requires_grad_(True)
    loss,diagnostics=direction_balance_objective(torch,energy,epsilon);denergy,=torch.autograd.grad(loss,energy);denergy=denergy.detach()
    for identity in local_ids:
        with torch.enable_grad():
            coords=coordinates_for_condition(identity);need(sorted(coords)==keys,'Same condition coordinate axis on VJP replay')
            value=torch.stack([coords[key].float().squeeze(0)for key in keys]);need(torch.equal(value.detach(),first_values[identity]),'Deterministic grad-enabled condition coordinate replay; no stale hidden cache or stochastic mask')
            upstream=value.detach()*(2.*world_size*coefficient/len(ids))*denergy
            torch.autograd.backward(value,upstream)
            del value,coords,upstream
        del first_values[identity]
    diagnostics.update(direction_balance_coefficient=coefficient,direction_objective_evaluations=1,direction_condition_count_global=len(ids),direction_condition_count_local=len(local_ids),direction_condition_forward_calls=2*len(local_ids),direction_energy_collectives=int(world_size>1),direction_VJP_uses_original_trainable_meta_memory=True,direction_factor_decodes=0,direction_RMS_weights_used=False)
    return diagnostics
