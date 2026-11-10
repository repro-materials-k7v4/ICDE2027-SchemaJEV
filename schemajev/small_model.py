"""929984-parameter generator + original PCA decoder and native typed-query ABI."""
import contextlib,json,time,types
from . import feature_encoder

def need(v,m):
 if not v:raise ValueError(m)

def hyper_class(torch,released,rank,native):
 _,BaseDecoder=released.classes(torch,rank,native)
 class Decoder(BaseDecoder):
  def forward(self,z):
   first=next(iter(z.values()))
   with (contextlib.nullcontext() if first.device.type=='meta' else torch.autocast(first.device.type,enabled=False)):return super().forward({k:v.float() for k,v in z.items()})
 class Hyper(torch.nn.Module):
  def __init__(self,geometry,manifest,arrays):
   super().__init__();self.spec=geometry;self.target_specs={i:geometry.targets(i) for i in range(geometry.layers)}
   self.decoder=Decoder(geometry,manifest,arrays)
   self.input_norm=torch.nn.LayerNorm(768);self.fc1=torch.nn.Linear(768,544);self.activation=torch.nn.GELU();self.fc2=torch.nn.Linear(544,256);self.hidden_norm=torch.nn.LayerNorm(256)
   self.coordinate_heads=torch.nn.ModuleDict({f'{i:03d}':torch.nn.Linear(256,60) for i in range(24)})
   for head in self.coordinate_heads.values():torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)
   need(sum(p.numel() for p in self.parameters() if p.requires_grad)==929984,'exact complete small trainable parameter count')
  def initialization_protocol(self):return {'prior':'original fixed teacher factor mixture','coordinate_heads':'zero weights and biases','description_encoder':'frozen bare Laya','trainable_parameters':929984}
  def forward_coordinates(self,feature):
   need(feature.ndim==2 and feature.shape[1]==768,'exact pooled description features')
   with (contextlib.nullcontext() if feature.device.type=='meta' else torch.autocast(feature.device.type,enabled=False)):
    h=self.hidden_norm(self.fc2(self.activation(self.fc1(self.input_norm(feature.float())))))
    return {key:self.coordinate_heads[key](h) for key in self.decoder.layer_keys}
  @staticmethod
  def adapter_for_document(adapters,index=0):return {i:{name:(a[index],b[index]) for name,(a,b) in targets.items()} for i,targets in adapters.items()}
 return Hyper

def make_model_class(torch,original_models):
 class SmallNative(original_models.FullT2LLaya):
  def train(self,mode=True):
   super().train(mode);self.backbone.eval();self.hypernet.train(mode);return self
  def parameter_inventory(self):
   named={n:p for n,p in self.named_parameters() if p.requires_grad};hyper={n:p for n,p in self.hypernet.named_parameters() if p.requires_grad}
   expected={'input_norm.weight':(768,),'input_norm.bias':(768,),'fc1.weight':(544,768),'fc1.bias':(544,),'fc2.weight':(256,544),'fc2.bias':(256,),'hidden_norm.weight':(256,),'hidden_norm.bias':(256,)}
   for i in range(24):expected.update({f'coordinate_heads.{i:03d}.weight':(60,256),f'coordinate_heads.{i:03d}.bias':(60,)})
   need(set(hyper)==set(expected) and all(tuple(hyper[n].shape)==v for n,v in expected.items()) and len(hyper)==56 and sum(p.numel() for p in hyper.values())==929984,'exact new 929984 params and 56 shapes')
   need(set(named)=={'hypernet.'+n for n in hyper} and all(p.dtype==torch.float32 for p in hyper.values()),'only FP32 small generator trains')
   need(not any(p.requires_grad for p in self.backbone.parameters()) and not any(p.requires_grad for p in self.hypernet.decoder.parameters()),'released backbone and original PCA buffers frozen')
   return {'total_trainable':929984,'frozen_base':True,'frozen_decoder':True,'parameters':{n:p.numel() for n,p in hyper.items()},'memory_parameters':0,'meta_adapter_parameters':0}
  def _autocast(self):
   device=next(self.hypernet.parameters()).device
   return torch.autocast(device.type,dtype=self.amp_dtype,enabled=self.amp_enabled)
  def _context_encoding(self,group):return feature_encoder.tokens(self,group)
  @staticmethod
  def _condition_key(group):return json.dumps(group,sort_keys=True,separators=(',',':'),ensure_ascii=False)
  def install_train_condition_cache(self,conditions,lease):
   need(not hasattr(self,'_feature_cache') and len(conditions)==60,'one own process exact 60 train conditions')
   start=time.monotonic();cache={}
   for dataset_id,condition in sorted(conditions.items()):
    lease.boundary();key=self._condition_key(condition);need(key not in cache,'unique canonical conditions')
    ids=self._context_encoding(condition);feature=feature_encoder.encode(torch,self,ids);cache[key]=(feature,len(ids))
   need(len(cache)==60,'complete pure encoder cache');self._feature_cache=cache
   return {'complete':True,'conditions':60,'description_encoder_forward_count':60,'no_meta_adapter':True,'no_memory_appended':True,'frozen_encoder':True,'cache_scope':'process_only','feature_body_exported':False,'elapsed_seconds':time.monotonic()-start}
  def generate_adapters(self,group):
   key=self._condition_key(group);need(hasattr(self,'_feature_cache') and key in self._feature_cache,'qualified same process train-condition feature required');feature,prefix=self._feature_cache[key];z=self.hypernet.forward_coordinates(feature)
   return self.hypernet.adapter_for_document(self.hypernet.decoder(z),0),prefix
  def question_logits(self,group,adapters=None):
   batch,items,proofs=self._question_encoding(group);device=next(self.hypernet.parameters()).device;ids=batch['input_ids'].to(device);real=batch['attention_mask'].to(device)
   with self._autocast():
    hidden,_=self._encoder_hidden(self.encoder.embeddings.tok_embeddings(ids),real,adapters);hidden,_=self._decision_hidden(hidden,real,adapters)
    markers=batch['marker_pos'].to(device);idx=markers[:,:,None].expand(-1,-1,hidden.shape[-1]);logits=self.backbone.scorer(torch.gather(hidden,1,idx)).squeeze(-1).float()
   return [logits[i,:len(item['markers'])] for i,item in enumerate(items)],items,proofs
 return SmallNative

def make_query_class(torch,query_source):
 class SmallCachedQuery(query_source.CachedTypedStudentQuery):
  def __init__(self,native,adapter_module,contracts,prepared):
   super().__init__(native,adapter_module,contracts,prepared)
   def logits(view,prepared,adapters):
    model=view.native_model;batch=model.common.collate_items([prepared['items']],model.tok.pad_token_id);device=next(model.hypernet.parameters()).device
    ids=batch['input_ids'].to(device);real=batch['attention_mask'].to(device);qtype=batch['qtype'].to(device)
    with model._autocast():
     hidden,_=model._encoder_hidden(model.encoder.embeddings.tok_embeddings(ids),real,adapters);hidden=hidden+model.backbone.type_emb(qtype)[:,None,:]
     for layer in model.backbone.head.layers:
      def run(x,layer=layer,adapters=adapters):
       with model.controller.use(adapters):return layer(x,src_key_padding_mask=~real.bool())
      hidden=model._step(run,hidden)
     markers=batch['marker_pos'].to(device);gather=markers[:,:,None].expand(-1,-1,hidden.shape[-1]);out=model.backbone.scorer(torch.gather(hidden,1,gather)).squeeze(-1).float()
    return [out[i,:len(item['markers'])] for i,item in enumerate(prepared['items'])]
   self.view._laya_logits=types.MethodType(logits,self.view)
  def query_logits(self,contracts,*,factor_adapters,prefix_tokens):
   result=super().query_logits(contracts,factor_adapters=factor_adapters,prefix_tokens=prefix_tokens)
   result['condition_path']='frozen bare Laya description encoder -> pooled small generator -> PCA adapters only';return result
 return SmallCachedQuery
