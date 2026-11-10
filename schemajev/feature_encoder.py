"""Frozen bare Laya description encoding. No learned memory/meta adapters."""
import hashlib,json

def need(v,m):
 if not v:raise ValueError(m)

def tokens(model,group):
 text=model.conditioning_text(group)
 need(model.tok.mask_token not in text,'native mask sanitizer token forbidden')
 ids=model.common.encode_text(model.tok,text,add_special_tokens=False)['input_ids']
 ids=[model.tok.cls_token_id]+list(ids)+[model.tok.sep_token_id]
 # Keep the old 512-token reservation as an input eligibility restriction;
 # no memory token is appended or learned in this model.
 need(len(ids)+512<=min(model.max_length,model.encoder.config.max_position_embeddings),'same native description length gate without truncation')
 return ids

def encode(torch,model,ids):
 need(model.controller.active is None and not model.backbone.training,'idle unadapted frozen backbone')
 device=next(model.hypernet.parameters()).device
 x=torch.tensor([ids],device=device,dtype=torch.long);mask=torch.ones_like(x)
 with torch.no_grad(),model._autocast():
  hidden,mem=model._encoder_hidden(model.encoder.embeddings.tok_embeddings(x),mask,None,capture_memory=False)
  need(not mem and hidden.ndim==3 and hidden.shape[0]==1 and hidden.shape[1]==len(ids) and hidden.shape[2]==768,'bare encoder final hidden geometry')
  pooled=(hidden.float()*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True)
 need(tuple(pooled.shape)==(1,768) and bool(torch.isfinite(pooled).all()) and model.controller.active is None,'finite detached pure frozen pooled feature')
 return pooled.detach()

def token_proof(model,conditions):
 need(isinstance(conditions,dict) and len(conditions)==60,'60 exact TRAIN conditions')
 seen={};lengths=[]
 for key,condition in sorted(conditions.items()):
  ids=tokens(model,condition);h=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest();need(h not in seen,'distinct actual TRAIN token sequences');seen[h]=key;lengths.append(len(ids))
 return {'complete':True,'conditions':60,'unique_token_sequences':60,'truncated':0,'max_tokens':max(lengths),'min_tokens':min(lengths),'memory_tokens_appended':0,'adapters_used_for_feature':False,'token_body_exported':False}
