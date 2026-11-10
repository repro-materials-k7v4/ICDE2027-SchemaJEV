"""Pure deterministic three-type FIT course, 200x80, no label use."""
import hashlib,json,random
TYPES=('choice','binary','score')
def need(x,m):
 if not x:raise ValueError(m)
def course(pools,types,seed=514,steps=200,batch=80,expected_tasks=60):
 need(type(seed) is int and seed==514 and type(steps) is int and steps==200 and type(batch) is int and batch==80,'exact fixed course')
 need(set(pools)==set(types) and len(pools)==expected_tasks and set(types.values())==set(TYPES),'exact typed roster')
 need(all(sum(t==k for t in types.values())==expected_tasks//3 for k in TYPES),'equal task type roster')
 streams={};pos={}
 for d in sorted(pools):
  xs=list(pools[d]);need(xs and all(type(i) is int and i>=0 for i in xs) and len(set(xs))==len(xs),'unique FIT identities');random.Random(int.from_bytes(hashlib.sha256((str(seed)+'|'+d).encode()).digest(),'big')).shuffle(xs);streams[d]=xs;pos[d]=0
 groups={k:sorted(d for d in pools if types[d]==k) for k in TYPES};shufflers={k:random.Random(int.from_bytes(hashlib.sha256((str(seed)+'|'+k).encode()).digest(),'big')) for k in TYPES};pending={k:[] for k in TYPES};result=[]
 for step in range(steps):
  batchrows=[];remaining={k:batch//3+(1 if (i-step)%3<batch%3 else 0) for i,k in enumerate(TYPES)};turn=step%3
  while sum(remaining.values()):
   kind=TYPES[turn%3];turn+=1
   if remaining[kind]==0:continue
   active=[d for d in groups[kind] if pos[d]<len(streams[d])];need(active,'insufficient FIT rows for exact type course')
   if not pending[kind]:pending[kind]=active[:];shufflers[kind].shuffle(pending[kind])
   while pending[kind] and pos[pending[kind][-1]]>=len(streams[pending[kind][-1]]):pending[kind].pop()
   if not pending[kind]:pending[kind]=active[:];shufflers[kind].shuffle(pending[kind])
   d=pending[kind].pop();n=min(4,remaining[kind],len(streams[d])-pos[d]);need(n>0,'nonempty same-task microbatch')
   for _ in range(n):batchrows.append({'dataset_id':d,'row_id':streams[d][pos[d]],'task_type':kind});pos[d]+=1
   remaining[kind]-=n
  counts=[sum(x['task_type']==k for x in batchrows) for k in TYPES];need(len(batchrows)==batch and max(counts)-min(counts)<=1,'per-update balanced types');result.extend(batchrows)
 need(len(result)==16000 and len({(r['dataset_id'],r['row_id']) for r in result})==16000,'no repeated FIT rows');return result
def digest(rows):return hashlib.sha256(json.dumps(rows,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
