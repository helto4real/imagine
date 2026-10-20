#!/usr/bin/env python3
"""Kitchen geometry on tiny initialized graphs/meshes; no model weights."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_schedules.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
from reference_layout_completion import encoder
import comfy_kitchen as ck
from comfy_kitchen.backends import cuda
paths=['comfy_kitchen/backends/cuda/__init__.py'];sources={}
for path in paths:
 sha=hashlib.sha256((a.site/path).read_bytes()).hexdigest();assert sha==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==path);sources[path]=sha
base=encoder(torch,lambda t:[]if t.numel()==0 else t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist(),payload_hash=True)
def encode(v):
 if isinstance(v,cuda.MeshBVH):return dict(type='namedtuple',name='MeshBVH',items=[[k,encode(t)]for k,t in zip(v._fields,v,strict=True)])
 if isinstance(v,(tuple,list)):return dict(type='tuple'if isinstance(v,tuple)else'list',items=[encode(x)for x in v])
 return base(v)
# Input bytes are small and initialized. The standard encoder preserves strides.
input_encoder=encoder(torch,lambda t:[]if t.numel()==0 else t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist())
cases=[]
def record(op,args,kw=None,label=''):
 kw={}if kw is None else kw;c=dict(op=op,args=[input_encoder(x)for x in args],kwargs=[[k,input_encoder(v)]for k,v in kw.items()],label=label)
 try:
  c['output']=encode(getattr(cuda,op)(*args,**kw))
  if op=='delaunay3d':assert encode(getattr(cuda,op)(*args,**kw))==c['output'],'Original Delaunay repeated result changed'
 except Exception as e:
  reject_resource_failure(e)
  if any(t in str(e).lower()for t in ['illegal memory access','misaligned address']):raise
  c.update(error=type(e).__name__,message=str(e))
 cases.append(c)
with torch.no_grad():
 for dtype in [torch.int32,torch.int64]:
  for n in [0,1,4,7]:
   for edges in [[],[[0,0]],[[0,1],[1,2],[3,3]],[[0,1],[0,1],[2,3]]]:
    if edges and max(max(v)for v in edges)>=n:continue
    e=torch.tensor(edges,device='cuda:0',dtype=dtype).reshape(-1,2)
    record('connected_components',[e,n],label=f'components/{dtype}/{n}/{edges}')
  for value in [-1,False,True,.5,'own',None]:record('connected_components',[torch.empty(0,2,device='cuda:0',dtype=dtype),value],label=f'count/{dtype}/{value}')
  for edges,n in [([[-1,0]],3),([[0,3]],3),([[3,4]],2)]:record('connected_components',[torch.tensor(edges,device='cuda:0',dtype=dtype),n],label=f'range/{dtype}/{edges}')
 for shape in [(0,3,3),(2,4,3),(2,3)]:record('mesh_bvh',[torch.zeros(shape,device='cuda:0')],label=f'bvh_shape/{shape}')
 meshes=[]
 for n in [1,2,5]:
  triangles=(torch.arange(n*9,device='cuda:0',dtype=torch.float32)%13).reshape(n,3,3)*.125
  record('mesh_bvh',[triangles],label=f'bvh/{n}')
  bvh=cuda.mesh_bvh(triangles)
  for npoints in [0,1,7]:
   points=(torch.arange(npoints*3,device='cuda:0',dtype=torch.float32)%17).reshape(npoints,3)*.125
   for cutoff in [-.25,0.,.5,float('inf')]:
    c=dict(op='closest_point_on_mesh',triangles=input_encoder(triangles),args=[input_encoder(points)],kwargs=[['max_dist',input_encoder(cutoff)]],label=f'closest/{n}/{npoints}/{cutoff}')
    try:c['output']=encode(cuda.closest_point_on_mesh(bvh,points,cutoff))
    except Exception as e:reject_resource_failure(e);c.update(error=type(e).__name__,message=str(e))
    cases.append(c)
 vertices=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],device='cuda:0')
 faces=torch.tensor([[0,1,2],[0,1,3],[0,2,3],[1,2,3]],device='cuda:0',dtype=torch.int64)
 for dtype in [torch.int32,torch.int64]:
  for es in [[],[[0,1]],[[0,1],[0,2],[2,3]]]:
   edges=torch.tensor(es,device='cuda:0',dtype=dtype).reshape(-1,2)
   p=vertices[edges.long()].mean(1)if es else torch.empty(0,3,device='cuda:0')
   for threshold in [-1.,0.,.5,1.]:record('edge_collapse_checks',[vertices,faces.to(dtype),edges,p],dict(cos_threshold=threshold),label=f'collapse/{dtype}/{es}/{threshold}')
 # Symmetric nonnegative graphs; terminal-only and empty graphs are defined.
 for n,k in [(0,0),(1,0),(4,0),(2,1),(4,2)]:
  nbr=torch.full((n,k),-1,device='cuda:0',dtype=torch.int64)
  for i in range(n):
   for j in range(k):nbr[i,j]=(i+(1 if j==0 else-1))%n
  cap=torch.full((n,k),.25,device='cuda:0');s=torch.zeros(n,device='cuda:0');t=torch.zeros(n,device='cuda:0')
  if n:s[0]=1.;t[-1]=1.
  for every in [1,3,64,0,-2]:record('min_cut',[nbr,cap,s,t],dict(relabel_every=every),label=f'mincut/{n}/{k}/{every}')
 for points in [torch.empty(0,3,device='cuda:0'),torch.empty(2,4,device='cuda:0')]:record('delaunay3d',[points],label='delaunay_shape')
 for dtype in [torch.float32,torch.float64]:
  for n in [1,2,4,7,16,31]:
   for geometry in ['ordinary','planar','duplicate']:
    points=(((torch.arange(n*3,device='cuda:0',dtype=torch.float64)*17+11)%31)*.125).reshape(n,3).to(dtype)
    if geometry=='planar':points[:,2]=0.
    if geometry=='duplicate':points.zero_()
    record('delaunay3d',[points],label=f'delaunay/{dtype}/{n}/{geometry}')
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cases=cases),separators=(',',':'))+'\n')
print(f'Captured {len(cases)} bounded original geometry cases; model weights allocated: 0')
