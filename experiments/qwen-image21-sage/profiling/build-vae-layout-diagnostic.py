"""Build metadata-only instrumentation from source copies; leave runtime source/build intact."""
from pathlib import Path
import concurrent.futures,hashlib,json,shlex,shutil,subprocess
ROOT=Path(__file__).resolve().parents[3]
SRC=ROOT/'stable-diffusion.cpp'; BUILD=SRC/'build-cuda'
OUT=ROOT/'experiments/qwen-image21-sage/runs/vae-layout-diagnostic-build';OUT.mkdir(exist_ok=True)
OLD=OUT.parent/'vae-upscale-im2col-build'
entries=json.loads((BUILD/'compile_commands.json').read_text())
paths=['src/core/ggml_runner.cpp','src/model_manager.cpp','ggml/src/ggml-alloc.c']
original={p:(SRC/p).read_text() for p in paths}
(OUT/'source-before.json').write_text(json.dumps({p:hashlib.sha256((SRC/p).read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
header=r'''
#include <cstdio>
#include <unordered_map>
extern "C" int sd_vae_allocation_trace;
static void sd_dump_vae_graph(ggml_cgraph* graph, const char* dir, int sequence) {
    std::vector<ggml_tensor*> ts;
    std::unordered_map<ggml_tensor*, int> ids;
    auto add=[&](ggml_tensor* t){if(t && !ids.count(t)){ids[t]=int(ts.size());ts.push_back(t);}};
    for(int i=0;i<ggml_graph_n_nodes(graph);++i)add(ggml_graph_node(graph,i));
    for(int i=0;i<sd::ggml_graph_cut::leaf_count(graph);++i)add(sd::ggml_graph_cut::leaf_tensor(graph,i));
    for(size_t i=0;i<ts.size();++i){add(ts[i]->view_src);for(auto s:ts[i]->src)add(s);}
    auto id=[&](ggml_tensor* t){return t?ids.at(t):-1;};
    const std::string path=std::string(dir)+"/graph-"+std::to_string(sequence)+".tsv";
    FILE* f=std::fopen(path.c_str(),"w");GGML_ASSERT(f);
    std::fprintf(f,"id\tname\top\ttype\tshape\tstrides\tflags\tview\tview_offs\tsources\top_params\texternal_data\n");
    for(size_t i=0;i<ts.size();++i){auto t=ts[i];
        std::fprintf(f,"%zu\t%s\t%s\t%s\t",i,t->name,ggml_op_name(t->op),ggml_type_name(t->type));
        for(int d=0;d<4;++d)std::fprintf(f,"%s%lld",d?",":"",(long long)t->ne[d]);
        std::fprintf(f,"\t");for(int d=0;d<4;++d)std::fprintf(f,"%s%zu",d?",":"",t->nb[d]);
        std::fprintf(f,"\t%d\t%d\t%zu\t",t->flags,id(t->view_src),t->view_offs);
        for(int d=0;d<GGML_MAX_SRC;++d)std::fprintf(f,"%s%d",d?",":"",id(t->src[d]));
        std::fprintf(f,"\t");for(size_t d=0;d<sizeof(t->op_params)/sizeof(t->op_params[0]);++d)std::fprintf(f,"%s%08x",d?",":"",(unsigned)t->op_params[d]);
        std::fprintf(f,"\t%d\n",t->data!=nullptr);
    }
    std::fclose(f);
}
'''
r=original[paths[0]].replace('using namespace sd;',header+'\nusing namespace sd;')
needle='    const auto params           = collect_used_param_tensors(graph);'
assert r.count(needle)==1
r=r.replace(needle,r'''    const char* trace_dir=std::getenv("SD_VAE_LAYOUT_TRACE_DIR");
    const bool trace_vae=get_desc()=="wan_vae" && trace_dir && *trace_dir;
    const int old_trace=sd_vae_allocation_trace;
    struct RestoreTrace { int value; ~RestoreTrace(){sd_vae_allocation_trace=value;} } restore_trace{old_trace};
    static int trace_sequence=0;
    if(trace_vae) {
        sd_vae_allocation_trace=1;
        ++trace_sequence;
        std::fprintf(stderr,"VAETRACE BEGIN graph=%d nodes=%d\n",trace_sequence,ggml_graph_n_nodes(graph));
        sd_dump_vae_graph(graph,trace_dir,trace_sequence);
    }
'''+needle)
needle='    if (full_measurement.buffers.empty()) {'
r=r.replace(needle,r'''    if(trace_vae) {
        std::fprintf(stderr,"VAETRACE PLAN direct=%zu scheduler=%d pending_cache=%zu\n",cached_plan.compute_buffer_size,full_measurement.scheduler,cache_.pending_bytes(graph));
        for(const auto& b:full_measurement.buffers)std::fprintf(stderr,"VAETRACE BUFFER backend=%s bytes=%zu\n",ggml_backend_name(b.backend),b.bytes);
    }
'''+needle)
needle='        requests.push_back({size.backend, reinterpret_cast<uintptr_t>(this), pending,'
r=r.replace(needle,r'''        if(sd_vae_allocation_trace)std::fprintf(stderr,"VAETRACE REQUEST graph=%zu reusable=%zu retained=%zu cache=%zu pending=%zu pool_resident=%zu pool_growth=%zu limit=%zu\n",size.bytes,reusable,retained,cache_bytes,pending,pool_resident,pool_growth,limit);
'''+needle)
m=original[paths[1]]
m='extern "C" int sd_vae_allocation_trace;\n'+m
needle='    auto available_device_bytes    = [&](ggml_backend_t backend) {'
assert needle in m
m=m.replace(needle,r'''    if(sd_vae_allocation_trace)std::fprintf(stderr,"VAETRACE CAPACITY pending=%zu missing_weights=%zu pool_resident=%zu pool_growth=%zu safety=%zu required_device=%zu required_budget=%zu\n",request.pending_allocation_bytes,missing,request.backend_pool_resident_bytes,request.backend_pool_growth_bytes,safety_margin,result.required_device_bytes,result.required_budget_bytes);
'''+needle)
a=original[paths[2]].replace('//#define GGML_ALLOCATOR_DEBUG','#define GGML_ALLOCATOR_DEBUG\nint sd_vae_allocation_trace = 0;')
a=a.replace('if (cur_max > chunk->max_size) {','if (sd_vae_allocation_trace && cur_max > chunk->max_size) {')
needle='    // reallocate buffers from a measure graph' # actual insertion below
needle='    // reallocate buffers if needed\n'
assert a.count(needle)==1
a=a.replace(needle,r'''    if(sd_vae_allocation_trace) {
        for(int i=0;i<graph->n_nodes;++i){
            struct node_alloc * n=&galloc->node_allocs[i];
            if(n->dst.buffer_id>=0)fprintf(stderr,"VAETRACE ALLOC node=%d name=%s buffer=%d chunk=%d offset=%zu size=%zu\n",i,graph->nodes[i]->name,n->dst.buffer_id,n->dst.addr.chunk,n->dst.addr.offset,n->dst.size_max);
        }
    }
'''+needle)
changed=dict(zip(paths,[r,m,a]));commands=[]
for p in paths:
 dest=OUT/Path(p).name;dest.write_text(changed[p])
 entry=next(e for e in entries if e['file']==str(SRC/p));cmd=shlex.split(entry['command'])
 obj=OUT/(Path(p).name+'.o');cmd[cmd.index('-o')+1]=str(obj);cmd[cmd.index('-c')+1]=str(dest)
 commands.append(cmd)
def compile_one(cmd):
 with (OUT/(Path(cmd[cmd.index('-c')+1]).name+'.build.log')).open('w') as f:subprocess.run(cmd,cwd=BUILD,stdout=f,stderr=subprocess.STDOUT,check=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(compile_one,commands))
for archive,objects in [('libstable-diffusion.a',['ggml_runner.cpp.o','model_manager.cpp.o']),('ggml/src/libggml-base.a',['ggml-alloc.c.o'])]:
 target=OUT/Path(archive).name;shutil.copy2(BUILD/archive,target)
 subprocess.run(['ar','r',str(target),*[str(OUT/o) for o in objects]],check=True)
link=subprocess.check_output(['ninja','-t','commands','bin/sd-server'],cwd=BUILD,text=True).splitlines()[-1]
assert link.startswith(': && ') and link.endswith(' && :')
cmd=shlex.split(link[5:-5]);cmd=[s for s in cmd if not s.startswith('-Wl,--dependency-file=')]
cmd=[str(OUT/Path(s).name) if s in ('libstable-diffusion.a','ggml/src/libggml-base.a') else s for s in cmd]
cmd[cmd.index('-o')+1]=str(OUT/'sd-server');commands.append(cmd)
with (OUT/'link.log').open('w') as f:subprocess.run(cmd,cwd=BUILD,stdout=f,stderr=subprocess.STDOUT,check=True)
for n in ('baseline-environment.json','after-hashes.json'):shutil.copy2(OLD/n,OUT/n)
(OUT/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
(OUT/'binary-hashes.json').write_text(json.dumps({'sd-server':hashlib.sha256((OUT/'sd-server').read_bytes()).hexdigest()},indent=2)+'\n')
assert all((SRC/p).read_text()==original[p] for p in paths)
print(OUT,flush=True)
