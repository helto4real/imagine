// Metadata-only probe through the actual VAE runner; no weights or inference.
#include "model/vae/wan_vae.hpp"
#include "ggml-cuda.h"
#include <cstdio>
#include <cstdlib>
struct Probe : WAN::WanVAERunner {
    using WanVAERunner::WanVAERunner;
    void run(bool fused) {
        reset_compute_ctx();
        sd::Tensor<float> input({100,56,1,64});
        auto graph=get_compute_graph([&](){return build_graph(input,true);});
        collect_used_param_tensors(graph);
        for(int i=0;i<ggml_graph_n_nodes(graph);++i) {
            auto t=ggml_graph_node(graph,i);char name[160];
            std::snprintf(name,sizeof(name),"n%d_%s",i,ggml_op_name(t->op));ggml_set_name(t,name);
            if(!ggml_backend_supports_op(runtime_backend,t)) std::printf("UNSUPPORTED %s\n",name);
        }
        const auto& plan=resolve_graph_cut_plan(graph);
        sd::ComputeWorkspace workspace(runtime_backend);
        auto m=workspace.measure(graph,plan.compute_buffer_size,[&](const ggml_tensor* t){return params_tensor_set_.count(t)?runtime_backend:nullptr;},[](ggml_backend_sched_t,ggml_cgraph*){});
        std::printf("GRAPH fused=%d nodes=%d direct=%.6f MiB scheduler=%d\n",fused,ggml_graph_n_nodes(graph),plan.compute_buffer_size/1048576.,m.scheduler);
        for(const auto& b:m.buffers)std::printf("BUFFER %s %.6f MiB\n",ggml_backend_name(b.backend),b.bytes/1048576.);
        size_t resident=0,growth=0;sd_backend_graph_pool_memory(runtime_backend,graph,&resident,&growth);
        std::printf("POOL resident=%.6f MiB growth=%.6f MiB\n",resident/1048576.,growth/1048576.);
    }
};
int main(int argc,char**argv) {
    bool fused=argc>1 && std::atoi(argv[1]);
    setenv("SD_VAE_FUSED_UPSCALE_IM2COL",fused?"1":"0",1);
    auto backend=ggml_backend_cuda_init(0);
    // Only the shape marker used by the runner to select the 2D architecture.
    String2TensorStorage shapes;shapes["first_stage_model.decoder.conv1.weight"].ne[2]=64;
    { Probe p(backend,shapes,"first_stage_model",true,VERSION_QWEN_IMAGE_2_1);p.run(fused); }
    ggml_backend_free(backend);
}
