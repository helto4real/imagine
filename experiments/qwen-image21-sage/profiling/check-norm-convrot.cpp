// Synthetic packed INT8/scale identity and fusion-guard checks; no model data.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

bool ggml_cuda_can_fuse_bf16_norm_mul_convrot(const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, const ggml_tensor *);
static void require(bool ok, const char * why) {
    if (!ok) { std::fprintf(stderr, "FAIL %s\n", why); std::exit(1); }
}
static ggml_tensor * dependency(ggml_context * ctx, ggml_tensor * x, ggml_tensor * other) {
    auto v = ggml_view_4d(ctx,x,x->ne[0],x->ne[1],x->ne[2],x->ne[3],x->nb[1],x->nb[2],x->nb[3],0);
    v->src[1]=other; return v;
}
int main(int argc,char **) {
    auto backend=ggml_backend_cuda_init(0); require(backend,"CUDA backend");
    size_t checked=0;
    for(int test=0;test<15;++test) {
        const int width=test==7?2048:4096;
        const int rows=test==0?1:(argc==1 && (test==2||test==3)?5600:17);
        const float eps=test==6?.5f:1e-6f;
        auto ctx=ggml_init({16*1024*1024,nullptr,true});
        auto storage=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,width,rows);
        auto weights=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,width);
        auto x=ggml_cast(ctx,storage,GGML_TYPE_BF16);
        auto scale=ggml_scale_bias(ctx,weights,1.f,1.f);
        auto input=dependency(ctx,x,scale);
        auto ref_cast=ggml_cast(ctx,input,GGML_TYPE_F32);
        auto ref_norm=ggml_norm(ctx,ref_cast,eps);
        auto ref_mul=ggml_mul(ctx,ref_norm,scale);
        auto ref_view=dependency(ctx,ref_mul,input);ref_view->src[2]=scale;
        ggml_set_output(ref_view); // Force the previous norm/modulation + quantization chain.
        auto reference=ggml_quantize_i8_convrot(ctx,ref_view,256);
        auto cast=ggml_cast(ctx,input,GGML_TYPE_F32);
        auto norm=ggml_norm(ctx,cast,eps);
        auto mul=ggml_mul(ctx,norm,scale);
        auto view=dependency(ctx,mul,input);view->src[2]=scale;
        auto quant=ggml_quantize_i8_convrot(ctx,view,256);
        if(test==8)ggml_set_output(cast);
        if(test==9)ggml_set_output(norm);
        if(test==10)ggml_set_output(mul);
        if(test==11)ggml_set_output(view);
        ggml_tensor * kept=test==8?cast:test==9?norm:test==10?mul:test==11?view:nullptr;
        ggml_tensor * ref_kept=test==8?ref_cast:test==9?ref_norm:test==10?ref_mul:test==11?ref_view:nullptr;
        if(ref_kept)ggml_set_output(ref_kept);
        auto graph=ggml_new_graph(ctx);
        for(auto v:{reference,quant}) { ggml_set_output(v);ggml_build_forward_expand(graph,v); }
        ggml_tensor * extra=nullptr,*ref_extra=nullptr;
        if(test>=12) {
            extra=ggml_scale(ctx,test==12?cast:test==13?norm:view,2.f);
            ref_extra=ggml_scale(ctx,test==12?ref_cast:test==13?ref_norm:ref_view,2.f);
            ggml_set_output(extra);ggml_build_forward_expand(graph,extra);
            ggml_set_output(ref_extra);ggml_build_forward_expand(graph,ref_extra);
        }
        auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        require(ggml_gallocr_alloc_graph(alloc,graph),"allocation");
        std::mt19937 rng(111+test);
        std::vector<float> data(size_t(width)*rows),row(width);
        for(auto &v:data)v=test==1?0.f:test==4?1.f:(int(rng()%20001)-10000)*(test==3?1e-8f:.001f);
        for(auto &v:row)v=test==5?-1.f:(int(rng()%8001)-4000)*.001f;
        ggml_backend_tensor_set(storage,data.data(),0,data.size()*4);
        ggml_backend_tensor_set(weights,row.data(),0,row.size()*4);
        require(ggml_cuda_can_fuse_bf16_norm_mul_convrot(cast,norm,mul,view,quant)==(width==4096),"layout eligibility");
        auto alias=*quant;alias.data=input->data;
        require(!ggml_cuda_can_fuse_bf16_norm_mul_convrot(cast,norm,mul,view,&alias),"input overlap rejection");
        alias.data=scale->data;
        require(!ggml_cuda_can_fuse_bf16_norm_mul_convrot(cast,norm,mul,view,&alias),"scale overlap rejection");
        auto bad_view=*view;bad_view.view_offs=2;
        require(!ggml_cuda_can_fuse_bf16_norm_mul_convrot(cast,norm,mul,&bad_view,quant),"offset rejection");
        auto bad_quant=*quant;bad_quant.op_params[0]=64;
        require(!ggml_cuda_can_fuse_bf16_norm_mul_convrot(cast,norm,mul,view,&bad_quant),"group rejection");
        require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"compute");
        const size_t scale_offset=size_t((rows+3)/4*4)*width;
        const size_t bytes=scale_offset+rows*sizeof(float);
        std::vector<unsigned char> a(bytes),b(bytes);
        ggml_backend_tensor_get(reference,a.data(),0,bytes);ggml_backend_tensor_get(quant,b.data(),0,bytes);
        size_t differences=0;for(size_t i=0;i<bytes;++i)differences+=a[i]!=b[i];
        for(int i=0;i<rows;++i) {float value;std::memcpy(&value,b.data()+scale_offset+i*4,4);require(std::isfinite(value)&&value>=0,"finite scale");}
        for(size_t i=size_t(rows)*width;i<scale_offset;++i)require(b[i]==0,"zero padded rows");
        for(int check=0;check<2;++check) {
            auto actual=check?extra:kept, expected=check?ref_extra:ref_kept;
            if(!actual)continue;
            std::vector<float> v(data.size()),r(data.size());
            ggml_backend_tensor_get(actual,v.data(),0,v.size()*4);
            ggml_backend_tensor_get(expected,r.data(),0,r.size()*4);
            for(float value:v)require(std::isfinite(value),"retained intermediate finite");
            require(std::memcmp(v.data(),r.data(),v.size()*4)==0,"retained/reused intermediate bits");
        }
        std::printf("case=%d width=%d rows=%d bytes=%zu differences=%zu\n",test,width,rows,bytes,differences);
        require(differences==0,"INT8 bytes and F32 scale bits");checked+=bytes;
        ggml_gallocr_free(alloc);ggml_free(ctx);
    }
    ggml_backend_free(backend);std::printf("PASS bytes=%zu expected_fused_calls=7\n",checked);
}
