// Synthetic graph/allocator validation of split-prefix Sage inputs.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
static void require(bool ok, const char* what) { if (!ok) { fprintf(stderr,"FAIL %s\n",what); exit(1); } }
static size_t check(ggml_tensor* a, ggml_tensor* b, size_t offset, size_t bytes, const char* label) {
    std::vector<unsigned char> x(bytes),y(bytes);
    for(auto t:{a,b})require(offset+bytes <= ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(t->buffer),t),"workspace bounds");
    require(cudaMemcpy(x.data(),(char*)a->data+offset,bytes,cudaMemcpyDeviceToHost)==cudaSuccess,"read reference");
    require(cudaMemcpy(y.data(),(char*)b->data+offset,bytes,cudaMemcpyDeviceToHost)==cudaSuccess,"read candidate");
    require(x==y,label); return bytes;
}
int main(int argc,char** argv) {
    auto backend=ggml_backend_cuda_init(0); require(backend,"CUDA"); size_t checked=0;
    const int ns[]={17,33,63,64,128,257,5600,5600,17};
    const int ps[]={1,31,32,64,57,230,57,230,57};
    for (int mode=1;mode<=2;++mode) for (int test=0;test<(argc>1 && std::strcmp(argv[1],"small")==0?1:9);++test) {
        const int n=ns[test],p=ps[test],d=128,h=32,l=n+p;
        auto ctx=ggml_init({16*1024*1024,nullptr,true}); auto g=ggml_new_graph(ctx);
        auto q=ggml_new_tensor_4d(ctx,GGML_TYPE_BF16,d,n,h,1);
        auto k=ggml_new_tensor_4d(ctx,GGML_TYPE_BF16,d,n,h,1);
        auto v=ggml_new_tensor_4d(ctx,GGML_TYPE_BF16,d,h,n,1);
        auto pk=ggml_new_tensor_4d(ctx,GGML_TYPE_BF16,d,p,h,1);
        auto pv=ggml_new_tensor_4d(ctx,GGML_TYPE_BF16,d,h,p,1);
        for(auto t:{q,k,v,pk,pv})ggml_set_input(t);
        auto kc=ggml_concat(ctx,pk,k,1), vc=ggml_concat(ctx,pv,v,2);
        auto vp=ggml_permute(ctx,vc,0,2,1,3);
        auto ref=ggml_sage_attn(ctx,q,kc,vp,0.08838834764831845f,GGML_SAGE_ATTN_2_PLUS_PLUS);
        auto out=ggml_sage_attn(ctx,q,kc,vp,0.08838834764831845f,GGML_SAGE_ATTN_2_PLUS_PLUS);
        out->src[2]=ggml_permute(ctx,v,0,2,1,3); out->src[4]=ggml_permute(ctx,pv,0,2,1,3);
        if(mode==2){out->src[1]=k;out->src[5]=pk;}
        require(ggml_backend_supports_op(backend,out),"split support");
        ggml_set_output(ref);ggml_set_output(out);ggml_build_forward_expand(g,ref);ggml_build_forward_expand(g,out);
        auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));require(ggml_gallocr_alloc_graph(alloc,g),"alloc");
        unsigned state=42;
        for(auto t:{q,k,v,pk,pv}) {
            std::vector<float> input(ggml_nelements(t));std::vector<ggml_bf16_t> bf(input.size());
            for(auto& a:input){state=state*1664525u+1013904223u;a=test==8?0.f:(int(state%20001)-10000)*0.0002f;}
            ggml_fp32_to_bf16_row_ref(input.data(),bf.data(),bf.size());ggml_backend_tensor_set(t,bf.data(),0,bf.size()*2);
        }
        require(ggml_backend_graph_compute(backend,g)==GGML_STATUS_SUCCESS,"compute");
        checked+=check(ref,out,0,ggml_nbytes(ref),"attention output");
        size_t offset=GGML_PAD(ggml_nbytes(ref),256);
        auto region=[&](size_t bytes,const char* label){checked+=check(ref,out,offset,bytes,label);offset+=GGML_PAD(bytes,256);};
        region(size_t(d)*n*h,"Q int8");region(size_t(d)*l*h,"K int8");
        region(size_t(GGML_PAD(n,128))/32*8*h*4,"Q scales");
        region(size_t(GGML_PAD(l,64))/64*4*h*4,"K scales");region(d*h*4,"K mean");
        region(size_t(d)*GGML_PAD(l,64)*h,"V fp8 padded");region(d*h*4,"V scales");
        region(ggml_nelements(ref)*2,"attention fp16");
        auto saved=out->src[4]; saved->ne[1]++;
        require(!ggml_backend_supports_op(backend,out),"reject mismatched prefix length");saved->ne[1]--;
        if(mode==2) {
            pk->ne[1]++;
            require(!ggml_backend_supports_op(backend,out),"reject mismatched K prefix");pk->ne[1]--;
            out->src[4]=nullptr;
            require(!ggml_backend_supports_op(backend,out),"reject split K without split V");out->src[4]=saved;
            auto type=pk->type;pk->type=GGML_TYPE_F16;
            require(!ggml_backend_supports_op(backend,out),"reject non-BF16 K prefix");pk->type=type;
        }
        printf("mode=%d case=%d n=%d prefix=%d exact=true\n",mode,test,n,p);fflush(stdout);
        ggml_gallocr_free(alloc);ggml_free(ctx);
    }
    printf("PASS checked_bytes=%zu\n",checked);ggml_backend_free(backend);
}
