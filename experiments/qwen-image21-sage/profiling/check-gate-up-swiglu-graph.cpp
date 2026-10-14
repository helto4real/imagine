#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
static void require(bool ok,const char* s){if(!ok){fprintf(stderr,"FAIL %s\n",s);exit(2);}}
static size_t compare(ggml_tensor* a,ggml_tensor* b,size_t off,size_t size){std::vector<unsigned char>x(size),y(size);ggml_backend_tensor_get(a,x.data(),off,size);ggml_backend_tensor_get(b,y.data(),off,size);require(x==y,"bit identity");return size;}
int main(int argc,char**){auto backend=ggml_backend_cuda_init(0);require(backend,"CUDA");size_t bytes=0;
for(int test=0;test<(argc>1?1:10);++test){int m=test==1?5600:test==2?5657:test==3?5830:test==4?1024:1025;const int k=4096,n=test==7?8192:24576;
 auto ctx=ggml_init({32*1024*1024,nullptr,true});auto graph=ggml_new_graph(ctx);
 auto x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,m),w=ggml_new_tensor_2d(ctx,GGML_TYPE_I8,k,n),ws=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,n);
 for(auto t:{x,w,ws})ggml_set_input(t);
 ggml_tensor *mat[2],*h[2],*out[2],*reuse[2]={};
 for(int v=0;v<2;++v){auto input=ggml_scale(ctx,x,1.f);auto q=ggml_quantize_i8_convrot(ctx,input,256);mat[v]=ggml_mul_mat_i8_tensorwise(ctx,w,q,ws,nullptr,256);
  if(v==0||test==5)ggml_set_output(mat[v]);
  h[v]=test==6?ggml_swiglu_swapped(ctx,mat[v]):ggml_swiglu(ctx,mat[v]);
  if(test!=9)ggml_set_output(h[v]);
  auto live=ggml_view_4d(ctx,h[v],h[v]->ne[0],h[v]->ne[1],h[v]->ne[2],h[v]->ne[3],h[v]->nb[1],h[v]->nb[2],h[v]->nb[3],0);
  live->src[1]=w;live->src[2]=q;live->src[3]=ws;
  out[v]=ggml_quantize_i8_convrot(ctx,live,256);ggml_set_output(out[v]);ggml_build_forward_expand(graph,out[v]);
  if(test==8){reuse[v]=ggml_scale(ctx,live,2.f);ggml_set_output(reuse[v]);ggml_build_forward_expand(graph,reuse[v]);}
 }
 auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));require(ggml_gallocr_alloc_graph(alloc,graph),"allocation");
 unsigned state=17;auto rnd=[&](){state=state*1664525u+1013904223u;return state;};std::vector<float> in(size_t(k)*m),scales(n);std::vector<int8_t>weight(size_t(k)*n);
 for(auto&v:in)v=(int(rnd()%20001)-10000)*.0003f;
 for(auto&v:weight)v=test==9?0:int(rnd()%255)-127;
 for(auto&v:scales)v=(int(rnd()%2001)-1000)*.000017f;
 ggml_backend_tensor_set(x,in.data(),0,in.size()*4);ggml_backend_tensor_set(w,weight.data(),0,weight.size());ggml_backend_tensor_set(ws,scales.data(),0,scales.size()*4);
 require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"compute");
 if(test!=9)bytes+=compare(h[0],h[1],0,ggml_nbytes(h[0]));
 size_t padded=size_t((m+3)/4*4)*(n/2);bytes+=compare(out[0],out[1],0,padded);bytes+=compare(out[0],out[1],padded,m*4);
 std::vector<float> qs(m);ggml_backend_tensor_get(out[1],qs.data(),padded,m*4);for(float v:qs)require(std::isfinite(v),"finite scale");if(test==5)bytes+=compare(mat[0],mat[1],0,ggml_nbytes(mat[0]));if(reuse[0])bytes+=compare(reuse[0],reuse[1],0,ggml_nbytes(reuse[0]));
 printf("case=%d rows=%d width=%d exact=true\n",test,m,n);fflush(stdout);ggml_gallocr_free(alloc);ggml_free(ctx);
}printf("PASS checked_bytes=%zu\n",bytes);ggml_backend_free(backend);}
