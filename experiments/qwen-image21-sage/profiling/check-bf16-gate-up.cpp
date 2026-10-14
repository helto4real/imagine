// Verify direct BF16 production/consumption against explicit F32 -> BF16 -> F32.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
static void require(bool ok,const char *what){if(!ok){fprintf(stderr,"FAIL %s\n",what);exit(1);}}
static size_t compare(ggml_tensor *a,ggml_tensor *b,size_t offset,size_t bytes){std::vector<unsigned char>x(bytes),y(bytes);ggml_backend_tensor_get(a,x.data(),offset,bytes);ggml_backend_tensor_get(b,y.data(),offset,bytes);require(memcmp(x.data(),y.data(),bytes)==0,"exact rounded reference");return bytes;}
int main(int argc,char **){auto backend=ggml_backend_cuda_init(0);require(backend,"backend");size_t bytes=0;
 for(int test=0;test<(argc>1?1:13);++test){int m=test==1?5600:test==2?5657:test==3?5830:test==4?511:test==5?513:512;int k=4096,n=test==10?8192:24576;
  auto ctx=ggml_init({32*1024*1024,nullptr,true});auto g=ggml_new_graph(ctx);
  auto x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,m);auto w=ggml_new_tensor_2d(ctx,GGML_TYPE_I8,k,n);auto ws=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,n);
  ggml_set_input(x);ggml_set_input(w);ggml_set_input(ws);
  auto quant=ggml_quantize_i8_convrot(ctx,x,256);
  ggml_tensor *mat[2],*bf[2],*cast[2],*glu[2],*out[2],*reuse[2]={};
  for(int v=0;v<2;++v){
   mat[v]=ggml_mul_mat_i8_tensorwise(ctx,w,quant,ws,nullptr,256);
   if(v==0||test==6)ggml_set_output(mat[v]);
   bf[v]=ggml_cast(ctx,mat[v],GGML_TYPE_BF16);
   auto live=ggml_view_4d(ctx,bf[v],bf[v]->ne[0],bf[v]->ne[1],bf[v]->ne[2],bf[v]->ne[3],bf[v]->nb[1],bf[v]->nb[2],bf[v]->nb[3],0);
   live->src[1]=w;live->src[2]=quant;live->src[3]=ws;
   cast[v]=ggml_cast(ctx,live,GGML_TYPE_F32);
   if(v==0||test==7)ggml_set_output(cast[v]);
   glu[v]=test==9?ggml_swiglu_swapped(ctx,cast[v]):ggml_swiglu(ctx,cast[v]);
   out[v]=ggml_quantize_i8_convrot(ctx,glu[v],256);out[v]->src[1]=live;
   ggml_set_output(out[v]);if(test!=12)ggml_set_output(bf[v]);ggml_build_forward_expand(g,out[v]);
   if(test==8){reuse[v]=ggml_scale(ctx,glu[v],2.f);ggml_set_output(reuse[v]);ggml_build_forward_expand(g,reuse[v]);}
  }
  auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));require(ggml_gallocr_alloc_graph(alloc,g),"allocate");
  unsigned state=42;auto rnd=[&](){state=1664525u*state+1013904223u;return state;};
  std::vector<float> in(size_t(k)*m),scale(n);std::vector<int8_t> weight(size_t(k)*n);
  for(auto &v:in)v=(int(rnd()%20001)-10000)*.001f;
  for(auto &v:weight)v=int(rnd()%255)-127;
  for(auto &v:scale)v=(int(rnd()%2001)-1000)*.000017f;
  if(test==11)for(auto &v:weight)v=0;
  ggml_backend_tensor_set(x,in.data(),0,in.size()*4);ggml_backend_tensor_set(w,weight.data(),0,weight.size());ggml_backend_tensor_set(ws,scale.data(),0,scale.size()*4);
  require(ggml_backend_graph_compute(backend,g)==GGML_STATUS_SUCCESS,"compute");
  if(test!=12)bytes+=compare(bf[0],bf[1],0,ggml_nbytes(bf[0]));
  size_t padded=size_t((m+3)/4*4)*(n/2);bytes+=compare(out[0],out[1],0,padded);bytes+=compare(out[0],out[1],padded,m*4);
  std::vector<float> scales(m);ggml_backend_tensor_get(out[1],scales.data(),padded,m*4);for(float v:scales)require(std::isfinite(v),"finite scale");
  if(test==6)compare(mat[0],mat[1],0,ggml_nbytes(mat[0]));
  if(test==7)compare(cast[0],cast[1],0,ggml_nbytes(cast[0]));
  if(test==8)compare(reuse[0],reuse[1],0,ggml_nbytes(reuse[0]));
  printf("case=%d rows=%d n=%d exact=true\n",test,m,n);fflush(stdout);
  ggml_gallocr_free(alloc);ggml_free(ctx);
 }
 ggml_backend_free(backend);printf("PASS bf16_int8_padding_scale_bytes=%zu expected_bf16_gemm=%d expected_bf16_quant=%d\n",bytes,argc>1?1:10,argc>1?1:9);
}
