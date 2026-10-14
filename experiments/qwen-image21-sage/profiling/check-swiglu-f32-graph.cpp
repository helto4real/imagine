#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
static void require(bool ok,const char *what){if(!ok){fprintf(stderr,"FAIL %s\n",what);exit(1);}}
static void compare(ggml_tensor *a,ggml_tensor *b,size_t offset,size_t bytes){std::vector<char>x(bytes),y(bytes);ggml_backend_tensor_get(a,x.data(),offset,bytes);ggml_backend_tensor_get(b,y.data(),offset,bytes);require(memcmp(x.data(),y.data(),bytes)==0,"exact tensor bytes");}
int main(int argc,char **){auto backend=ggml_backend_cuda_init(0);require(backend,"backend");size_t checked=0;
 for(int t=0;t<(argc>1?1:10);++t){int rows=t==1?5600:t==2?5657:t==3?5830:t==4?3:17,k=t==9?4096:12288;
  auto ctx=ggml_init({32*1024*1024,nullptr,true});auto g=ggml_new_graph(ctx);
  auto x=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,int64_t(rows)*2*k+(t==8));ggml_set_input(x);
  auto packed=ggml_view_2d(ctx,x,2*k,rows,size_t(2*k)*4,t==8?4:0);
  ggml_tensor *h[2],*q[2],*reuse[2]={};
  for(int v=0;v<2;++v){h[v]=t==7?ggml_swiglu_swapped(ctx,packed):ggml_swiglu(ctx,packed);if(v==0||t==5)ggml_set_output(h[v]);q[v]=ggml_quantize_i8_convrot(ctx,h[v],256);ggml_set_output(q[v]);ggml_build_forward_expand(g,q[v]);
   if(t==6){reuse[v]=ggml_scale(ctx,h[v],2.f);ggml_set_output(reuse[v]);ggml_build_forward_expand(g,reuse[v]);}
  }
  auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));require(ggml_gallocr_alloc_graph(alloc,g),"alloc");
  std::vector<float> input(ggml_nelements(x));unsigned s=42;for(auto &v:input){s=1664525u*s+1013904223u;v=(int(s%20001)-10000)/997.f;}ggml_backend_tensor_set(x,input.data(),0,input.size()*4);
  require(ggml_backend_graph_compute(backend,g)==GGML_STATUS_SUCCESS,"compute");
  size_t padded=size_t((rows+3)/4*4)*k;compare(q[0],q[1],0,padded);compare(q[0],q[1],padded,rows*4);checked+=padded+rows*4;
  if(t==5)compare(h[0],h[1],0,ggml_nbytes(h[0]));
  if(t==6)compare(reuse[0],reuse[1],0,ggml_nbytes(reuse[0]));
  printf("case=%d rows=%d k=%d exact=true\n",t,rows,k);fflush(stdout);ggml_gallocr_free(alloc);ggml_free(ctx);
 }
 ggml_backend_free(backend);printf("PASS int8_padding_scale_bytes=%zu expected_vector_calls=%d\n",checked,argc>1?1:5);
}
