#include "ggml-cuda/common.cuh"
#include "ggml-cuda/unary.cuh"
#include "../runs/swiglu-f32-build/baseline-kernel.cuh"
#include "swiglu-f32-kernel.cuh"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
static void check(cudaError_t s){if(s!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(s));exit(2);}}
static unsigned rng(unsigned &s){s^=s<<13;s^=s>>17;s^=s<<5;return s;}
static size_t run(int rows,int k,int mode,bool time, bool confirm = false) {
 std::vector<float> input(size_t(rows)*2*k);unsigned state=42;
 for(size_t i=0;i<input.size();++i){float v=(int(rng(state)%20001)-10000)/997.f;
  if(mode==1)v=i%2?-0.f:0.f;
  if(mode==2)v*=1.e-12f;
  if(mode==3)v*=((i%(2*k))<size_t(k)?8.f:1.e12f);
  if(mode==4)v=(i%257==0)?v*100.f:v*.001f;
  input[i]=v;
 }
 float *x,*s;int8_t *y;check(cudaMalloc(&x,input.size()*4));check(cudaMemcpy(x,input.data(),input.size()*4,cudaMemcpyHostToDevice));
 check(cudaMalloc(&s,rows*4));check(cudaMalloc(&y,size_t(rows)*k));
 #define LIMIT(F) check(cudaFuncSetAttribute(F,cudaFuncAttributeMaxDynamicSharedMemorySize,49152))
 LIMIT((quantize_rowwise_i8_convrot_cuda<1024,true>));LIMIT((quantize_rowwise_i8_convrot_cuda<512,true>));LIMIT((quantize_rowwise_i8_convrot_cuda<256,true>));
 LIMIT((quantize_swiglu_i8_convrot_vec_cuda<128>));LIMIT((quantize_swiglu_i8_convrot_vec_cuda<256>));LIMIT((quantize_swiglu_i8_convrot_vec_cuda<512>));
 LIMIT((quantize_swiglu_i8_convrot_warp_cuda<128>));LIMIT((quantize_swiglu_i8_convrot_warp_cuda<256>));LIMIT((quantize_swiglu_i8_convrot_warp_cuda<512>));LIMIT((quantize_swiglu_i8_convrot_warp_cuda<1024>));
 auto launch=[&](int v){
  switch(v){
   case 0:quantize_rowwise_i8_convrot_cuda<1024,true><<<rows,1024,k*4>>>(x,y,s,k,rows,k);break;
   case 1:quantize_rowwise_i8_convrot_cuda<512,true><<<rows,512,k*4>>>(x,y,s,k,rows,k);break;
   case 2:quantize_rowwise_i8_convrot_cuda<256,true><<<rows,256,k*4>>>(x,y,s,k,rows,k);break;
   case 3:quantize_swiglu_i8_convrot_warp_cuda<128><<<rows,128,k*4>>>(x,y,s,k,rows);break;
   case 4:quantize_swiglu_i8_convrot_warp_cuda<256><<<rows,256,k*4>>>(x,y,s,k,rows);break;
   case 5:quantize_swiglu_i8_convrot_warp_cuda<512><<<rows,512,k*4>>>(x,y,s,k,rows);break;
   case 6:quantize_swiglu_i8_convrot_warp_cuda<1024><<<rows,1024,k*4>>>(x,y,s,k,rows);break;
   case 7:quantize_swiglu_i8_convrot_vec_cuda<128><<<rows,128,k*4>>>(x,y,s,k,rows);break;
   case 8:quantize_swiglu_i8_convrot_vec_cuda<256><<<rows,256,k*4>>>(x,y,s,k,rows);break;
   case 9:quantize_swiglu_i8_convrot_vec_cuda<512><<<rows,512,k*4>>>(x,y,s,k,rows);break;
  }check(cudaGetLastError());
 };
 std::vector<int8_t> ref(size_t(rows)*k),actual(ref.size());std::vector<float> rs(rows),as(rows);
 launch(0);check(cudaDeviceSynchronize());check(cudaMemcpy(ref.data(),y,ref.size(),cudaMemcpyDeviceToHost));check(cudaMemcpy(rs.data(),s,rows*4,cudaMemcpyDeviceToHost));
 for(int v=1;v<10;++v){launch(v);check(cudaDeviceSynchronize());check(cudaMemcpy(actual.data(),y,actual.size(),cudaMemcpyDeviceToHost));check(cudaMemcpy(as.data(),s,rows*4,cudaMemcpyDeviceToHost));
  if(memcmp(ref.data(),actual.data(),ref.size())||memcmp(rs.data(),as.data(),rows*4)){fprintf(stderr,"MISMATCH rows=%d k=%d mode=%d variant=%d\n",rows,k,mode,v);exit(3);}
 }
 for(float v:rs)if(!std::isfinite(v)){fprintf(stderr,"nonfinite scale\n");exit(4);}
 printf("{\"rows\":%d,\"k\":%d,\"mode\":%d,\"bytes_per_variant\":%zu,\"exact_variants\":9,\"timings\":[",rows,k,mode,ref.size()+rows*4);
 if(time){for(int i=0;i<12;++i)for(int v=0;v<10;++v)launch(v);check(cudaDeviceSynchronize());cudaEvent_t a,b;check(cudaEventCreate(&a));check(cudaEventCreate(&b));
  for(int batch=0;batch<(confirm?24:10);++batch){float ms[10]={};for(int j=0;j<(confirm?2:10);++j){int v=confirm?(((batch+j)%2)?8:0):(batch+j)%10;check(cudaEventRecord(a));for(int i=0;i<(confirm?100:30);++i)launch(v);check(cudaEventRecord(b));check(cudaEventSynchronize(b));check(cudaEventElapsedTime(ms+v,a,b));}
   printf("%s[",batch?",":"");for(int v=0;v<10;++v)printf("%s%.6f",v?",":"",ms[v]/(confirm?100:30));printf("]");
  }check(cudaEventDestroy(a));check(cudaEventDestroy(b));
 }printf("]}\n");fflush(stdout);check(cudaFree(x));check(cudaFree(y));check(cudaFree(s));return ref.size()+rows*4;
}
int main(int argc,char **argv){if(argc>1 && std::strcmp(argv[1],"confirm")==0){run(5600,12288,0,true,true);return 0;}if(argc>1){run(17,12288,0,false);return 0;}
 for(int n:{1,17,57,230,5600,5657,5830})run(n,12288,0,n>=5600);
 for(int mode=1;mode<=4;++mode)run(17,12288,mode,false);
 run(17,256,0,false);run(17,768,0,false);run(17,4096,0,false);
}
