// Synthetic F32 gate/up GEMM -> SwiGLU -> ConvRot experiment. No model data.
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/unary.cuh"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "../runs/gate-up-swiglu-build/gemm-reference.cuh"
#include "../runs/gate-up-swiglu-build/convrot-reference.cuh"
#include "gate-up-swiglu-kernel.cuh"
#include "gate-up-direct-kernel.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

namespace Candidate {
namespace evt = cutlass::epilogue::threadblock;
using Store = GateUpSwiGLU::VisitorSwiGLUStore<Reference::Map,float,Reference::rounding,cute::Stride<int64_t,cute::_1,cute::_0>>;
using Epilogue = evt::Sm80EVT<Store,Reference::ScaledOutput>;
using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
 int8_t,cutlass::layout::RowMajor,cutlass::ComplexTransform::kNone,16,
 int8_t,cutlass::layout::ColumnMajor,cutlass::ComplexTransform::kNone,16,
 float,cutlass::layout::RowMajor,4,int32_t,float,
 cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
 Reference::Tile,Reference::Warp,cutlass::gemm::GemmShape<16,8,32>,Epilogue,
 cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,3,
 cutlass::arch::OpMultiplyAddSaturate>::GemmKernel;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
}
using Direct = GateUpDirect::Projection<Reference::Tile,Reference::Warp,Reference::Map,
    Reference::ScaledInput,Reference::Multiply,Candidate::Store>;
static void check(cudaError_t s){if(s!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(s));exit(2);}}
static void require(bool ok,const char* s){if(!ok){fprintf(stderr,"FAIL: %s\n",s);exit(3);}}
template<class T> static T* alloc(size_t n){T* p;check(cudaMalloc(&p,n*sizeof(T)));return p;}
static __device__ unsigned random_word(unsigned i){i^=i>>16;i*=0x7feb352d;i^=i>>15;i*=0x846ca68b;i^=i>>16;return i;}
static __global__ void fill_i8(int8_t* p,size_t n,unsigned seed,bool zero){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)p[i]=zero?0:int(random_word(unsigned(i)+seed)%255)-127;}
static __global__ void fill_scale(float* s,int n,bool channels){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=(random_word(i+17)%1001+1)*.0000027f;if(i%257==0)v=0.f;if(i%257==1)v*=1.e-15f;if(i%257==2)v*=100.f;s[i]=v*(channels?.7f:1.13f);}}
static __global__ void interleave_weights(const int4* w,int4* dst,const float* scale,float* packed,int k,int half){int col=blockIdx.y;int original=(col&1)*half+col/2;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<k/16)dst[size_t(col)*(k/16)+i]=w[size_t(original)*(k/16)+i];if(i==0)packed[col]=scale[original];}
static __global__ void reference_swiglu(const float* p,float* y,size_t n,int width){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n){size_t j=i/width*(2*width)+i%width;y[i]=ggml_cuda_op_silu_single(p[j])*p[j+width];}}
template<class Gemm,class Epilogue> static void gemm(const int8_t* w,const int8_t* x,const float* xs,const float* ws,float* out,int m,int k,int n){
 typename Epilogue::Arguments ep{{{{},{xs,1.f,{}},{}},{ws,1.f,{}},{}},{out,{n,cute::_1{},cute::_0{}}}};
 typename Gemm::Arguments a{cutlass::gemm::GemmUniversalMode::kGemm,{m,n,k},1,ep,x,w,nullptr,out,0,0,0,0,k,k,n,n};
 Gemm g;require(g.can_implement(a)==cutlass::Status::kSuccess,"GEMM support");require(g.get_workspace_size(a)==0,"workspace");require(g(a)==cutlass::Status::kSuccess,"GEMM launch");
}
template<class T> static size_t compare(T* a,T* b,size_t n,const char* name){std::vector<T>x(n),y(n);check(cudaMemcpy(x.data(),a,n*sizeof(T),cudaMemcpyDeviceToHost));check(cudaMemcpy(y.data(),b,n*sizeof(T),cudaMemcpyDeviceToHost));if(std::memcmp(x.data(),y.data(),n*sizeof(T))){size_t count=0,first=0;for(size_t i=0;i<n;++i)if(std::memcmp(&x[i],&y[i],sizeof(T))){if(!count)first=i;++count;}fprintf(stderr,"MISMATCH %s: %zu values, first %zu: %.9g vs %.9g\n",name,count,first,double(x[first]),double(y[first]));exit(4);}if constexpr(std::is_same<T,float>::value)for(float v:x)require(std::isfinite(v),"finite output");return n*sizeof(T);}
static size_t run(int m,bool zero,bool timed,int batches=12,int iterations=30){
 constexpr int k=4096,h=12288,n=2*h;size_t size=size_t(m)*h;
 auto x=alloc<int8_t>(size_t(m)*k),w=alloc<int8_t>(size_t(n)*k),wp=alloc<int8_t>(size_t(n)*k);
 auto xs=alloc<float>(m),ws=alloc<float>(n),wsp=alloc<float>(n),packed=alloc<float>(size*2),ref_h=alloc<float>(size),out=alloc<float>(size);
 auto qr=alloc<int8_t>(size),qc=alloc<int8_t>(size);auto sr=alloc<float>(m),sc=alloc<float>(m);
 fill_i8<<<(size_t(m)*k+255)/256,256>>>(x,size_t(m)*k,7,false);fill_i8<<<(size_t(n)*k+255)/256,256>>>(w,size_t(n)*k,999,zero);
 fill_scale<<<(m+255)/256,256>>>(xs,m,false);fill_scale<<<(n+255)/256,256>>>(ws,n,true);
 cudaEvent_t a,b;check(cudaEventCreate(&a));check(cudaEventCreate(&b));check(cudaEventRecord(a));
 interleave_weights<<<dim3((k/16+255)/256,n),256>>>(reinterpret_cast<int4*>(w),reinterpret_cast<int4*>(wp),ws,wsp,k,h);
 check(cudaEventRecord(b));check(cudaEventSynchronize(b));float packing_ms;check(cudaEventElapsedTime(&packing_ms,a,b));
 check(cudaFuncSetAttribute(quantize_rowwise_i8_convrot_cuda<1024,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,h*4));
 check(cudaFuncSetAttribute(quantize_rowwise_i8_convrot_cuda<1024,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,h*4));
 auto launch=[&](int mode){
   if(mode==0) {
     interleave_weights<<<dim3((k/16+255)/256,n),256>>>(reinterpret_cast<int4*>(w),reinterpret_cast<int4*>(wp),ws,wsp,k,h);
     gemm<Candidate::Gemm,Candidate::Epilogue>(wp,x,xs,wsp,ref_h,m,k,n);
     quantize_rowwise_i8_convrot_cuda<1024,false><<<m,1024,h*4>>>(ref_h,qr,sr,h,m,h);
   } else {
     gemm<Direct::Gemm,Direct::Epilogue>(w,x,xs,ws,out,m,k,n);
     quantize_rowwise_i8_convrot_cuda<1024,false><<<m,1024,h*4>>>(out,qc,sc,h,m,h);
   }
   check(cudaGetLastError());
 };
 launch(0);launch(1);check(cudaDeviceSynchronize());
 size_t bytes=compare(ref_h,out,size,"F32 SwiGLU")+compare(qr,qc,size,"INT8 ConvRot")+compare(sr,sc,m,"row scales");
 printf("{\"rows\":%d,\"zero_weights\":%s,\"exact_bytes\":%zu,\"batches_ms\":[",m,zero?"true":"false",bytes);fflush(stdout);
 if(timed){for(int i=0;i<20;++i)launch(i%2);check(cudaDeviceSynchronize());for(int batch=0;batch<batches;++batch){float ms[2];for(int side=0;side<2;++side){int mode=(batch+side)%2;check(cudaEventRecord(a));for(int j=0;j<iterations;++j)launch(mode);check(cudaEventRecord(b));check(cudaEventSynchronize(b));check(cudaEventElapsedTime(&ms[mode],a,b));ms[mode]/=iterations;}printf("%s[%.6f,%.6f]",batch?",":"",ms[0],ms[1]);fflush(stdout);}}
 printf("]}\n");fflush(stdout);
 for(auto p:{(void*)x,(void*)w,(void*)wp,(void*)xs,(void*)ws,(void*)wsp,(void*)packed,(void*)ref_h,(void*)out,(void*)qr,(void*)qc,(void*)sr,(void*)sc})check(cudaFree(p));check(cudaEventDestroy(a));check(cudaEventDestroy(b));return bytes;
}
int main(int argc,char**argv){size_t bytes=0;if(argc>1 && std::strcmp(argv[1],"small")==0){run(17,false,false);return 0;}if(argc>1 && std::strcmp(argv[1],"confirm")==0){run(5600,false,true,24,50);return 0;}for(int m:{1,17,57,230,5600,5657,5830})bytes+=run(m,false,m>=5600);bytes+=run(17,true,false);fprintf(stderr,"PASS exact_bytes=%zu\n",bytes);}
