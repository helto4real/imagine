// Python-independent binding to the frozen Triton NVIDIA driver's own load/
// launch bodies. No tensor/kernel arithmetic or launch calculations are changed.
#include <cuda.h>
#include <dlfcn.h>
#include <cstdint>
#include <stdexcept>
#include <string>
static thread_local std::string error_message;
static void checked(CUresult status) {
    if(status==CUDA_SUCCESS) return;
    const char* text=nullptr;
    cuGetErrorString(status,&text);
    throw std::runtime_error(std::string("Triton Error [CUDA]: ")+text);
}
#define CUDA_CHECK(ans) checked(ans)
#define CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(ans) checked(ans)
using cuLaunchKernelEx_t=CUresult (*)(const CUlaunchConfig*,CUfunction,void**,void**);
static cuLaunchKernelEx_t getLaunchKernelExHandle() {
    void* handle=dlopen("libcuda.so.1",RTLD_LAZY);
    if(!handle) throw std::runtime_error("Failed to open libcuda.so.1");
    dlerror();auto result=reinterpret_cast<cuLaunchKernelEx_t>(dlsym(handle,"cuLaunchKernelEx"));
    if(dlerror()) {dlclose(handle);throw std::runtime_error("Failed to retrieve cuLaunchKernelEx from libcuda.so.1");}
    return result;
}
#include "original_triton_driver.inc"
extern "C" const char* imagine_triton_error() {return error_message.c_str();}
extern "C" int imagine_triton_load(const char* name,const void* data,int shared,int device,void** module,void** function,int* info) {
    try {original_load(name,data,shared,device,module,function,info);return 0;}
    catch(const std::exception& e){error_message=e.what();return -1;}
}
extern "C" int imagine_triton_pointer(uint64_t* pointer) {
    if(!*pointer) return 0;
    auto status=cuPointerGetAttribute(pointer,CU_POINTER_ATTRIBUTE_DEVICE_POINTER,*pointer);
    if(status==CUDA_ERROR_INVALID_VALUE) {error_message="Pointer argument cannot be accessed from Triton (cpu tensor?)";return -2;}
    try{checked(status);return 0;}catch(const std::exception& e){error_message=e.what();return -1;}
}
extern "C" int imagine_triton_launch(int gx,int gy,int gz,int warps,int ctas,int coop,int pdl,int shared,void* stream,void* function,void** args) {
    try{_launch(gx,gy,gz,warps,ctas,coop,pdl,shared,reinterpret_cast<CUstream>(stream),reinterpret_cast<CUfunction>(function),args);return 0;}
    catch(const std::exception& e){error_message=e.what();return -1;}
}
