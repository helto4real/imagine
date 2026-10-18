#include "bridge.hpp"
#include "tensor_view.hpp"
#include <ATen/CPUGeneratorImpl.h>
#include <c10/core/DefaultDtype.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/_scaled_mm_v2.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <ATen/native/transformers/cuda/sdp_utils.h>
#include <ATen/ops/embedding.h>
#include <ATen/ops/amax.h>
#include <ATen/ops/cat.h>
#include <ATen/record_function.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/csrc/autograd/function.h>
#include <c10/util/StringUtil.h>
#include <array>
#include <string_view>
#include <ATen/BlasBackend.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
// LibTorch declares this exported entrypoint only under its build's USE_CUDA.
// This bridge already links the pinned CUDA build; keep upstream compile flags.
namespace torch { bool CudaIPCCollect(); }

#include <cstring>
#include <string>
#include <sstream>
#include <utility>

namespace {
thread_local std::string last_error;
thread_local std::string last_metadata;
thread_local at::CallbackHandle rms_callback = 0;
thread_local RmsTemporaryMetadata rms_temporary{};
std::unique_ptr<at::ObserverContext> rms_metadata_start(const at::RecordFunction&) { return nullptr; }
void rms_metadata_end(const at::RecordFunction& fn, at::ObserverContext*) {
    if (std::strcmp(fn.name(), "aten::_fused_rms_norm") != 0) return;
    const auto& outputs = fn.outputs();
    if (outputs.size() != 2 || !outputs[1].isTensor()) return;
    const auto& stats = outputs[1].toTensor();
    if (!stats.is_cuda() || stats.scalar_type() != at::kFloat) return;
    // Metadata only; no tensor retained, no storage read, fill or stream change.
    rms_temporary.start = reinterpret_cast<uintptr_t>(stats.data_ptr());
    rms_temporary.bytes = stats.nbytes();
    ++rms_temporary.count;
}
template<class F> int protect(F&& f) noexcept {
    try { f(); last_error.clear(); return 0; }
    catch(const std::exception& e) { last_error=e.what(); return -1; }
    catch(...) { last_error="Unknown native exception"; return -1; }
}
template<class T> struct Convert {
    static T get(const KitchenArg& x) {
        if constexpr(std::is_integral_v<T>) {
            if(x.kind!=2 && x.kind!=4) throw std::runtime_error("Expected integer/bool argument");
            return static_cast<T>(x.integer);
        } else if constexpr(std::is_floating_point_v<T>) {
            if(x.kind!=3) throw std::runtime_error("Expected floating argument");
            return static_cast<T>(x.real);
        } else { static_assert(sizeof(T)==0,"Unhandled native argument type"); }
    }
};
template<class... Tags> struct Convert<TensorView<Tags...>> {
    static auto get(const KitchenArg& x) {
        if(x.kind!=1 || !x.tensor) throw std::runtime_error("Expected tensor argument");
        return TensorView<Tags...>(*static_cast<const at::Tensor*>(x.tensor));
    }
};
template<> struct Convert<OptionalTensorView> {
    static OptionalTensorView get(const KitchenArg& x) {
        if(x.kind!=0 && x.kind!=1) throw std::runtime_error("Expected tensor or None");
        return {static_cast<const at::Tensor*>(x.tensor)};
    }
};
template<class T> struct Convert<std::optional<T>> {
    static std::optional<T> get(const KitchenArg& x) {
        return x.kind==0 ? std::nullopt : std::optional<T>(Convert<T>::get(x));
    }
};
template<class R,class... A,size_t... I>
int64_t invoke_impl(R(*fn)(A...),const KitchenArg* args,std::index_sequence<I...>) {
    if constexpr(std::is_void_v<R>) {
        fn(Convert<std::remove_cvref_t<A>>::get(args[I])...); return 0;
    } else if constexpr(std::is_same_v<R,PlanMap>) {
        auto plan=fn(Convert<std::remove_cvref_t<A>>::get(args[I])...);
        std::ostringstream json; json << "{"; bool first=true;
        for(const auto& [key,value]:plan) {
            if(key.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789")!=std::string::npos)
                throw std::runtime_error("Invalid plan metadata key");
            if(!first) json << ","; first=false;
            json << '"' << key << "\":" << value;
        }
        json << "}"; last_metadata=json.str(); return 0;
    } else { return fn(Convert<std::remove_cvref_t<A>>::get(args[I])...); }
}
template<class R,class... A>
int64_t invoke(R(*fn)(A...),const KitchenArg* args,size_t count) {
    if(count!=sizeof...(A)) throw std::runtime_error("Kernel argument count mismatch");
    return invoke_impl(fn,args,std::index_sequence_for<A...>{});
}
}

// Generated exclusively by binding-type substitutions in the pinned launch
// wrapper source. Kernel source and launch calculations remain unchanged.
#include "kitchen_launch_wrappers.inc"

extern "C" int imagine_rms_metadata_begin() {
    return protect([] {
        if (rms_callback) throw std::runtime_error("RMS metadata observer already active");
        rms_temporary = {};
        rms_callback = at::addThreadLocalCallback(at::RecordFunctionCallback(rms_metadata_start, rms_metadata_end).needsOutputs(true));
    });
}
extern "C" int imagine_rms_metadata_end(RmsTemporaryMetadata* output) {
    return protect([&] {
        if (!rms_callback) throw std::runtime_error("RMS metadata observer is not active");
        at::removeCallback(rms_callback);
        rms_callback = 0;
        if (output) *output = rms_temporary;
        rms_temporary = {};
    });
}
extern "C" int imagine_host_registration(const void* input,int64_t pin,int64_t* success) {
    return protect([&] {
    const auto& t=*static_cast<const at::Tensor*>(input);
    *success=0;
    if (!t.device().is_cpu() || !t.is_contiguous() || t.data_ptr()==nullptr) return;
    if (pin && t.is_pinned()) return;
    const auto status=pin ? cudaHostRegister(t.data_ptr(),t.nbytes(),1) : cudaHostUnregister(t.data_ptr());
    *success=status==cudaSuccess;
    if (status!=cudaSuccess) {
        // Original discard_cuda_async_error; failure is a nonfatal pin refusal.
        try {auto a=at::ones({1},at::TensorOptions().device(at::kCUDA).dtype(at::kByte));auto b=at::ones_like(a);auto c=a+b;cudaDeviceSynchronize();} catch (...) {}
    }
    });
}
extern "C" const char* imagine_last_error() { return last_error.c_str(); }
extern "C" const char* imagine_last_metadata() { return last_metadata.c_str(); }
extern "C" int imagine_aten_boxed(const char* name,const char* overload,
    const void* const* args,size_t nargs,const char* const* keys,
    const void* const* values,size_t nkwargs,void** output) {
    try {
        auto op=c10::Dispatcher::singleton().findSchemaOrThrow(name,overload);
        std::vector<c10::IValue> stack;
        for(size_t i=0;i<nargs;++i) stack.push_back(*static_cast<const c10::IValue*>(args[i]));
        std::unordered_map<std::string,c10::IValue> kwargs;
        for(size_t i=0;i<nkwargs;++i) kwargs.emplace(keys[i],*static_cast<const c10::IValue*>(values[i]));
        const auto& schema=op.schema();
        auto sequence=[](c10::IValue& value,const c10::TypePtr& type) {
            auto expected=type;
            if(auto optional=expected->cast<c10::OptionalType>()) expected=optional->getElementType();
            if(auto list=expected->cast<c10::ListType>();list && (value.isList() || value.isTuple())) {
                c10::impl::GenericList result(list->getElementType());
                if(value.isTuple()) for(const auto& x:value.toTupleRef().elements()) result.push_back(x);
                else for(const auto& x:value.toListRef()) result.push_back(x);
                value=std::move(result);
            }
        };
        for(size_t i=0;i<stack.size() && i<schema.arguments().size();++i) sequence(stack[i],schema.arguments()[i].type());
        for(const auto& arg:schema.arguments()) if(auto it=kwargs.find(arg.name());it!=kwargs.end()) sequence(it->second,arg.type());
        schema.checkAndNormalizeInputs(stack,kwargs);
        op.callBoxed(&stack);
        if(stack.size()==1) *output=new c10::IValue(std::move(stack[0]));
        else *output=new c10::IValue(c10::ivalue::Tuple::create(std::move(stack)));
        last_error.clear();return 0;
    } catch(const c10::ValueError& e) {last_error=e.what_without_backtrace();return -2;}
      catch(const c10::IndexError& e) {last_error=e.what_without_backtrace();return -3;}
      catch(const c10::TypeError& e) {last_error=e.what_without_backtrace();return -4;}
      catch(const c10::NotImplementedError& e) {last_error=e.what_without_backtrace();return -5;}
      catch(const c10::Error& e) {last_error=e.what_without_backtrace();return -1;}
      catch(const std::exception& e) {last_error=e.what();return -1;}
}
extern "C" int imagine_cublaslt_available(int64_t* output) {
    return protect([&]{*output=comfy::CublasLtRuntime::instance().is_available();});
}
extern "C" int imagine_cuda_current_device(int64_t* output) {
    return protect([&]{ *output=c10::cuda::current_device(); });
}
extern "C" int imagine_cuda_device_count(int64_t* output) {
    return protect([&]{*output=c10::cuda::device_count();});
}
extern "C" int imagine_cuda_current_capability(int64_t* major,int64_t* minor) {
    return protect([&]{
        const auto* props=at::cuda::getDeviceProperties(c10::cuda::current_device());
        *major=props->major; *minor=props->minor;
    });
}
extern "C" int imagine_cuda_device_properties(int64_t device,CudaDeviceProperties* output) {
    return protect([&]{
        const auto* props=at::cuda::getDeviceProperties(device);
        *output={props->major,props->minor,static_cast<int64_t>(props->sharedMemPerBlock),
                 static_cast<int64_t>(props->sharedMemPerBlockOptin),props->multiProcessorCount,props->l2CacheSize};
    });
}
namespace {
#include "pytorch_error_message.inc"
template<class F> int aten_tensor_result(F&& fn,void** output) noexcept {
    try {
        *output=new at::Tensor(fn());last_error.clear();return 0;
    } catch(const c10::ValueError& e) {last_error=processErrorMsg(e.what_without_backtrace());return -2;}
      catch(const c10::IndexError& e) {last_error=processErrorMsg(e.what_without_backtrace());return -3;}
      catch(const c10::BufferError& e) {last_error=processErrorMsg(e.what_without_backtrace());return -4;}
      catch(const c10::SyntaxError& e) {last_error=processErrorMsg(e.what_without_backtrace());return -5;}
      catch(const c10::TypeError& e) {last_error=processErrorMsg(e.what_without_backtrace());return -7;}
      catch(const c10::NotImplementedError& e) {last_error=processErrorMsg(e.what_without_backtrace());return -8;}
      catch(const c10::OutOfMemoryError& e) {last_error=processErrorMsg(e.what_without_backtrace());return -9;}
      catch(const c10::AcceleratorError& e) {last_error=processErrorMsg(e.what_without_backtrace());return -10;}
      catch(const c10::Error& e) {last_error=processErrorMsg(e.what_without_backtrace());return -1;}
      catch(const std::exception& e) {last_error=processErrorMsg(e.what());return -1;}
      catch(...) {last_error="Unknown native exception";return -6;}
}
}
extern "C" int imagine_embedding(const void* weight,const void* indices,void** output) {
    return aten_tensor_result([&]{return at::embedding(*static_cast<const at::Tensor*>(weight),
        *static_cast<const at::Tensor*>(indices),-1,false,false);},output);
}
namespace {
std::array<bool,5> sdpa_flags() {
    auto& c=at::globalContext();
    return {c.userEnabledMathSDP(),c.userEnabledFlashSDP(),c.userEnabledMemEfficientSDP(),c.userEnabledCuDNNSDP(),c.userEnabledOverrideableSDP()};
}
void sdpa_set_flags(const std::array<bool,5>& flags) {
    auto& c=at::globalContext();c.setSDPUseMath(flags[0]);c.setSDPUseFlash(flags[1]);
    c.setSDPUseMemEfficient(flags[2]);c.setSDPUseCuDNN(flags[3]);c.setSDPUseOverrideable(flags[4]);
}
void sdpa_kernel(const std::vector<int64_t>& backends) {
    auto& c=at::globalContext();std::array<bool,5> flags{};
    auto priority=backends;
    for(auto b:backends) flags.at(b)=true;
    sdpa_set_flags(flags);
    for(auto b:c.sDPPriorityOrder()) {
        auto value=static_cast<int64_t>(b);
        if(std::find(priority.begin(),priority.end(),value)==priority.end()) priority.push_back(value);
    }
    c.setSDPPriorityOrder(priority);
}
struct SdpaKernelContext {
    std::vector<int64_t> previous;
    SdpaKernelContext() {
        auto flags=sdpa_flags();
        for(auto b:at::globalContext().sDPPriorityOrder()) if(flags.at(static_cast<size_t>(b))) previous.push_back(static_cast<int64_t>(b));
        // Original SDPA_BACKEND_PRIORITY: flash, cuDNN, efficient, math.
        sdpa_kernel({1,3,2,0});
    }
    ~SdpaKernelContext() { sdpa_kernel(previous); }
};
void repeat_gqa(const at::Tensor& q,at::Tensor& k,at::Tensor& v) {
    const auto qh=q.size(-3),kh=k.size(-3),vh=v.size(-3);
    TORCH_CHECK_VALUE(kh==vh,"Key/value head count mismatch for GQA: ",kh," != ",vh);
    if(qh==kh) return;
    TORCH_CHECK_VALUE(kh!=0 && qh%kh==0,"Query heads must be divisible by key/value heads for GQA: ",qh," vs ",kh);
    const auto repeats=qh/kh;
    if(repeats>1) { k=k.repeat_interleave(repeats,-3);v=v.repeat_interleave(repeats,-3); }
}
}
extern "C" int imagine_sdpa_state(int64_t* state,int64_t write) {
    return protect([&] {
        auto& c=at::globalContext();
        if(write) {
            std::array<bool,5> flags{};for(size_t i=0;i<5;++i)flags[i]=state[i]!=0;
            sdpa_set_flags(flags);c.setSDPPriorityOrder(std::vector<int64_t>(state+5,state+10));
        } else {
            auto flags=sdpa_flags();auto priority=c.sDPPriorityOrder();
            for(size_t i=0;i<5;++i) { state[i]=flags[i];state[i+5]=static_cast<int64_t>(priority[i]); }
        }
    });
}
extern "C" int imagine_cuda_memory(int64_t index,uint64_t* result) {
    return protect([&] {
        // Python's CUDA startup/mem_get_info performs _lazy_init first. Device
        // properties alone do not initialize the CUDACachingAllocator in C++.
        at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
        c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(index));
        auto stats=c10::cuda::CUDACachingAllocator::getDeviceStats(index);
        size_t free=0,total=0;C10_CUDA_CHECK(cudaMemGetInfo(&free,&total));
        result[0]=free;result[1]=total;
        result[2]=stats.active_bytes[0].current;result[3]=stats.reserved_bytes[0].current;
    });
}
extern "C" int imagine_comfy_math_state(int64_t* state,int64_t write) {
    return protect([&] {
        auto& c=at::globalContext();
        if(write) {
            c.setAllowFP16BF16ReductionMathSDP(state[0]!=0);
            c.setBenchmarkCuDNN(state[1]!=0);c.setAllowFP16AccumulationCuBLAS(state[2]!=0);
        } else {
            state[0]=c.allowFP16BF16ReductionMathSDP();state[1]=c.benchmarkCuDNN();
            state[2]=c.allowFP16AccumulationCuBLAS();
        }
    });
}
extern "C" int imagine_cuda_soft_empty_cache(int64_t index) {
    return protect([&] {
        at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
        c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(index));
        C10_CUDA_CHECK(cudaDeviceSynchronize());
        c10::cuda::CUDACachingAllocator::emptyCache();torch::CudaIPCCollect();
    });
}
extern "C" int imagine_comfy_sdpa(const void* query,const void* key,const void* value,const void* mask,
                                  int64_t has_scale,double scale,int64_t gqa,int64_t priority_wrapper,void** output) {
    return aten_tensor_result([&] {
        const auto& q=*static_cast<const at::Tensor*>(query);
        auto k=*static_cast<const at::Tensor*>(key),v=*static_cast<const at::Tensor*>(value);
        const auto m=mask ? std::optional<at::Tensor>(*static_cast<const at::Tensor*>(mask)) : std::nullopt;
        const auto s=has_scale ? std::optional<double>(scale) : std::nullopt;
        bool grouped=gqa!=0;
        if(!priority_wrapper) {
            if(grouped && m) { repeat_gqa(q,k,v);grouped=false; }
            return at::scaled_dot_product_attention(q,k,v,m,0.,false,s,grouped);
        }
        if(q.numel()<1024*128) return at::scaled_dot_product_attention(q,k,v,m,0.,false,s,grouped);
        // Frozen platform is NVIDIA: AMD-only contiguous/L2 and pre-context GQA
        // branches are deliberately outside this Linux/SM89 binding.
        SdpaKernelContext context;
        if(grouped && m && q.size(-3)!=k.size(-3)) {
            sdp::sdp_params params{q,k,v,m,0.,false,true};
            if(!(sdp::can_use_flash_attention(params,false)||sdp::can_use_cudnn_attention(params,false)||sdp::can_use_mem_efficient_attention(params,false))) {
                repeat_gqa(q,k,v);grouped=false;
            }
        }
        return at::scaled_dot_product_attention(q,k,v,m,0.,false,s,grouped);
    },output);
}
extern "C" int imagine_cat(const void* const* inputs,size_t count,int64_t dim,void** output) {
    return aten_tensor_result([&]{
        std::vector<at::Tensor> tensors;tensors.reserve(count);
        for(size_t i=0;i<count;++i) tensors.push_back(*static_cast<const at::Tensor*>(inputs[i]));
        return at::cat(tensors,dim);
    },output);
}
extern "C" int imagine_amax_last(const void* input,void** output) {
    return aten_tensor_result([&]{return at::amax(*static_cast<const at::Tensor*>(input),{-1},true);},output);
}
extern "C" int imagine_fp8_scaled_mm(const void* a,const void* b,const void* sa,const void* sb,const void* bias,int64_t kind,void** output) {
    return aten_tensor_result([&] {
        const auto& scale_a=*static_cast<const at::Tensor*>(sa);
        const auto& scale_b=*static_cast<const at::Tensor*>(sb);
        const std::vector<int64_t> recipe{static_cast<int64_t>(at::blas::ScalingType::TensorWise)};
        const std::vector<int64_t> swizzle{static_cast<int64_t>(at::blas::SwizzleType::NO_SWIZZLE)};
        const auto optional_bias=bias ? std::optional<at::Tensor>(*static_cast<const at::Tensor*>(bias)) : std::nullopt;
        const auto optional_kind=kind<0 ? std::nullopt : std::optional<at::ScalarType>(static_cast<at::ScalarType>(kind));
        const auto input=static_cast<const at::Tensor*>(a)->contiguous();
        return at::_scaled_mm_v2(input,*static_cast<const at::Tensor*>(b),
            {scale_a},recipe,swizzle,{scale_b},recipe,swizzle,optional_bias,optional_kind,{},false);
    },output);
}

extern "C" int imagine_kitchen_call(const char* name,const KitchenArg* args,size_t n,int64_t* result) {
    auto invoke_original = [&] {
        if(!name || !result || (!args && n)) throw std::runtime_error("Null bridge argument");
        // Match the caller's current stream/device. Upstream wrapper functions
        // retain their own argument checks and launch contracts.
        last_metadata.clear();
        std::optional<c10::cuda::CUDAGuard> guard;
        for(size_t i=0;i<n;i++) {
            if(args[i].kind==1 && args[i].tensor) {
                const auto& tensor=*static_cast<const at::Tensor*>(args[i].tensor);
                guard.emplace(tensor.device());
                break;
            }
        }
        #include "kitchen_dispatch.inc"
        throw std::runtime_error("Unknown Kitchen kernel");
    };
    // Nanobind translates these standard exceptions to ValueError/IndexError.
    // Preserve that class across the standalone ABI without editing wrappers.
    try { invoke_original(); last_error.clear(); return 0; }
    catch(const std::invalid_argument& e) { last_error=e.what(); return -20; }
    catch(const std::out_of_range& e) { last_error=e.what(); return -21; }
    catch(const std::exception& e) { last_error=e.what(); return -1; }
    catch(...) { last_error="Unknown native exception"; return -1; }
}
extern "C" int imagine_parse_device(const char* text,int64_t* type,int64_t* index) {
    try { auto d=c10::Device(text); *type=static_cast<int64_t>(d.type()); *index=d.index();last_error.clear();return 0; }
    catch(const c10::Error& e) {last_error=processErrorMsg(e.what_without_backtrace());return -1;}
    catch(const std::exception& e) {last_error=e.what();return -1;}
}
extern "C" int imagine_tensor_empty_like_device_spec(const void* input,int64_t type,int64_t index,void** output) {
    return aten_tensor_result([&]{ const auto& t=*static_cast<const at::Tensor*>(input); return at::empty_like(t,t.options().device(c10::Device(static_cast<c10::DeviceType>(type),static_cast<c10::DeviceIndex>(index)))); },output);
}
extern "C" int imagine_tensor_to_device_spec(const void* input,int64_t type,int64_t index,void** output) {
    return aten_tensor_result([&]{ return static_cast<const at::Tensor*>(input)->to(c10::Device(static_cast<c10::DeviceType>(type),static_cast<c10::DeviceIndex>(index))); },output);
}
extern "C" int imagine_default_dtype(int64_t* output) {
    return protect([&]{ *output=static_cast<int64_t>(c10::get_default_dtype_as_scalartype()); });
}
extern "C" int imagine_set_default_dtype(int64_t kind) {
    return protect([&]{ c10::set_default_dtype(c10::scalarTypeToTypeMeta(static_cast<at::ScalarType>(kind))); });
}
extern "C" int imagine_current_stream(int64_t device,uint64_t* output) {
    return protect([&]{ *output=reinterpret_cast<uint64_t>(c10::cuda::getCurrentCUDAStream(device).stream()); });
}
extern "C" int imagine_stream_guard_new(int64_t device,void** output) {
    return protect([&]{*output=new c10::cuda::CUDAStreamGuard(c10::cuda::getStreamFromPool(false,device));});
}
extern "C" void imagine_stream_guard_free(void* guard) {delete static_cast<c10::cuda::CUDAStreamGuard*>(guard);}
extern "C" int imagine_event_record(int64_t device,void** output) {
    return protect([&]{
        auto event=std::make_unique<c10::cuda::CUDAEvent>();
        event->record(c10::cuda::getCurrentCUDAStream(device));
        *output=event.release();
    });
}
extern "C" void imagine_event_free(void* event) {delete static_cast<c10::cuda::CUDAEvent*>(event);}
extern "C" int imagine_event_wait(void* event,int64_t device) {
    return protect([&]{static_cast<c10::cuda::CUDAEvent*>(event)->block(c10::cuda::getCurrentCUDAStream(device));});
}
extern "C" int imagine_event_synchronize(void* event) {
    return protect([&]{static_cast<c10::cuda::CUDAEvent*>(event)->synchronize();});
}
extern "C" int imagine_timing_event_new(void** output) {
    return protect([&]{*output=new c10::cuda::CUDAEvent(cudaEventDefault);});
}
extern "C" int imagine_event_record_existing(void* event,int64_t device) {
    return protect([&]{static_cast<c10::cuda::CUDAEvent*>(event)->record(c10::cuda::getCurrentCUDAStream(device));});
}
extern "C" int imagine_event_elapsed(void* start,void* end,double* output) {
    return protect([&]{*output=static_cast<c10::cuda::CUDAEvent*>(start)->elapsed_time(*static_cast<c10::cuda::CUDAEvent*>(end));});
}
extern "C" int imagine_device_synchronize(int64_t device) {
    return protect([&]{c10::cuda::CUDAGuard guard(device);c10::cuda::device_synchronize();});
}
extern "C" int imagine_tensor_clone(const void* input,void** output) {
    return protect([&]{ *output=new at::Tensor(static_cast<const at::Tensor*>(input)->clone(at::MemoryFormat::Preserve)); });
}
extern "C" int imagine_tensor_empty_scalar(const int64_t* shape,size_t rank,int64_t kind,void** output) {
    return protect([&] { *output=new at::Tensor(at::empty(at::IntArrayRef(shape,rank),
        at::TensorOptions().device(at::kCPU).dtype(static_cast<at::ScalarType>(kind)))); });
}
// Buffer ownership is transferred on every call, including error returns.
// ATen views retain the original Storage's deleter and its mmap owner.
extern "C" int imagine_tensor_owned_buffer(void* data,const int64_t* shape,size_t rank,int64_t kind,
                                           void* owner,void(*release)(void*),void** output) {
    auto buffer=at::for_blob(data,at::IntArrayRef(shape,rank));
    buffer.context(owner,release);
    return protect([&] {
        // ATen holds a plain Rust context/deleter directly, so Storage views
        // remain valid after dlclose of this bridge. No DSO pin or global map.
        *output=new at::Tensor(buffer.options(at::TensorOptions().device(at::kCPU)
            .dtype(static_cast<at::ScalarType>(kind))).make_tensor());
    });
}
extern "C" int imagine_tensor_buffer_context(const void* tensor,void(*release)(void*),void** output) {
    return protect([&] {
        const auto& data=static_cast<const at::Tensor*>(tensor)->storage().data_ptr();
        *output=data.get_deleter()==release ? data.get_context() : nullptr;
    });
}
extern "C" int imagine_tensor_empty_like(const void* input,int64_t device,void** output) {
    return protect([&] {
        const auto& tensor=*static_cast<const at::Tensor*>(input);
        const auto target=device<0 ? at::Device(at::kCPU) : at::Device(at::kCUDA,device);
        *output=new at::Tensor(at::empty_like(tensor,tensor.options().device(target).requires_grad(false),at::MemoryFormat::Preserve));
    });
}
extern "C" int imagine_tensor_copy(void* destination,const void* source,int64_t non_blocking) {
    return protect([&]{ static_cast<at::Tensor*>(destination)->copy_(*static_cast<const at::Tensor*>(source),non_blocking!=0); });
}
extern "C" int imagine_tensor_empty_like_dtype(const void* input,int64_t kind,void** output) {
    return protect([&] {
        const auto& tensor=*static_cast<const at::Tensor*>(input);
        *output=new at::Tensor(at::empty_like(tensor,tensor.options().dtype(static_cast<at::ScalarType>(kind)).requires_grad(false),at::MemoryFormat::Preserve));
    });
}
extern "C" uint64_t imagine_tensor_identity(const void* input) {
    return reinterpret_cast<uint64_t>(static_cast<const at::Tensor*>(input)->unsafeGetTensorImpl());
}
extern "C" int imagine_tensor_storage(const void* input,TensorStorageInfo* output) {
    return protect([&]{
        const auto& t=*static_cast<const at::Tensor*>(input);
        output->base=reinterpret_cast<uintptr_t>(t.storage().data_ptr().get());
        output->offset=t.storage_offset();
        output->element_bytes=t.element_size();
    });
}
extern "C" int imagine_tensor_type(const void* input,TensorTypeInfo* output) {
    return protect([&] {
        const auto& tensor=*static_cast<const at::Tensor*>(input);
        auto* managed=at::toDLPack(tensor);
        const auto dtype=managed->dl_tensor.dtype;
        managed->deleter(managed);
        *output={static_cast<int64_t>(tensor.scalar_type()),dtype.code,dtype.bits,dtype.lanes,static_cast<int64_t>(tensor.element_size())};
    });
}
extern "C" int imagine_tensor_shell_view(const void* input,int64_t kind,void** output) {
    return protect([&] {
        static_assert(static_cast<int>(at::ScalarType::Float8_e8m0fnu)==44);
        static_assert(static_cast<int>(at::ScalarType::Float4_e2m1fn_x2)==45);
        if(kind!=44 && kind!=45) throw std::runtime_error("Expected E8M0 or packed E2M1 dtype");
        *output=new at::Tensor(static_cast<const at::Tensor*>(input)->view(static_cast<at::ScalarType>(kind)));
    });
}
extern "C" int imagine_generator_new(int64_t device,uint64_t seed,void** output) {
    return protect([&] {
        at::Generator generator=device<0 ? at::make_generator<at::CPUGeneratorImpl>() : at::cuda::detail::createCUDAGenerator(device);
        generator.set_current_seed(seed);
        *output=new at::Generator(std::move(generator));
    });
}
extern "C" void imagine_generator_free(void* handle) { delete static_cast<at::Generator*>(handle); }
extern "C" int imagine_generator_state(void* handle,void** output) {
    return protect([&]{ *output=new at::Tensor(static_cast<at::Generator*>(handle)->get_state()); });
}
extern "C" int imagine_generator_set_state(void* handle,const void* state) {
    return protect([&]{ static_cast<at::Generator*>(handle)->set_state(*static_cast<const at::Tensor*>(state)); });
}
extern "C" int imagine_generator_randn(void* handle,const int64_t* dims,size_t count,int64_t kind,void** output) {
    return protect([&] {
        auto& generator=*static_cast<at::Generator*>(handle);
        *output=new at::Tensor(at::randn(at::IntArrayRef(dims,count),generator,at::TensorOptions().device(generator.device()).dtype(static_cast<at::ScalarType>(kind))));
    });
}
extern "C" int imagine_generator_randbytes(void* handle,const int64_t* dims,size_t count,void** output) {
    return protect([&] {
        auto& generator=*static_cast<at::Generator*>(handle);
        *output=new at::Tensor(at::randint(0,256,at::IntArrayRef(dims,count),generator,at::TensorOptions().device(generator.device()).dtype(at::kByte)));
    });
}
extern "C" int imagine_generator_randint(void* handle,int64_t high,const int64_t* dims,size_t count,void** output) {
    return protect([&]{
        auto& generator=*static_cast<at::Generator*>(handle);
        *output=new at::Tensor(at::randint(0,high,at::IntArrayRef(dims,count),generator,at::TensorOptions().device(generator.device()).dtype(at::kLong)));
    });
}
extern "C" int imagine_generator_rand(void* handle,const int64_t* dims,size_t count,int64_t kind,void** output) {
    return protect([&] {
        auto& generator=*static_cast<at::Generator*>(handle);
        *output=new at::Tensor(at::rand(at::IntArrayRef(dims,count),generator,at::TensorOptions().device(generator.device()).dtype(static_cast<at::ScalarType>(kind))));
    });
}

extern "C" int imagine_ivalue_uint(uint64_t value,void** out) {
    return protect([&]{ *out = new c10::IValue(at::Scalar(value)); });
}
extern "C" int imagine_ivalue_get_uint(const void* raw,uint64_t* out) {
    const auto& value=*static_cast<const c10::IValue*>(raw);
    if(!value.isUnsigned()) return 0;
    *out=value.toUInt(); return 1;
}

extern "C" int imagine_tensor_item_double(const void* raw,double* out) {
    try { *out=static_cast<const at::Tensor*>(raw)->item<double>(); last_error.clear(); return 0; }
    catch(const c10::ValueError& error) { last_error=error.what_without_backtrace(); return -2; }
    catch(const c10::IndexError& error) { last_error=error.what_without_backtrace(); return -3; }
    catch(const c10::TypeError& error) { last_error=error.what_without_backtrace(); return -4; }
    catch(const c10::NotImplementedError& error) { last_error=error.what_without_backtrace(); return -5; }
    catch(const c10::OutOfMemoryError& error) { last_error=error.what_without_backtrace(); return -9; }
    catch(const c10::AcceleratorError& error) { last_error=error.what_without_backtrace(); return -10; }
    catch(const c10::Error& error) { last_error=error.what_without_backtrace(); return -1; }
    catch(const std::exception& error) { last_error=error.what(); return -1; }
    catch(...) { last_error="Unknown native exception"; return -1; }
}

extern "C" int imagine_tensor_grad_name(const void* raw,const char** out) {
    try {
        try {
            const auto node=static_cast<const at::Tensor*>(raw)->grad_fn();
            last_metadata=node ? node->name() : "";
        } catch(const c10::Error&) { last_metadata="Invalid"; }
        *out=last_metadata.c_str();last_error.clear();return 0;
    } catch(const std::exception& error) { last_error=error.what();return -1; }
      catch(...) { last_error="Unknown native exception";return -1; }
}
