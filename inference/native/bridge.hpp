#pragma once
#include <cstddef>
#include <cstdint>

// kind: 0=None, 1=borrowed at::Tensor*, 2=int64, 3=double, 4=bool.
struct KitchenArg { int64_t kind; void* tensor; int64_t integer; double real; };
struct TensorTypeInfo {int64_t scalar_type;int64_t dlpack_code;int64_t bits;int64_t lanes;int64_t element_bytes;};
struct TensorStorageInfo {uint64_t base;int64_t offset;int64_t element_bytes;};
struct CudaDeviceProperties { int64_t major; int64_t minor; int64_t shared_memory_per_block; int64_t shared_memory_per_block_optin; int64_t multiprocessor_count; int64_t l2_cache_size; };
struct RmsTemporaryMetadata { uint64_t start; uint64_t bytes; int64_t count; };
#pragma GCC visibility push(default)
extern "C" {
int imagine_beta_ppf(double,double,double,double*,const char**,const char**);
const char* imagine_last_error();
int imagine_last_accelerator_code();
const char* imagine_last_metadata();
int imagine_tensor_owned_buffer(void*,const int64_t*,size_t,int64_t,void*,void(*)(void*),void**);
int imagine_tensor_empty_scalar(const int64_t*,size_t,int64_t,void**);
int imagine_tensor_buffer_context(const void*,void(*)(void*),void**);
int imagine_comfy_sdpa(const void*,const void*,const void*,const void*,int64_t,double,int64_t,int64_t,void**);
int imagine_sdpa_state(int64_t*,int64_t);
int imagine_cuda_memory(int64_t,uint64_t*);
int imagine_host_registration(const void*,int64_t,int64_t*);
int imagine_host_registration_raw(uint64_t,uint64_t,int64_t,int64_t*);
int imagine_tensor_video_slice_assign(void*,int64_t,int64_t,const void*);
int imagine_cuda_soft_empty_cache(int64_t);
int imagine_comfy_math_state(int64_t*,int64_t);
int imagine_kitchen_call(const char*,const KitchenArg*,size_t,int64_t*);
// Borrowed c10::IValue handles from the pinned torch-sys ABI; owned result.
int imagine_aten_boxed(const char*,const char*,const void* const*,size_t,const char* const*,const void* const*,size_t,void**);
int imagine_cublaslt_available(int64_t*);
int imagine_cuda_current_device(int64_t*);
int imagine_cuda_device_count(int64_t*);
int imagine_cuda_current_capability(int64_t*,int64_t*);
int imagine_cuda_device_properties(int64_t,CudaDeviceProperties*);
int imagine_rms_metadata_begin();
int imagine_rms_metadata_end(RmsTemporaryMetadata*);
// -1: RuntimeError/TypeError family (reference catches). Other negative codes
// preserve ValueError/IndexError/BufferError/SyntaxError instead of falling back.
int imagine_cat(const void* const*,size_t,int64_t,void**);
int imagine_amax_last(const void*,void**);
int imagine_embedding(const void*,const void*,void**);
int imagine_fp8_scaled_mm(const void*,const void*,const void*,const void*,const void*,int64_t,void**);
int imagine_parse_device(const char*,int64_t*,int64_t*);
int imagine_tensor_empty_like_device_spec(const void*,int64_t,int64_t,void**);
int imagine_tensor_to_device_spec(const void*,int64_t,int64_t,void**);
int imagine_default_dtype(int64_t*);
int imagine_set_default_dtype(int64_t);
int imagine_current_stream(int64_t,uint64_t*);
int imagine_stream_guard_new(int64_t,void**);
void imagine_stream_guard_free(void*);
int imagine_event_record(int64_t,void**);
void imagine_event_free(void*);
int imagine_event_wait(void*,int64_t);
int imagine_event_synchronize(void*);
int imagine_timing_event_new(void**);
int imagine_event_record_existing(void*,int64_t);
int imagine_event_elapsed(void*,void*,double*);
int imagine_device_synchronize(int64_t);
int imagine_tensor_clone(const void*,void**);
int imagine_tensor_empty_like(const void*,int64_t,void**);
int imagine_tensor_empty_like_dtype(const void*,int64_t,void**);
int imagine_tensor_copy(void*,const void*,int64_t);
int imagine_tensor_type(const void*,TensorTypeInfo*);
uint64_t imagine_tensor_identity(const void*);
int imagine_tensor_storage(const void*,TensorStorageInfo*);
int imagine_tensor_shell_view(const void*,int64_t,void**);
int imagine_generator_new(int64_t,uint64_t,void**);
void imagine_generator_free(void*);
int imagine_generator_state(void*,void**);
int imagine_generator_set_state(void*,const void*);
int imagine_generator_randn(void*,const int64_t*,size_t,int64_t,void**);
int imagine_generator_rand(void*,const int64_t*,size_t,int64_t,void**);
int imagine_generator_randbytes(void*,const int64_t*,size_t,void**);
int imagine_generator_randint(void*,int64_t,const int64_t*,size_t,void**);
int imagine_tensor_item_double(const void*,double*);
int imagine_tensor_grad_name(const void*,const char**);
int imagine_ivalue_uint(uint64_t,void**);
int imagine_ivalue_get_uint(const void*,uint64_t*);
}
#pragma GCC visibility pop
