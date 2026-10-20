// Owned, thread-local LibTorch InferenceMode context; no Python dependency.
#include <torch/torch.h>
#include <c10/core/InferenceMode.h>
#include <ATen/TensorIndexing.h>
#include <ATen/functorch/TensorWrapper.h>
#include <caffe2/serialize/inline_container.h>
#include "torch_api.h"
#if __has_include(<cuda_runtime_api.h>)
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#endif

void* at_imagine_inference_mode_new(int enabled) {
  PROTECT(return new c10::InferenceMode(enabled != 0);)
  return nullptr;
}
void at_imagine_inference_mode_free(void* value) {
  delete static_cast<c10::InferenceMode*>(value);
}
using ImagineArchive = caffe2::serialize::PyTorchStreamReader;
int at_imagine_storage_resizable(tensor value) {
  PROTECT(return value->storage().resizable();)
  return 0;
}
using ImagineArchiveRecord = std::tuple<at::DataPtr, size_t>;
using ImagineArchiveNames = std::vector<std::string>;
void* at_imagine_archive_open(const uint8_t* path, size_t size) {
  PROTECT(return new ImagineArchive(std::string(reinterpret_cast<const char*>(path), size));)
  return nullptr;
}
void at_imagine_archive_free(void* reader) { delete static_cast<ImagineArchive*>(reader); }
int at_imagine_archive_has(void* reader, const uint8_t* key, size_t size) {
  PROTECT(return static_cast<ImagineArchive*>(reader)->hasRecord(std::string(reinterpret_cast<const char*>(key), size));)
  return 0;
}
void* at_imagine_archive_record(void* reader, const uint8_t* key, size_t size) {
  PROTECT(return new ImagineArchiveRecord(static_cast<ImagineArchive*>(reader)->getRecord(std::string(reinterpret_cast<const char*>(key), size)));)
  return nullptr;
}
const uint8_t* at_imagine_archive_data(void* record, size_t* size) {
  const auto& value = *static_cast<ImagineArchiveRecord*>(record);
  *size = std::get<1>(value);
  return static_cast<const uint8_t*>(std::get<0>(value).get());
}
void at_imagine_archive_record_free(void* record) { delete static_cast<ImagineArchiveRecord*>(record); }
void* at_imagine_archive_records(void* reader, size_t* size) {
  PROTECT(
    auto values = std::make_unique<ImagineArchiveNames>(static_cast<ImagineArchive*>(reader)->getAllRecords());
    *size = values->size();
    return values.release();
  )
  return nullptr;
}
const uint8_t* at_imagine_archive_record_name(void* names, size_t index, size_t* size) {
  const auto& name = static_cast<ImagineArchiveNames*>(names)->at(index);
  *size = name.size();
  return reinterpret_cast<const uint8_t*>(name.data());
}
void at_imagine_archive_records_free(void* names) { delete static_cast<ImagineArchiveNames*>(names); }
uint64_t at_imagine_archive_offset(void* reader, const uint8_t* key, size_t size) {
  PROTECT(return static_cast<ImagineArchive*>(reader)->getRecordOffset(std::string(reinterpret_cast<const char*>(key), size));)
  return 0;
}
tensor at_imagine_archive_storage(void* reader, const uint8_t* key, size_t size, size_t numel, int dtype) {
  PROTECT(
    return ([&]() {
    auto [data, bytes] = static_cast<ImagineArchive*>(reader)->getRecord(std::string(reinterpret_cast<const char*>(key), size));
    const auto scalar_type = at::ScalarType(dtype);
    TORCH_CHECK(bytes == numel * elementSize(scalar_type), "record size (", bytes,
      " bytes) does not match expected size (", numel * elementSize(scalar_type),
      " bytes = ", numel, " elements * ", elementSize(scalar_type), " bytes/element) for dtype ", scalar_type);
    c10::Storage storage(c10::Storage::use_byte_size_t(), numel * elementSize(scalar_type),
      std::move(data), /*allocator=*/nullptr, /*resizable=*/false);
    auto ptr = c10::make_intrusive<at::TensorImpl, at::UndefinedTensorImpl>(
      std::move(storage), at::DispatchKeySet(), at::CPU(scalar_type).typeMeta());
    return new at::Tensor(std::move(ptr));
    })();
  )
  return nullptr;
}
void at_imagine_storage_info(tensor value, uintptr_t* identity, uint64_t* bytes) {
  PROTECT(
    const auto& storage = value->storage();
    *identity = reinterpret_cast<uintptr_t>(storage.unsafeGetStorageImpl());
    *bytes = storage.nbytes();
  )
}
void at_imagine_tensor_item(tensor value, int* kind, int64_t* integer,
                            uint64_t* unsigned_integer, double* real, double* imaginary) {
  PROTECT(
    if (value->is_meta() && value->numel() == 1) {
      TORCH_CHECK(false, "Tensor.item() cannot be called on meta tensors");
    }
    const auto scalar = value->item();
    if (scalar.isBoolean()) { *kind = 0; *integer = scalar.toBool(); }
    else if (scalar.isIntegral(false)) {
      if (scalar.isUnsigned()) { *kind = 2; *unsigned_integer = scalar.toUInt64(); }
      else { *kind = 1; *integer = scalar.toLong(); }
    }
    else if (scalar.isFloatingPoint()) { *kind = 3; *real = scalar.toDouble(); }
    else if (scalar.isComplex()) {
      const auto number = scalar.toComplexDouble();
      *kind = 4; *real = number.real(); *imaginary = number.imag();
    }
    else { TORCH_CHECK(false, "Unsupported native item scalar category"); }
  )
}
int64_t at_imagine_shape_unsigned_extent(uint64_t value, int64_t position) {
  PROTECT(
    try {
      TORCH_CHECK_VALUE(value <= uint64_t(INT64_MAX), "Overflow when unpacking long long");
      return int64_t(value);
    } catch (const c10::ValueError& error) {
      // Original throw_intlist_exception includes e.what(), including its stack.
      TORCH_CHECK_TYPE(false, "zeros(): argument 'size' failed to unpack the object at pos ",
        position, " with error \"", error.what(), "\"");
    }
  )
  return 0;
}
int64_t at_imagine_shape_tensor_extent(tensor value) {
  PROTECT(
    const auto scalar = value->item();
    TORCH_CHECK(scalar.isIntegral(/*include bool*/ false));
    return scalar.toSymInt().expect_int();
  )
  return 0;
}
tensor at_imagine_rebuild_storage_tensor(tensor source, int dtype, int64_t offset,
    const int64_t* sizes, const int64_t* strides, int64_t dimensions, int64_t stride_dimensions) {
  PROTECT(
    auto value = at::empty({0}, source->options().dtype(at::ScalarType(dtype)));
    value.set_(source->storage(), offset, at::IntArrayRef(sizes, dimensions),
      at::IntArrayRef(strides, stride_dimensions));
    return new torch::Tensor(value);
  )
  return nullptr;
}
tensor at_imagine_rebuild_empty(tensor storage, int dtype, int requires_grad) {
  PROTECT(return new at::Tensor(torch::empty({0}, storage->options().dtype(at::ScalarType(dtype)).requires_grad(requires_grad != 0)));)
  return nullptr;
}
void at_imagine_rebuild_set_storage(tensor value, tensor storage, int64_t offset,
    const int64_t* sizes, const int64_t* strides, int64_t dimensions, int64_t stride_dimensions) {
  PROTECT(value->set_(storage->storage(), offset, at::IntArrayRef(sizes, dimensions), at::IntArrayRef(strides, stride_dimensions));)
}
void at_imagine_rebuild_requires_grad(tensor value, int requires_grad) {
  PROTECT(
    // THPVariable_set_requires_grad performs this check before the native setter.
    TORCH_CHECK(!requires_grad || torch::autograd::isDifferentiableType(value->scalar_type()),
      "only Tensors of floating point and complex dtype can require gradients");
    value->set_requires_grad(requires_grad != 0);
  )
}
void at_imagine_rebuild_metadata_flags(tensor value, int conjugate_present, int negative_present) {
  PROTECT(
    // CPU/CUDA setTensorMetadata checks key presence, ignoring mapped bools.
    if (conjugate_present) value->_set_conj(true);
    if (negative_present) value->_set_neg(true);
  )
}
static const at::Tensor& imagine_unwrap(const at::Tensor& value) {
  if (auto* wrapper = at::functorch::maybeGetTensorWrapper(value)) {
    return imagine_unwrap(wrapper->value());
  }
  return value;
}
tensor at_imagine_tolist_data(tensor value) {
  PROTECT(
    TORCH_CHECK(!value->unsafeGetTensorImpl()->is_python_dispatch(),
      "[imagine-binding-gap] tolist source tensor subclass");
    at::Tensor data = value->resolve_conj().resolve_neg();
    data = imagine_unwrap(data);
    if (!data.device().is_cpu()) {
      data = data.toBackend(at::Backend::CPU);
      data = imagine_unwrap(data);
    }
    TORCH_CHECK(value->numel() == 0 || data.const_data_ptr(),
      "tolist() shouldn't be called on a tensor with unallocated storage");
    // The original obtains this pointer even when numel==0.
    (void)data.const_data_ptr();
    return new torch::Tensor(data);
  )
  return nullptr;
}
// Original python_scalars.h load_scalar conversions, without PyObject creation.
static at::Scalar imagine_load_scalar(const char* data, at::ScalarType dtype) {
  switch (dtype) {
#define IMAGINE_SIGNED(label, type) case at::label: return at::Scalar(int64_t(*reinterpret_cast<const type*>(data)));
    IMAGINE_SIGNED(kByte, uint8_t)
    IMAGINE_SIGNED(kChar, int8_t)
    IMAGINE_SIGNED(kShort, int16_t)
    IMAGINE_SIGNED(kInt, int32_t)
    IMAGINE_SIGNED(kLong, int64_t)
#undef IMAGINE_SIGNED
    case at::kUInt16: return at::Scalar(uint64_t(*reinterpret_cast<const uint16_t*>(data)));
    case at::kUInt32: return at::Scalar(uint64_t(*reinterpret_cast<const uint32_t*>(data)));
    case at::kUInt64: return at::Scalar(*reinterpret_cast<const uint64_t*>(data));
    case at::kBool: return at::Scalar(static_cast<bool>(*reinterpret_cast<const uint8_t*>(data)));
#define IMAGINE_REAL(label, type) case at::label: return at::Scalar(at::convert<double, type>(*reinterpret_cast<const type*>(data)));
    IMAGINE_REAL(kHalf, at::Half)
    IMAGINE_REAL(kFloat, float)
    IMAGINE_REAL(kDouble, double)
    IMAGINE_REAL(kBFloat16, at::BFloat16)
    IMAGINE_REAL(kFloat8_e5m2, at::Float8_e5m2)
    IMAGINE_REAL(kFloat8_e4m3fn, at::Float8_e4m3fn)
    IMAGINE_REAL(kFloat8_e5m2fnuz, at::Float8_e5m2fnuz)
    IMAGINE_REAL(kFloat8_e4m3fnuz, at::Float8_e4m3fnuz)
    IMAGINE_REAL(kFloat8_e8m0fnu, at::Float8_e8m0fnu)
#undef IMAGINE_REAL
#define IMAGINE_COMPLEX(label, type) case at::label: { const auto value = *reinterpret_cast<const c10::complex<type>*>(data); return at::Scalar(c10::complex<double>(value.real(), value.imag())); }
    IMAGINE_COMPLEX(kComplexHalf, at::Half)
    IMAGINE_COMPLEX(kComplexFloat, float)
    IMAGINE_COMPLEX(kComplexDouble, double)
#undef IMAGINE_COMPLEX
    default: TORCH_CHECK(false, "load_scalar: invalid type");
  }
}
void at_imagine_tolist_scalar(tensor value, int64_t offset, int* kind, int64_t* integer,
                              uint64_t* unsigned_integer, double* real, double* imaginary) {
  PROTECT(
    const auto* data = static_cast<const char*>(value->const_data_ptr());
    const auto scalar = imagine_load_scalar(data + offset * value->dtype().itemsize(), value->scalar_type());
    if (scalar.isBoolean()) { *kind = 0; *integer = scalar.toBool(); }
    else if (scalar.isIntegral(false)) {
      if (scalar.isUnsigned()) { *kind = 2; *unsigned_integer = scalar.toUInt64(); }
      else { *kind = 1; *integer = scalar.toLong(); }
    }
    else if (scalar.isComplex()) { const auto value = scalar.toComplexDouble(); *kind = 4; *real = value.real(); *imaginary = value.imag(); }
    else { *kind = 3; *real = scalar.toDouble(); }
  )
}
using ImagineWeakStorage = c10::weak_intrusive_ptr<c10::StorageImpl>;
void* at_imagine_weak_storage_new(tensor value) {
  PROTECT(return new ImagineWeakStorage(value->storage().getWeakStorageImpl());)
  return nullptr;
}
int at_imagine_weak_storage_expired(void* value) {
  return static_cast<ImagineWeakStorage*>(value)->expired();
}
void at_imagine_weak_storage_free(void* value) {
  delete static_cast<ImagineWeakStorage*>(value);
}
tensor at_imagine_source_normal(tensor source, void* handle) {
  PROTECT(
    std::optional<at::Generator> generator = handle
      ? std::optional<at::Generator>(*static_cast<at::Generator*>(handle)) : std::nullopt;
    return new torch::Tensor(at::randn(source->sizes(), generator,
      at::TensorOptions().dtype(source->scalar_type()).layout(source->layout()).device(source->device())));
  )
  return nullptr;
}
void at_imagine_index_copy_to(tensor target, tensor source) {
  PROTECT(at::indexing::copy_to(*target, *source);)
}
static std::vector<at::indexing::TensorIndex> imagine_context_indices(
    tensor full, const int64_t* values, int64_t count, int64_t dim) {
  const auto prefix = std::max<int64_t>(dim, 0);
  TORCH_CHECK_INDEX(prefix < full->dim(),
    "too many indices for tensor of dimension ", full->dim());
  // Python sequenceToVariable(self.options(), list) constructs a CPU Long
  // tensor, then transfers it to self's device without nonblocking/copy flags.
  auto indices = at::empty({count}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  if (count != 0) std::memcpy(indices.data_ptr<int64_t>(), values, count * sizeof(int64_t));
  indices = indices.to(full->device(), at::kLong, false, false);
  std::vector<at::indexing::TensorIndex> out(prefix, at::indexing::Slice());
  out.emplace_back(indices);
  return out;
}
tensor at_imagine_context_index(tensor full, const int64_t* values, int64_t count, int64_t dim) {
  PROTECT(return new torch::Tensor(at::indexing::get_item(*full,
    imagine_context_indices(full, values, count, dim)));)
  return nullptr;
}
void at_imagine_context_index_put(tensor full, const int64_t* values, int64_t count, int64_t dim, tensor source) {
  PROTECT(at::indexing::set_item(*full,
    imagine_context_indices(full, values, count, dim), *source);)
}
int64_t at_imagine_context_size(tensor full, int64_t dim) {
  PROTECT(return full->size(dim);)
  return 0;
}
tensor at_imagine_context_scalar_index(tensor full, const int64_t* values, int64_t count) {
  PROTECT(
    std::vector<at::indexing::TensorIndex> indices;
    for (int64_t i = 0; i < count; ++i) indices.emplace_back(values[i]);
    return new torch::Tensor(at::indexing::get_item(*full, indices));
  )
  return nullptr;
}
tensor at_imagine_context_slice(tensor full, int64_t dim, int64_t start, int64_t end) {
  PROTECT(
    TORCH_CHECK_INDEX(dim >= 0 && dim < full->dim(),
      "too many indices for tensor of dimension ", full->dim());
    std::vector<at::indexing::TensorIndex> indices(dim, at::indexing::Slice());
    indices.emplace_back(at::indexing::Slice(start, end));
    return new torch::Tensor(at::indexing::get_item(*full, indices));
  )
  return nullptr;
}
static std::vector<at::indexing::TensorIndex> imagine_context_scalar_at_indices(tensor full, int64_t index, int64_t dim) {
  const auto prefix = std::max<int64_t>(dim, 0);
  TORCH_CHECK_INDEX(prefix < full->dim(), "too many indices for tensor of dimension ", full->dim());
  std::vector<at::indexing::TensorIndex> indices(prefix, at::indexing::Slice());
  indices.emplace_back(index);
  return indices;
}
tensor at_imagine_context_scalar_at(tensor full, int64_t index, int64_t dim) {
  PROTECT(return new torch::Tensor(at::indexing::get_item(*full, imagine_context_scalar_at_indices(full,index,dim)));)
  return nullptr;
}
void at_imagine_context_scalar_at_put(tensor full, int64_t index, int64_t dim, tensor source) {
  PROTECT(at::indexing::set_item(*full, imagine_context_scalar_at_indices(full,index,dim), *source);)
}
tensor at_imagine_global_rng_state(int device) {
  PROTECT(
    auto gen = at::globalContext().defaultGenerator(device < 0 ? c10::Device(c10::DeviceType::CPU) : c10::Device(c10::DeviceType::CUDA, device));
    std::lock_guard<std::mutex> lock(gen.mutex());
    return new torch::Tensor(gen.get_state());
  )
  return nullptr;
}
tensor at_imagine_clone(tensor value) {
  PROTECT(return new torch::Tensor(value->clone());)
  return nullptr;
}

int at_imagine_current_stream_capturing() {
#if __has_include(<cuda_runtime_api.h>)
  PROTECT(return at::cuda::currentStreamCaptureStatus() != at::cuda::CaptureStatus::None;)
#else
  PROTECT(TORCH_CHECK(false, "CUDA capture status requires the pinned CUDA headers");)
#endif
  return 0;
}

tensor at_imagine_generator_randperm(void* handle, int64_t n) {
  PROTECT(
    auto& generator = *static_cast<at::Generator*>(handle);
    return new torch::Tensor(at::randperm(n, generator, at::TensorOptions().device(generator.device()).dtype(at::kLong)));
  )
  return nullptr;
}
