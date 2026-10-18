#pragma once
// Metadata-only replacement for the ndarray objects accepted by the pinned
// upstream Kitchen launch wrappers. No Python or alternate kernel arithmetic.
#include <ATen/ATen.h>
#include <ATen/DLConvertor.h>
#include <optional>
#include <map>
#include <stdexcept>
#include <type_traits>

enum class DTypeCode : uint8_t { Int=0, UInt=1, Float=2, Bfloat=4, Bool=6 };
struct DType { uint8_t code; uint8_t bits; uint16_t lanes; bool operator==(const DType&) const = default; };
template<class T> DType view_dtype() {
    if constexpr(std::is_same_v<T,float>) return {2,32,1};
    if constexpr(std::is_same_v<T,int32_t>) return {0,32,1};
    static_assert(std::is_same_v<T,float> || std::is_same_v<T,int32_t>);
}
using PlanMap=std::map<std::string,int64_t>;
struct CudaDevice {};
template<int N> struct Dims { static constexpr int dims=N; };
template<class... Tags> struct ScalarTag {using Type=void;};
template<class First,class... Rest> struct ScalarTag<First,Rest...> {
    using Type=std::conditional_t<std::is_arithmetic_v<First>,First,typename ScalarTag<Rest...>::Type>;
};
template<class T, class=void> struct DimensionTag { static constexpr int value=-1; };
template<class T> struct DimensionTag<T,std::void_t<decltype(T::dims)>> { static constexpr int value=T::dims; };

template<class... Tags> class TensorView {
    at::Tensor tensor_;
    DType dtype_;
    template<class Tag> void validate() {
        if constexpr (std::is_arithmetic_v<Tag>) {
            constexpr uint8_t code=std::is_floating_point_v<Tag> ? 2 : (std::is_same_v<Tag,bool> ? 6 : (std::is_signed_v<Tag> ? 0 : 1));
            if(dtype_!=DType{code,static_cast<uint8_t>(8*sizeof(Tag)),1})
                throw std::runtime_error("Typed tensor view dtype mismatch");
        }
        if constexpr (DimensionTag<Tag>::value>=0) {
            if(tensor_.dim()!=DimensionTag<Tag>::value) throw std::runtime_error("Tensor rank mismatch");
        }
    }
public:
    explicit TensorView(const at::Tensor& tensor):tensor_(tensor) {
        if(!tensor_.is_cuda()) throw std::runtime_error("Expected CUDA tensor");
        // Use the same DLPack conversion as Tensor.__dlpack__. New shell dtypes
        // have distinct codes (e.g. E8M0=14); mapping all FP8 to float is wrong.
        auto* managed=at::toDLPack(tensor_);
        const auto dtype=managed->dl_tensor.dtype;
        dtype_={dtype.code,dtype.bits,dtype.lanes};
        managed->deleter(managed);
        (validate<Tags>(),...);
    }
    using Pointer=typename ScalarTag<Tags...>::Type*;
    Pointer data() const { return static_cast<Pointer>(tensor_.data_ptr()); }
    int64_t shape(size_t i) const { return tensor_.size(i); }
    int64_t stride(size_t i) const { return tensor_.stride(i); }
    int ndim() const { return tensor_.dim(); }
    int64_t size() const { return tensor_.numel(); }
    int device_id() const { return tensor_.get_device(); }
    DType dtype() const {return dtype_;}
};
struct OptionalTensorView {
    const at::Tensor* tensor;
    bool is_none() const { return tensor==nullptr; }
};
template<class T> T view_cast(OptionalTensorView view) {
    if(view.is_none()) throw std::runtime_error("Expected tensor");
    return T(*view.tensor);
}
