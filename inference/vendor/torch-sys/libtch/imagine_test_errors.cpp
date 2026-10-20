// Bounded fault injection, built only for the engine's dev-dependency feature.
#include "torch_api.h"
#include <torch/library.h>
static at::Tensor throw_bounded_error(int64_t kind, int64_t code, std::string message) {
    switch (kind) {
        case 0: throw c10::OutOfMemoryError(message);
        case 1: throw c10::AcceleratorError({__func__, __FILE__, __LINE__}, code, message);
        case 10: throw c10::ValueError(message);
        case 11: throw c10::TypeError(message);
        case 12: throw c10::NotImplementedError(message);
        case 13: throw c10::BufferError(message);
        case 14: throw c10::SyntaxError(message);
        default: throw std::runtime_error(message);
    }
}
extern "C" void at_imagine_test_error(int kind, int code, const char* message) {
    PROTECT(throw_bounded_error(kind, code, message);)
}
// Shape-only checkpoints for detector tests. Never allocate real model data.
extern "C" tensor at_imagine_test_meta_empty(const int64_t* dims, int64_t count, int64_t dtype) {
    PROTECT(return new torch::Tensor(at::empty(at::IntArrayRef(dims,count),
      at::TensorOptions().device(c10::DeviceType::Meta).dtype(static_cast<at::ScalarType>(dtype))));)
    return nullptr;
}
TORCH_LIBRARY(imagine_test, m) {
    m.def("throw_error(int kind, int code, str message) -> Tensor", throw_bounded_error);
}
