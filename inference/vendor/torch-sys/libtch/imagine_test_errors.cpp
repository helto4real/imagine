// Bounded fault injection, built only for the engine's dev-dependency feature.
#include "torch_api.h"
#include <torch/library.h>
static at::Tensor throw_bounded_error(int64_t kind, int64_t code, std::string message) {
    switch (kind) {
        case 0: throw c10::OutOfMemoryError(message);
        case 1: throw c10::AcceleratorError({__func__, __FILE__, __LINE__}, code, message);
        default: throw std::runtime_error(message);
    }
}
extern "C" void at_imagine_test_error(int kind, int code, const char* message) {
    PROTECT(throw_bounded_error(kind, code, message);)
}
TORCH_LIBRARY(imagine_test, m) {
    m.def("throw_error(int kind, int code, str message) -> Tensor", throw_bounded_error);
}
