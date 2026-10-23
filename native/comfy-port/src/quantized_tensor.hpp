// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef IMAGINE_COMFY_QUANTIZED_TENSOR_HPP
#define IMAGINE_COMFY_QUANTIZED_TENSOR_HPP
#include "native_tensor.hpp"

namespace imagine_comfy {
    // TensorWiseINT8Layout keeps the physical bytes separate from orig_dtype.
    // A dtype-only .to changes that logical dtype, without dequantizing bytes.
    struct Int8Tensor final : NativeTensor {
        std::shared_ptr<void> storage;
        size_t storage_bytes = 0;
        Shape dimensions;
        Dense scales;
        ScalarType original_dtype = ScalarType::Float16;
        int device                = -1;
        bool is_weight            = true;
        bool convrot              = false;
        int64_t group_size        = 256;
        Shape shape() const override { return dimensions; }
        ScalarType dtype() const override { return original_dtype; }
        int device_index() const override { return device; }
        bool quantized() const override { return true; }
    };
    using Int8 = std::shared_ptr<Int8Tensor>;
    void copy_storage_bytes(void* destination, int destination_device, const void* source, int source_device, size_t bytes);
}  // namespace imagine_comfy
#endif
