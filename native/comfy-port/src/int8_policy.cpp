// SPDX-License-Identifier: GPL-3.0-or-later
// Behavioral port of Kitchen 0.2.37 backends/cuda/__init__.py.
#include "int8_policy.hpp"

namespace imagine_comfy {
    int64_t convrot_fused_shared_memory(int64_t rows, int64_t columns) {
        const int threads = rows == 1         ? 512
                            : columns == 256  ? 64
                            : columns == 2560 ? 640
                            : columns == 6144 ? 768
                                              : 1024;
        return (columns + (threads / 64) * 2 * 256) * 4;
    }
    ConvRotQuantization convrot_quantization(int64_t rows, int64_t columns, int64_t group_size, int64_t shared_memory) {
        const bool native      = group_size == 256 && columns % 256 == 0;
        const bool shared_fits = convrot_fused_shared_memory(rows, columns) < shared_memory;
        if (native && columns >= 256 && columns <= 16384 && shared_fits)
            return ConvRotQuantization::Fused64;
        if (native && columns >= 1024 && columns < 8192)
            return ConvRotQuantization::Staged;
        if (native && columns <= 16384 && (columns <= 5120 || columns >= 8192) && shared_fits)
            return ConvRotQuantization::FusedRowwise;
        if (native)
            return ConvRotQuantization::Staged;
        return ConvRotQuantization::RotatedRowwise;
    }
    bool convrot_native_dequantization(int64_t columns, int64_t group_size) {
        return (group_size == 64 || group_size == 256) &&
               columns % group_size == 0 && columns <= 16384;
    }
}  // namespace imagine_comfy
