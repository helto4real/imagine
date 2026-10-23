// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef IMAGINE_COMFY_INT8_POLICY_HPP
#define IMAGINE_COMFY_INT8_POLICY_HPP
#include <cstdint>

namespace imagine_comfy {
    enum class ConvRotQuantization { Fused64,
                                     Staged,
                                     FusedRowwise,
                                     RotatedRowwise };
    int64_t convrot_fused_shared_memory(int64_t rows, int64_t columns);
    ConvRotQuantization convrot_quantization(int64_t rows, int64_t columns, int64_t group_size, int64_t max_dynamic_shared_memory);
    bool convrot_native_dequantization(int64_t columns, int64_t group_size);
}  // namespace imagine_comfy
#endif
