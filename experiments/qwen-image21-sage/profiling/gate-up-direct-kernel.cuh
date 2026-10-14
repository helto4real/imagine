/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#ifndef GGML_CUDA_GATE_UP_DIRECT_CUH
#define GGML_CUDA_GATE_UP_DIRECT_CUH
namespace GateUpDirect {
using namespace cute;
using namespace cutlass;
using namespace cutlass::epilogue::threadblock;
// Map logical paired columns to the original [gate, up] weight layout.
struct WeightPermutation {
  CUTLASS_HOST_DEVICE WeightPermutation() = default;
  CUTLASS_HOST_DEVICE WeightPermutation(layout::PitchLinearCoord, int) {}
  CUTLASS_HOST_DEVICE int64_t operator()(layout::PitchLinearCoord coord) const {
    const int col = coord.strided();
    return int64_t((col & 1) * 12288 + col / 2) * 4096 + coord.contiguous();
  }
};
// Row vector broadcast
template<
  class ThreadMap,
  class Element,
  class StrideMNL,
  bool EnableNullptr = true // Fallback scalar broadcast for nullptr params
>
struct VisitorDirectWeightScale {

  struct Arguments {
    Element const* ptr_row = nullptr;
    Element null_default = Element(0);
    StrideMNL dRow = {};
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  struct SharedStorage {};

  // Global load type
  static int constexpr vec_bits = ThreadMap::kElementsPerAccess * cutlass::sizeof_bits<Element>::value;
  using VecType = uint_bit_t<cute::min(128, vec_bits)>;
  static int constexpr VecLength = sizeof(VecType) / sizeof(Element);

  CUTLASS_HOST_DEVICE
  VisitorDirectWeightScale() { }

  CUTLASS_HOST_DEVICE
  VisitorDirectWeightScale(Params const& params, SharedStorage const& shared_storage)
    : params_ptr(&params) { }

  Params const* params_ptr;

  template <class GTensor, class RTensor, class CTensor, class ProblemShape>
  struct Callbacks : EmptyCallbacks {
    CUTLASS_DEVICE
    Callbacks(
      GTensor&& tC_gRow,
      RTensor&& tC_rRow,
      CTensor&& tC_cRow,
      ProblemShape problem_shape,
      Params const* params_ptr
    ):
      tC_gRow(cute::forward<GTensor>(tC_gRow)),
      tC_rRow(cute::forward<RTensor>(tC_rRow)),
      tC_cRow(cute::forward<CTensor>(tC_cRow)),
      n(get<1>(problem_shape)),
      params_ptr(params_ptr) { }

    GTensor tC_gRow;
    RTensor tC_rRow;
    CTensor tC_cRow;
    Params const* params_ptr;
    int n;

    CUTLASS_DEVICE void
    begin_epilogue() {
      if constexpr (EnableNullptr) {
        if (params_ptr->ptr_row == nullptr) {
          auto tC_rRow_vec = recast<Array<Element, VecLength>>(coalesce(tC_rRow));
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tC_rRow_vec); ++i) {
            tC_rRow_vec[i].fill(params_ptr->null_default);
          }
          return;
        }
      }
      clear(tC_rRow);
      auto src_v = filter(tC_gRow);
      auto coord_v = filter(tC_cRow);
      auto dst_v = filter(tC_rRow);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(src_v); ++i) {
        bool guard = get<1>(coord_v(i)) < n;
        static_assert(VecLength == 4);
        const int col = get<1>(coord_v(i));
        auto * values = reinterpret_cast<Element *>(&dst_v(i));
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < VecLength; ++j) {
          if (guard) values[j] = params_ptr->ptr_row[((col + j) & 1) * (n / 2) + (col + j) / 2];
        }
      }
    }

    template <class ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE auto // returns an Array
    visit(int iter_idx, int row_idx, int column_idx, int frg_idx,
          Array<ElementAccumulator, FragmentSize> const& frg_acc) {
      Tensor rRow_frg = recast<Array<Element, FragmentSize>>(coalesce(tC_rRow));
      return rRow_frg(column_idx);
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
    gemm::GemmCoord threadblock_tile_offset,
    int thread_idx,
    ProblemShape problem_shape
  ) {
    Tensor mRow = make_tensor(
      make_gmem_ptr(params_ptr->ptr_row),
      problem_shape,
      params_ptr->dRow);

    // VECTOR, FRAGMENT_COLUMN
    Tensor tC_gRow = recast<VecType>(
      ThreadMap::partition(mRow, thread_idx, threadblock_tile_offset)
    )(_,_,_0{},_0{},_0{},_0{});
    Tensor tC_rRow = make_tensor_like(tC_gRow);

    // Generate the pred tensor
    Tensor cRow = make_identity_tensor(mRow.shape());
    Tensor tC_cRow = outer_partition(
      ThreadMap::partition(cRow, thread_idx, threadblock_tile_offset)(_,_,_0{},_0{},_0{},_0{}),
      Shape<Int<VecLength>>{},
      (_0{})
    );

    return Callbacks<
      decltype(tC_gRow), decltype(tC_rRow),
      decltype(tC_cRow), ProblemShape>(
      cute::move(tC_gRow),
      cute::move(tC_rRow),
      cute::move(tC_cRow),
      problem_shape,
      params_ptr
    );
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////


template<class Tile, class Warp, class Map, class ScaledInput, class Multiply, class Store>
struct Projection {
  using WeightScale = VisitorDirectWeightScale<Map,float,cute::Stride<cute::_0,cute::_1,cute::_0>>;
  using ScaledOutput = cutlass::epilogue::threadblock::Sm80EVT<Multiply,ScaledInput,WeightScale>;
  using Epilogue = cutlass::epilogue::threadblock::Sm80EVT<Store,ScaledOutput>;
  using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;
  using Base = typename cutlass::gemm::kernel::DefaultGemmUniversal<
    int8_t,cutlass::layout::RowMajor,cutlass::ComplexTransform::kNone,16,
    int8_t,cutlass::layout::ColumnMajor,cutlass::ComplexTransform::kNone,16,
    float,cutlass::layout::RowMajor,int32_t,
    cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
    Tile,Warp,cutlass::gemm::GemmShape<16,8,32>,
    cutlass::epilogue::thread::LinearCombination<float,4,int32_t,float>,
    Swizzle,3,cutlass::arch::OpMultiplyAddSaturate,
    cutlass::gemm::SharedMemoryClearOption::kNone,false,false,false,
    cutlass::layout::NoPermute,cutlass::layout::NoPermute,WeightPermutation>::GemmKernel;
  using VisitorEpilogue = cutlass::epilogue::threadblock::EpilogueWithVisitorCallbacks<
    typename Base::Epilogue,Epilogue,1>;
  using Kernel = cutlass::gemm::kernel::GemmWithEpilogueVisitor<typename Base::Mma,VisitorEpilogue,Swizzle>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};
}
#endif
