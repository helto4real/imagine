# ComfyUI behavior port

## Binding scope

Port the actual installed ComfyUI behavior, not a smaller compatibility subset:
all built-in adapter types, weight patching, precision, rounding order and
decisions per weight/layer. Do not introduce native-specific format/rank limits
or a configuration-wide alternative calculation policy. If a behavior cannot
be ported, document its exact divergence and obtain explicit approval.

User decisions in this task:

- Install the existing combined implementation as the current server default
  without additional tests. This interim deployment is complete: the pinned
  native binary is installed, the release server is rebuilt and the running
  server's executable matches that build. Existing sessions were not read.
- Exclude external Python hooks and external Python adapter classes.
- Keep the current native/CUDA dependencies. Do not introduce Python or
  LibTorch/ATen.
- For OFT/BOFT, use CUDA's cuSOLVER/cuBLAS LU solvers in the cases where the
  installed PyTorch would use MAGMA. This explicitly approved exception can
  change the last bits of LU results. Keep the inverse formula, tensor dtypes,
  cast placement and patch order unchanged. See `approved_inverse_exception`
  in `reference-lock.json` for the exact branch condition.
- Do not run further tests. Builds and source inspection remain available.

The original requested pre-integration commit is `59a52ac`. Subsequent
integration and port changes remain uncommitted. Frozen v1/v2 and all earlier
reference/LoRA controls remain intact. The installed interim profile is recorded
in `../qwen-default/baseline.json`; it is not a claim of complete Comfy parity.

## Current implementation state

`src/adapter_loader.cpp` ports the built-in detection order and payload tuples
from `comfy/lora.py:load_lora` and the six registered adapter classes:
LoRA, LoHa, LoKr, GLoRA, OFT and BOFT. It includes seven LoRA naming variants,
mid/reshape/DoRA references, the LoKr direct/decomposed variants, OFT/BOFT
dimension discrimination, normalization/diff/bias/set/config patches and
loaded/unused key bookkeeping. Key-map insertion and overwrite order are kept.
There is no rank/group/width eligibility filter in this loader.

This component has only been compiled as a standalone static library. It is
not yet wired into model loading, tensor conversion or weight execution.
Metadata references do not implement numerical operations or prove parity.
`src/model_keys.cpp` now ports the QwenImage aliases, including Qwen 2.1's
gate_layer/proj slices into the fused gate_up output axis. Slice targets remain
distinct while loader entries are overwritten. `add_patches` converts them to
ordered offset patches for weights present in the supplied model catalog and
handles configurable-module patches separately. Binding this mapping to the
actual native catalog is still required.

`src/weight_policy.cpp` also ports the module memory-fit comparison and the
per-weight selection between stored patching and `LowVramPatch`, including
shape-changing/force-patch exceptions, distinct intermediate dtypes and layer
setter versus stochastic writeback. This policy is compiled but not connected
to the engine's residency/memory accounting. It must not be represented as
replacing the installed configuration-wide choice yet.

`src/adapter_math.cpp` now expresses the ordered tensor operations for all six
built-in matrix adapters and DoRA. LoRA includes mid/reshape; LoHa includes both
Tucker products; LoKr includes direct/decomposed components; GLoRA retains both
shape-dependent formulas. OFT and BOFT retain the exact cast placement, strength
use, norm constraint, inverse and butterfly expressions from the reference,
including their different missing-alpha behavior. Exception boundaries follow
the adapter methods rather than turning every error into a silent no-op.

`src/weight_patcher.cpp` ports the generic ordered patch loop: model strength,
offset views, nested patches, padding, diff/set and model-as-LoRA. Nested patches
deliberately do not inherit the original-weight map, matching the reference
recursive call. `src/model_patcher.cpp` ports the normal weight backup,
conversion, writeback and per-key seed sequence. Seed generation iterates Unicode
code points like the Python implementation, not UTF-8 bytes.

The concrete tensor backend now has native storage/views, CPU/CUDA copies and
casts, elementwise arithmetic, cuBLAS matrix products, Kronecker products and
einsum contraction. Casts preserve the input's dense permutation; elementwise
outputs use TensorIterator's stride ordering and dtype categories, including
zero-dimensional tensor promotion. Reshape materializes only when its original
strides cannot represent a view.

`src/tensor_reduce.*` ports CUDA sums and 2-norms, including TensorIterator's
dimension sorting/coalescing, input/output vector selection, four independent
thread accumulators, block/CTA reduction order and recursive 32-bit iterator
splitting. Half/BF16 partial reductions retain FP32 accumulation. Norms over
singleton dimensions use the reference absolute-value shortcut.

`src/tensor_inverse.cu` implements the general FP32/FP64 inverse as LU plus a
solve against identity. It uses cuSOLVER's looped getrf or cuBLAS's batched getrf
according to the pinned CUDA branch's dimension/dtype heuristic. Width <=16
uses batched getrs; larger inverses use pivot application followed by lower and
upper triangular solves, with the reference looped/batched trsm selection.
The approved MAGMA exception applies to factorization only.

`kitchen/` contains byte-identical Kitchen 0.2.37 native INT8 kernel source and
its three CUDA headers. Their Git blob identities match the pinned upstream
tree. They compile with Kitchen's own CUDA precision flags (`--use_fast_math`)
in a separate static target, without Python, nanobind or LibTorch. This avoids
rewriting its ConvRot rotation, partial-absmax, dtype rounding and stochastic
INT8 kernels. The tensor layout dispatch and native layer-storage binding for
these entry points are still required; compiling them does not activate them.

These components compile into the standalone static library; they are **not
installed in the server and do not yet constitute executable model parity**.
Actual model storage, source loading, quantized layer conversion, writeback and
memory-accounting integration are still missing. Numerical/stride review of
the concrete primitives also remains necessary, including general matmul
layout/dispatch. No tests or model execution have been run for these components.
The installed interim motor is unchanged.

Remaining required work:

- Finish source review and engine binding of the concrete current-native/CUDA
  `TensorOps` operations. Every operation must materialize its reference dtype;
  do not fuse away rounding boundaries or promote mixed matrix-multiply inputs
  that PyTorch would reject. General matmul dispatch/layout still needs review.
- Implement concrete `LayerStorage`, actual model alias mapping and native
  catalog binding, including quantized `convert_weight`/`set_weight`, backups
  and stochastic conversion per weight. F32/F16/BF16 in `comfy.float` use
  ordinary casts; FP8 and layer-specific INT8 setters have their own rounding.
- Bind the pinned Kitchen kernel entry points with its actual Python-side
  native dispatch conditions, including its staged path for wide ConvRot rows.
  Do not retain the interim merge kernel's rank/row-width restrictions or
  substitute a different rounding/rotation path.
- Wire the ordered adapter/generic patch code to actual native tensor storage,
  preserving the offset aliases and original-weight lifetime. Only external
  Python functions are excluded; built-in operations retain their semantics.
- Integrate the ported `LowVramPatch` dtype/layer policy with actual native
  memory accounting. Replace the interim configuration-wide
  `can_merge_int8_loras` choice.
- Preserve the exact intermediate casts and operation ordering, including OFT's
  repeated strength use as written in the reference, rather than correcting or
  simplifying its expression.
- Replace the interim rank <=256/group-256/width <=24576 preparation restriction
  with the actual reference-supported behavior. Do not silently route around it
  or discard an adapter.
- Integrate the port into the combined reference/LoRA engine, publish complete
  source patches/hashes, build and activate it without claiming unrun tests.

## Reference evidence

`reference-lock.json` pins the local ComfyUI source files and the installed
PyTorch revision. Comfy's adapter registry is in
`comfy/weight_adapter/__init__.py`; trainable `adapter_maps` is not the inference
registry. Full registry loading must not be reduced to that smaller map.

Comfy's normal patch path uses `lora_compute_dtype(device)` for the base and
FP32 default intermediate adapter math. `LowVramPatch.__call__` instead passes
the supplied weight dtype as the intermediate dtype. These paths cannot be
represented by a single unconditional FP32-product rule.

OFT/BOFT use a general matrix inverse after an explicit FP32 cast. GGML exposes
a restricted triangular solver; the port therefore supplies a native CUDA
general inverse. Only the documented MAGMA-to-CUDA factorization substitution
is approved. No silently skipped adapter or unrelated alternative is approved.

The installed PyTorch revision implements inverse via a solve against an
identity matrix, with CUDA dispatch depending on dimensions and batch count:

- [inverse implementation](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/BatchLinearAlgebra.cpp#L1531).
- [CUDA solve dispatch](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cuda/linalg/BatchLinearAlgebra.cpp#L1231).

The loader is derived from GPL-3.0-or-later ComfyUI sources; its SPDX headers
and the accompanying `LICENSE` retain that provenance.
