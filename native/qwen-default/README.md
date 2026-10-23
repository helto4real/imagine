# Qwen app default

This motor combines the accepted reference-cache implementation with the
whole-matrix INT8/ConvRot LoRA preparation and the silent-loader wait fix.
The app selects the pinned executable for Qwen Image 2.1 passes. Reference
passes retain hybrid prefix placement and asynchronous prefetch; text-only
passes do not enable the reference cache. Turbo keeps its existing motor.

The app supplies `SD_QWEN21_MERGE_LORA_INT8=auto` and
`GGML_CUDA_FUSED_LORA_OUTPUT=1`. Before changing weights, the native engine
checks the entire selected adapter configuration. Ordinary linear LoRAs on
single-CUDA INT8/ConvRot group-256 weights, width <=24576 and rank <=256 use
Comfy-style FP16 base/addition, FP32 products and stochastic requantization.
Other configurations use the existing runtime adapter path. This choice does
not filter the model library or discard adapters. A failure after preparation
has started is reported; it is not retried against partially modified weights.

Switching to runtime adapters or removing LoRA clears the merged configuration
through the existing restoration/epoch mechanism. Memory reservations and
precision arithmetic are unchanged from the validated merge candidate.

The app verifies the executable against its compiled source-lock pin. An absent
or mismatched default executable retains the earlier motor selection: accepted
reference motor for references, frozen v2 for text-only passes or fallback.
`IMAGINE_QWEN_REFERENCE_CACHE=off` still disables reference caching. Frozen v1/v2
and the original reference and LoRA controls remain preserved.

## Reproduce

```sh
python3 native/qwen-default/build.py --verify-only
SD_BUILD_JOBS=4 python3 native/qwen-default/build.py
python3 native/qwen-default/install.py
cargo build --release --workspace
```

The build restores pinned upstream/submodule revisions, complete base patches,
the LoRA-output and LoRA-merge overlays, then the complete `qwen-default.patch`.
It verifies all final source hashes and frozen control hashes. The final patch
contains all five accepted reference overlays plus automatic LoRA selection.
New CUDA code and its license remain published in the parent LoRA overlay.
`install.py` accepts only the already pinned binary; a changed build requires
validation and an explicit pin update. Generated executables are ignored by Git.

The app's running server needs a restart to load a rebuilt release executable.
The existing v2 manifest remains the immutable source of common model paths and
settings; the new executable is pinned separately in this directory.
