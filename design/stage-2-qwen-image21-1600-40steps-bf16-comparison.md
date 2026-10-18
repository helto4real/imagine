# Stage 2 — BF16 DiT, 1600×896, 40 steps

2026-10-18. Status: **blocked by reference GPU out-of-memory with the retained
full-residency profile**. No BF16 image or completed timing/parity comparison
is available. The successful [INT8 forty-step comparison](stage-2-qwen-image21-1600-40steps-comparison.md)
and all its evidence remain unchanged. Full Stage 2 remains in progress.

This is the preserved initial full-residency attempt. Tomas subsequently
approved source-equivalent offload in both engines and necessary Rust offload
for future tests. See the [approved continuation](stage-2-qwen-image21-bf16-offload.md)
for that decision, repaired metadata oracle and subsequent results. The original
failed attempt/checkpoint below is retained as historical evidence.

## S2-13 — change DiT weights alone

Tomas requested the same comparison with BF16 at 1600×896 and 40 steps, keeping
other settings. The new explicit [development profile](../inference/profiles/qwen21-1600x896-40steps-bf16.json)
changes only the DiT checkpoint, variant label and descriptive purpose. The
text encoder remains Qwen3-VL INT8 ConvRot and the VAE remains BF16. Diffusion,
text-encoder and VAE precision are independent settings in the agreed design;
this workload must not be described as an all-BF16 pipeline.

The synthetic butterfly prompt, seed 17, batch 1, Euler/simple, CFG 1, FLUX
shift 0.69, no LoRA/references, BF16 compute, PyTorch attention, default cache
dtype/auto thresholds and unchanged Kitchen/autotuning policy are retained.
Static patchers, disabled compiler/async offload, full per-component GPU
residency and unload between components are also retained. The existing
development default and previous registered profiles are unchanged.

Only profile registration, the diagnostic result label and the comparison
report's variant label were extended. No model, loader, numerical, native,
cache or sampling implementation was changed. No setting from the separate
stable-diffusion.cpp baseline was substituted.

## Execution result

The original frozen ComfyUI Python reference completed text encoding and failed
in `comfy.model_management.load_models_gpu([model], force_full_load=True)`
while moving BF16 diffusion weights onto the GPU. The original call chain is
`LoadedModel.model_load` → `model_use_more_vram` →
`ModelPatcher.partially_load` → `ModelPatcher.load` → module `.to(device)`.
The exception is `torch.OutOfMemoryError`. An allocation of 33,554,432 bytes
failed; the allocator diagnostic reported 81,068,032 free device bytes and
16,769,744,896 total device bytes. Allocator fragmentation/rounding and other
allocations prevent interpreting that free counter as a guaranteed allocation
budget. No process inventory was queried or process/service stopped.

There were **zero sampling callbacks**, no latent or image, no completed
reference `result.json`, and no complete matched pair. The serialized runner
stopped on the Python error before starting Rust generation. Partial elapsed
times and cold-start/model-hash I/O must not be presented as comparable image
latency. A complete fresh measurement set is required after the blocker is
resolved; this failed attempt stays excluded.

The BF16 DiT file has 265 BF16 tensors, 14,230,249,472 tensor-data bytes and
14,230,280,616 total file bytes. It has no `.comfy_quant` tensors. A separate
Rust header-only preflight passes all constructor shapes for the three selected
files (265 DiT / 750 text / 238 VAE parameters) in the Python-free isolated
process. At that later preflight, aggregate CUDA free/total memory was
13,380,288,512 / 16,769,744,896 bytes. This is less free memory than the DiT's
weight data alone; it is not a measurement of peak inference VRAM. Header-only
success does not establish full BF16 loading, inference or numerical parity.

## Required decision before changing the retained profile

The frozen ComfyUI source supports its own memory-budgeted CPU/GPU weight
offload when `force_full_load` is false. In `model_management.py`, the budget
depends on actual free memory, loaded memory, minimum inference memory and
`MIN_WEIGHT_MEMORY_RATIO`; `model_patcher.py` then sorts module load entries,
reserves the original offload buffer, and selects resident/offloaded weights.
This must be implemented from source rather than replaced with an arbitrary
fixed number of CPU layers or a new approximate memory threshold.

The current Rust development services explicitly support the selected resident
path; general source-equivalent memory management/host staging remains pending.
Changing the test from full residency to memory-budgeted offload changes an
otherwise retained setting. **Approval was requested; no such change has been
made or assumed approved.** The alternatives presented were:

- Approve the original ComfyUI memory policy in both backends, implement the
  missing Rust binding and validate exact results before measuring fresh pairs.
- Retain the current profile while Tomas frees GPU memory himself. Available
  memory must still cover weights and working tensors; no claim that freeing a
  particular amount guarantees successful generation has been made.

Exact numerical acceptance and unchanged autotuning remain required in either
case. Any eventual offload timing comparison needs its own recorded profile;
the preserved resident INT8 times cannot isolate the effect of precision from
the effect of offload.

## Verification and evidence

All-target Clippy with warnings denied, Rust formatting and Python/shell syntax
checks pass. Source snapshots, the four Stage 0 locks and the 32 original CUDA
objects are verified; model hashes and the exact implementation delta are in
the [evidence directory](stage-2-qwen-image21-1600-40steps-bf16-comparison-evidence/).
The unchanged INT8 text encoder's exported conditioning matches the preceding
INT8 control byte for byte. No full component suite was rerun for this
profile/label-only extension.

Only selected local model assets, source files and this task's isolated
synthetic runtime were used. No production runtime, cloud service, Computer
Use, service replacement, commit or push was involved. Sandbox escalations
were approved; the CUDA OOM was an execution failure, not an automatic approval
rejection. Timing, BF16 image quality and trained BF16 parity remain unverified.
