# Stage 2 — complete Qwen Image 2.1 DiT computation

Continuation: 2026-10-18. The Rust DiT now runs from prepared source-named
parameters, latent/context tensors and timesteps to the predicted latent tensor.
INT8 ConvRot, BF16 and Turbo remain required. Stage 2 as a whole remains in
progress: a trained-checkpoint image-generation engine is not yet complete.

## Implementation

The implementation is in `inference/src/models/qwen_image21/`:

| File | Source behavior |
| --- | --- |
| `dit.rs` | Full timestep embedding, shared modulation, transformer block loop, final AdaLN and output projection/reshape; split/fused MLP; both attention normalization/RoPE branches; block-causal and cached-prefix attention; source wrapper/patch boundaries and FunControl |
| `layers.rs` | Resident mixed-precision weight/bias casting, actual Kitchen quantized storage, source fused INT8 SwiGLU dispatch and eager fallback; modulation splitting and FP16 block clamp |
| `cache.rs` | Qwen prefix-cache selection, exact 4x/2x strict memory thresholds, pin-budget decisions, LRU slot policy, pending-transfer drain, default/INT8/INT4 storage, per-block prefetch/take, repeat-to-batch and explicit reset/free |
| `geometry.rs` | Shared sequence assembly, reused by the earlier component API and complete DiT without changing its reference results |

`DiTConfig::default()` retains the original 32 blocks, 32 heads × 128 dimensions,
4096 context width, 64 input/output channels, MLP ratio 3 and 16/56/56 RoPE axes.
The prepared-state constructor binds original parameter names and retains actual
quantized tensor layouts. It does not load, replace or rewrite model files.
Turbo is not inferred from filenames and does not install a sampling preset.

The full DiT uses the original resident `MixedPrecisionOps.Linear` behavior,
including weight-only INT8 and `linear_input_act` fusion. The earlier component
fixtures remain the original `torch.nn` projection configuration. Both retain
the non-contiguous image projection's original `t/expand/bmm` decomposition.

## Shared runtime boundaries

The source imports shared runtime services rather than implementing them in the
DiT class. The Rust boundary makes these dependencies explicit:

- `DiTExecution` supplies the selected optimized attention function, metadata
  preferences, wrapper/patch execution, mutable forward metadata and allocation/
  prefetch lifecycle and the shared operation-level interruption check. Block
  replacements can call the original block, and model wrappers can call the original forward. An attention override can take
  precedence over saved layer preferences, as in the source.
- `PrefixKv` supplies the sampling-owned cache. Enabling the cache without binding
  its owner returns an explicit development binding gap; it does not silently
  recompute. The source starts with caching disabled until the sampling owner
  enables it, and hooks bypass caching under the original condition.
- `PrefixCache` implements slot, quantization and selection policy.
  `CacheServices` supplies actual patcher/free-memory accounting, inactive-only
  pin budget checks, pin/unpin and stream transfers. `PendingCacheTensor`
  distinguishes the host wait at select/free from the consumer-stream wait at
  take. Shared transfers must retain the original two staging buffers per
  shape/dtype and `block_index % 2` choice.

The reference and candidate tests explicitly supply **resident, same-device
services**, no dynamic vbars/allocation graph, no active offload stream and no
pinning. Operation-level checks are present and the reference executes the
original interruption flag/lock and exception code in its isolated namespace.
Separate cache-policy tests supply synthetic memory budgets. These are
source-supported configurations, not new production defaults. Actual shared
memory/offload, vbar/allocation graph integration, binding of the sampling-owned
cancellation state, weight/bias patch functions and LoRA lifecycle remain Stage 2
integration work.
External attention implementations beyond the selected PyTorch/Kitchen cases
remain shared algorithm coverage; no different attention backend is silently
chosen to fill a missing binding.

## Reference and regression coverage

`reference_qwen_image21_dit.py` verifies the frozen source hashes and executes
the unchanged original definitions, including the actual nested resident
mixed-precision Linear forward methods, `ComfyAttention`, `WrapperExecutor`,
`PoseBranchCache`, FunControl and all Qwen DiT classes. Dependency façades supply
only the explicit services above. The test uses real Torch and Kitchen. It
reads no model weights, prompts, job/session history, production logs or media.

The retained oracle `inference/fixtures/qwen-image21-dit.json` contains:

- **209 complete DiT forwards** in 18 groups: CPU/CUDA, BF16, actual INT8
  ConvRot weights, independently initialized Turbo-labelled BF16/INT8 weights,
  split/fused MLP and 2/32 blocks. Each uses batch 2 and encoded synthetic
  conditioning, with plain/reference, hook/wrapper/attention-override,
  training-branch and cache fill/hit cases. BF16/INT8 cases also change the
  conditioning key and revisit the earlier slot for each cache storage dtype.
- **12 FunControl results**: initialization plus both control block steps for
  BF16 and mixed INT8/BF16 control parameters on CPU/CUDA. In these fixtures, the
  129-column control input stays floating; that width cannot use ConvRot's
  power-of-four group. No replacement quantizer or padding is introduced.
- **48 original cache-selection decisions**, checking dtype-dependent byte
  accounting, strict thresholds, device/off choices, pin-budget arguments and
  exact service-call order.
- **20 interrupted DiT forwards**, injecting the original interruption before
  operation 1/8/20 for BF16/INT8 on CPU/CUDA, plus interruption during cache fill
  and cache hit. Exception class, empty message,
  flag clearing, executed operation count, partial lifecycle and retained cache
  bytes are compared. Prefix-compute pause/resume unwinds on an error, while
  the final graph-end call retains the original non-finally placement. Successful
  forwards also check the number of original interruption control points.

Saved attention configuration exercises an ignored unknown method followed by
Kitchen INT8, including CPU capability fallback. Hooks exercise attention
normalization/RoPE, `post_input`, `single_block`, original block callbacks,
original forward callbacks and attention-override precedence. Original
per-operation interruption checks are propagated through linears, norms and
fused/eager activation routes. Lifecycle order,
`total_blocks`/`block_type`/`block_index` mutation and physical cache payload bytes
are checked against the original, in addition to byte-exact numerical tensors.

Training-branch forward checks run under `no_grad`; backward/optimizer behavior
is outside this inference validation. The 32-block tests use small initialized
matrices and head dimension 64. They verify the complete loop, not the memory demands or output of the trained
4096-wide checkpoint. The attention fixtures exercise the original small-input
SDPA branch, plus real Kitchen INT8 attention. They do not validate large-input
SDPA priority switching or all registered attention implementations.

Turbo-labelled fixtures have distinct synthetic weights and exercise both
BF16 and INT8 execution. They establish DiT arithmetic coverage for those
representations, not parity of a trained Turbo checkpoint or its sampler.
Sampler/checkpoint-level acceptance remains pending for all required variants.

A first attempted 32-block fixture with head dimension 16 hit the original
INT4-cache error (`in_features 16 not divisible by quant_group_size 64`). The
successful numerical fixture uses head dimension 64, valid for the original
quantizer. Its validation was not weakened; no failing case was accepted as a
successful output or used to introduce a parity exception.

## Verification and continuation

Verification status is recorded in
[the DiT checkpoint](stage-2-qwen-image21-dit-checkpoint.json) and
[the retained evidence](stage-2-qwen-image21-dit-evidence/).
The reusable `inference/tools/check_qwen_image21.sh` entrypoint runs all four
Qwen tests and then each test separately in the Python-free, network-isolated
ELF-only runtime. The earlier 192 component/detection cases remain part of this
run, for **481 retained reference results/decisions** in total.

Final validation passed: formatting, Clippy with warnings denied, all four
focused tests and all four individual Python-free isolation runs. The exact
128 latent constants and the shell entrypoint syntax were also verified.
The old full Stage 1 suite is reused for its unchanged implementation; a new
unfiltered 150-primitive/whole-crate run was not performed. Dependency inspection
printed the existing shared-library executable-bit warnings, and the reference
logged its intentionally unknown attention method; neither prevented execution.

A successful resident DiT does not close the complete Stage 2 gate. Remaining
work includes the shared runtime integration described above, checkpoint
loading, Qwen3-VL text/vision encoding, VAE/reference preprocessing, sampling
(including Turbo), LoRA/requantization/restoration and trained-model vertical
validation. No production service or native frozen baseline was changed.
No new parity exception, numerical tolerance or autotuning change is adopted.

The changes remain uncommitted. No source, model or service outside this isolated
inference development scope was changed. GPU/bwrap validation used sandbox
escalation; no escalation was rejected and no additional approval was needed.
