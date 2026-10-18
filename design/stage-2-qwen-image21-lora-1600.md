# Qwen Image 2.1 — 1600-pixel portrait LoRA comparison

2026-10-18. Tomas requested a real Rust/ComfyUI portrait comparison, both images
shown in the conversation, with maximum side 1600 to exercise automatic offload.
This is isolated development outside ordinary Dreamchat. Production data,
services and defaults are untouched. No new parity exception is authorized.

## Workload and source decisions

The accepted 1024×1024 portrait-editing baseline at commit `5f3369a` is preserved.
The new profile is `inference/profiles/qwen21-lora-portrait-1600-int8.json`:
1600×1600 output, the same unmodified 1024×1024 adult-portrait source tensor,
seed 3061383787, 40 Euler/simple steps, CFG 1, shift 0.69, frozen garden-edit
prompt and empty negative prompt. DiT and Qwen3-VL use INT8 ConvRot; VAE and
compute are BF16. Lenovo LoRA model strength is 1 and CLIP strength is 0.
Attention, compiler, Kitchen and autotuning policy remain as in the control.

The requested larger image requires `reference_resolution=1600`: the original
TextEncodeQwenImage21 inference body derives output geometry from that value
and the reference aspect ratio. Both engines therefore resize the source with
the same original Pillow Lanczos path before reference encoding. The source
file/hash does not change. `force_full_load_per_component=false` selects the
existing automatic static memory policy, as authorized for this test.

A first preflight rejected an inconsistent profile (1600 output but original
reference resolution 0). It stopped before model loading. A second attempt
encoded the reference, then exposed an existing implementation gap: the static
DiT owner accepted ordinary BF16 storage only and rejected quantized weights.
Neither attempt produced a final image; preserve them as failures.

S2-LORA-INT8-OFFLOAD-01 extends the selected static owner according to the frozen
ComfyUI `_load_list`, `module_size`, `patch_weight_to_device`, pinning and
`cast_bias_weight` paths. Serialized INT8 scales and `comfy_quant` JSON count
in module placement bytes; pin registration counts the packed payload only.
Quantized transfers and CPU backups preserve layout/scales/logical dtype.
Unpatched quantized weights reserve the original dequantization workspace;
patched weights replace that with the original two-copy math allowance.
Resident patches retain the FP16→FP32→requantize path. Lazy patches dequantize
in compute dtype, apply ordered functions and preserve source storage; Linear
and SwiGLU retain original quantized versus ordinary operation selection.
These implement compatibility, without changing source arithmetic or policy.

## Validation status

The bounded new oracle captures 16 original module inventories and 12 tiny
CPU/GPU INT8/ConvRot cases with repeated Linear/SwiGLU calls. It never
constructs actual model weights. Initial oracle environment errors (missing
format constant and Kitchen alias) are retained separately from passing results.

## Trained result

The fresh-process pair `qwen-lora-portrait-1600-20261018-03` passes the strict
comparison gate. All six Float32 tensor exports are finite and byte-identical:
positive context, negative context, reference vision, reference VAE latent,
final sampled latent and final RGBA. Decoded PNGs have **zero unequal pixels**.
Conditioning metadata and all 192 actual DiT LoRA parameter records match;
CLIP has zero patches, as requested. No diagnostic placement replay was used.

| Measured phase, seconds | Rust | ComfyUI Python |
| --- | ---: | ---: |
| Reference VAE load + encode | 1.825 | 0.605 |
| Text component load | 1.452 | 1.475 |
| Text/vision encoding | 9.090 | 8.813 |
| DiT load + patch | 7.428 | 1.217 |
| Sampling, 40 steps | **90.544** | **91.645** |
| Final VAE load | 0.928 | 0.213 |
| Final VAE decode | 1.096 | 1.087 |
| Pipeline including unload and intermediate exports | **114.913** | **106.848** |

Rust is 8.065 seconds slower in the recorded pipeline; the largest recorded
phase difference is DiT load/patch (6.211 seconds). Sampling differs by 1.100
seconds in Rust's favor. These are one sequential fresh-process pair, Rust then
Python, with already warm OS model cache. CUDA is synchronized at phase
boundaries. Launcher/build/preflight and final PNG export are excluded. This
is neither a request-to-PNG benchmark nor an established general speed ranking.
No tuning or causal loading-performance investigation is part of this test.

The image shows the adult woman in a blue blouse against sunlit green foliage,
with coherent hair detail and skin texture. Both engines have exactly the same
visual quality for this input because their decoded pixels are identical.
Resolution and LoRA differ from the historical control, so this does not
isolate a LoRA quality improvement or guarantee quality for other seeds.

| Rust | Frozen ComfyUI Python |
| --- | --- |
| ![Rust portrait](stage-2-qwen-image21-lora-1600-evidence/rust/image.png) | ![ComfyUI portrait](stage-2-qwen-image21-lora-1600-evidence/python/image.png) |

## Actual memory behavior and remaining boundary

Both engines report the same 330 resident DiT module entries at initial load
and sampling, zero lazy-weight modules, zero offload buffer bytes and zero
pinned model-weight bytes. The **5,271,715,840-byte (4.910 GiB) prefix cache**
is in CPU RAM, pinned, with two GPU staging buffers in both engines. Thus the
larger trained workload exercises actual **cache offload**, while DiT weights
still fit on GPU. It does not establish trained lazy-patched DiT-weight
execution under memory pressure; that route has bounded synthetic evidence
here. No competing workload or artificial VRAM allocation was introduced.

An accounting boundary remains explicit: Rust's DiT-only loaded counter is
7,256,708,608 bytes, versus the original full wrapper's 7,256,748,608 bytes, a
40,000-byte difference in both phases. Original `ModelSamplingFlux` registers
10,000 Float32 sigmas (40,000 bytes), and original full `load()` assigns
`model_size()` including wrapper buffers. The current static owner inventories
DiT modules; complete wrapper-buffer ownership/accounting is still Stage 2
work. Equal physical module placement and exact image results here do **not**
constitute full wrapper/allocator parity or permission to keep that gap as a
permanent behavioral exception. No threshold-boundary claim is made.

The selected INT8 static binding still explicitly rejects unimplemented
registered scale extras, other quantized layouts and shape-changing patch
lifecycles. General clone/shared ownership, wrapper buffers, broader adapters,
OOM/tiling and the rest of Stage 2 remain pending. No approximation, fallback
policy, new numerical tolerance or parity exception has been adopted.

## Preservation and replay

The evidence directory retains both PNGs, results, conditioning/cache/memory
observations, strict comparison, namespace proof, failure logs and before-change
source snapshots. The six large raw tensor exports remain in the isolated run;
the comparison preserves their complete SHA256 values and exactness metrics.
The source fixture preserves all bounded oracle inputs/expected outputs.
Original portrait/model files, four Stage 0 locks, earlier profiles and earlier
408-file LoRA evidence are preserved. Model hashes are recomputed in full and
match the accepted baseline. No ordinary Dreamchat data was accessed.

```sh
export IMAGINE_PARITY_ROOT=/tmp/imagine-rust-comfy-parity-20261017
export IMAGINE_REFERENCE_SITE=/home/thhel/git/ComfyUI/.venv/lib/python3.12/site-packages
bash inference/tools/check_qwen_portrait_editing.sh \
  /home/thhel/git/ComfyUI /mnt/media/gemensam/AI/ComfyUI/models \
  FRESH_RUN_PREFIX \
  /home/thhel/ChatGPT/imagine/inference/profiles/qwen21-lora-portrait-1600-int8.json
```

The command requires an explicit fresh prefix and selected local model files.
It runs Rust without a Python interpreter or ComfyUI source in its namespace,
then the original local Python oracle, then the strict comparison. Its earlier
1 MP default is retained; this 1600 workload requires its explicit profile.
GPU/namespace escalations were approved automatically; none were rejected.
No commit, push, production service or engine-default change was made.


## Final checks

All **30 focused Qwen tests pass** on the final inference source, including
plain BF16/mixed offload, sampler, conditioning, vision, VAE and all LoRA tests.
The new INT8 inventory/lazy-Linear regression also passes alone in Python-free
isolation. Earlier in this turn the four existing LoRA tests pass together.
These are synthetic source/regression checks; the trained pair above is
separate evidence. No browser or production-service test was run.

Locked/offline driver build, formatting, all-target Clippy with warnings denied,
three Python AST checks and both affected shell syntax checks pass. The only
post-build changes are test-list registration, generic completion wording and
documentation; inference source is unchanged from the measured image pair.
The new checkpoint hashes all retained evidence and current implementation
inputs. Earlier source snapshots explain historical manifest differences;
old expected results and old evidence hashes were not rewritten.
