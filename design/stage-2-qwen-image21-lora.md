# Stage 2 — Qwen Image 2.1 LoRA continuation

Date: 2026-10-18. **Selected INT8/BF16/Turbo LoRA functional pairs are exact;
general LoRA and Stage 2 acceptance remain in progress.**
The accepted reference-editing baseline `5f3369a`, earlier T2I baseline `de955e5`
and Stage 1 foundation `059c875` remain intact. This continuation is uncommitted.

## Authority and reference

Tomas authorized the agreed LoRA implementation plan: “Ok gör så, fortsätt
implementering enligt planen”. Necessary ComfyUI-compatible behavior needs no
additional approval; an actual departure from the source or agreed design does.
No numerical tolerance, quantization option, algorithm, autotuning policy or
production default has been changed. All dependencies remain local. Rust
inference uses LibTorch and the required Comfy Kitchen, without Python inference
or stable-diffusion.cpp. Python is used separately as a development oracle.

The reference remains ComfyUI `ba994879b88d0ebd9950b1e5c1e923d928f216ba`, Kitchen
0.2.37 and Torch 2.13.0+cu130 on Arch Linux / RTX 4070 Ti SUPER. Four Stage 0
locks and 32 original CUDA object hashes still match. OFT/BOFT require two lazy
ELF dependencies in standalone isolation: `libtorch_cuda_linalg.so` and
`libcusolver.so.12`. Both already belong to the Stage 0 native linkage lock;
this extends the mounted closure without updating a dependency or kernel.

Only source, bounded synthetic fixtures, explicitly selected model/LoRA weights
and this task's isolated development outputs were read. Ordinary Dreamchat
runtime data, process inventories and services were not inspected or controlled.

## Implemented source path

`inference/src/patching/` implements the six original adapter calculation bodies:
LoRA, LoHa, LoKr, GLoRA, OFT and BOFT. The covered branches preserve alpha/rank,
DoRA axes, convolution/CP/Tucker operands, old/new GLoRA, offsets, stacking order,
model strengths, diff/set/model-as-LoRA/nested values and captured errors and
warnings. Reference quirks, including OFT/BOFT strength behavior and the caught
BOFT `None`-alpha error, are preserved. This is bounded branch coverage, not a
claim that every malformed input or every configuration is accepted.

The loader preserves original format precedence and registry order. Qwen DiT
aliases include fused `img_mlp.gate_up` half offsets. Qwen3-VL aliases bind to a
separate text owner, with separate model and CLIP strengths. Full original
parameter names are retained for the Unicode CRC-derived rounding seed.

Resident quantized weights are cast to the original LoRA compute dtype before
dequantization. Patch arithmetic uses the original intermediate dtype; the
patched weight is requantized with the existing layout/options, recalculated
scale, original key seed and `inplace_ops=True`. Logical original dtype is
restored. INT8 ConvRot is not silently converted to a permanently floating
weight. Static backup restoration permits LoRA A → LoRA B → no LoRA without
accumulating merges, including original `requires_grad=False` on restoration.

For selected static offload, `_load_list` sorting includes original patch math
workspace costs. GPU-resident parameters use the resident patch path; CPU
parameters retain original LowVramPatch execution after the cast, in the current
execution dtype. The latter BF16 arithmetic can differ from a resident FP16
patch. Consequently placement is numerically relevant when a LoRA is active.
Cross-device adapter casts now use original `empty_like` plus `copy_` semantics,
including Preserve memory format and nonblocking copies under the frozen
Linux/CUDA policy. Same-device casts retain the original copy flag behavior.

`qwen21-first-image` loads only the explicitly profiled, hash-checked adapters.
It records actual patch targets, order, strengths and offsets for separate text
and DiT owners. The strict comparator checks those records as well as tensors
and decoded PNG pixels; a zero-target LoRA test cannot pass unnoticed.

## Regression evidence

The saved original-source fixtures contain:

| Evidence | Cases |
| --- | ---: |
| Adapter/patch arithmetic, CPU/CUDA and FP32/FP16/BF16 | 284 |
| Loader formats, precedence and override records | 15 |
| Static plain/INT8/ConvRot patch and requantization | 24 |
| A → B → none backup/restoration lifecycles | 6 |
| Original patched-module memory costs | 4 |

Alias maps, Unicode key seeds and captured diagnostic strings are additional
assertions. Four new focused Rust tests cover these fixtures. The ordinary
Qwen suite now contains 29 tests. Standalone checks execute the new tests
without Python, a ComfyUI installation or model weights available inside the
namespace. The initial checkpoint passed all 29 focused tests in 166.96 s and
four standalone isolations. The allocation-order source subsequently passed all
29 in 152.86 s. After the final copy guard, four focused LoRA tests and four
standalone isolations pass again; the other 25 unchanged checks are reused.
Formatting, final all-target warnings-as-errors Clippy, nine Python AST checks
and shell syntax checks pass. Current hashes are recorded in the accompanying
[checkpoint](stage-2-qwen-image21-lora-checkpoint.json) and
[evidence directory](stage-2-qwen-image21-lora-evidence/).

The original helper correction was also recaptured: replacing the oracle's
initial blocking-policy stub with the actual `device_supports_non_blocking`
body and explicit frozen platform flags leaves all saved math/loader values
unchanged. The earlier failed compile, no-GPU sandbox attempts, stronger
restoration assertion failure and repaired errors remain labeled in the test
history. A failed attempt is not counted as a passing check.

## Initial trained functional comparisons

These are **320×224 functional tests**, not portrait-quality acceptance or
performance benchmarks. They use one synthetic reference, seed 3061383787,
Euler/simple, CFG 1, four INT8/BF16 steps or the existing eight Turbo sigmas.
The text encoder remains INT8 ConvRot and the VAE BF16. The accepted
1024×1024/40-step portrait and frozen 1600×896 controls are unchanged.

Real model LoRA: `loras/qwen-image-21/lenovo_qwen21.safetensors`, 79,743,896 bytes,
SHA-256 `7f1e95c3ec1e9c7cea197f7b6f80f601af8e6b479f6ead78ab16df542f447c0e`.
It supplies 384 tensors and patches 192 DiT parameters. The separate initialized
rank-2 text fixture patches one Qwen3-VL parameter. Its source generator and
33 KB safetensors file are preserved in `inference/fixtures/qwen21-lora/`.

The following table records the initial runs. The successful freed-memory
continuation and source corrections are recorded below.

| Run | Exact tensors / PNG | Status |
| --- | --- | --- |
| Automatic INT8 + model LoRA, strength 1 | Yes / yes | Selected functional pair passes |
| Automatic INT8 + text LoRA, CLIP strength 1 | Yes / yes | Selected functional pair passes |
| Automatic INT8 + same model LoRA twice, +0.375 then −0.5 | Yes / yes | Ordered stacking pair passes |
| Automatic BF16 + model LoRA | No / no | Failed; placement differs |
| Automatic Turbo + model LoRA | No / no | Failed; placement differs |
| BF16, Rust replays recorded Python placements | Yes / yes | Diagnostic only |
| Turbo, Rust replays recorded Python placements | Yes / yes | Diagnostic only |

INT8's six boundaries, text conditioning and all decoded PNG pixels match.
Turbo additionally compares explicit sigma bytes. Exactness is byte equality,
with finite-value checks; no relaxed tolerance has been introduced.

In the failed BF16 pair, the reference retained `transformer_blocks.20.attn.to_q`
and `to_out.0` on GPU while Rust placed them on CPU. Initial loaded bytes were
12,317,646,848 Python versus 12,250,537,984 Rust. The offload buffer was the same
96 MiB. Final latent values and 55,331 PNG pixels differed. Turbo also differed
in several patched attention matrices and 50,273 PNG pixels. Conditioning and
reference inputs remained exact. These failed automatic runs remain preserved.

The diagnostic file `reference-memory-plan.json` is accepted only in the
explicit development run's own directory and must match its complete profile.
Rust first executes normal automatic loading, then applies the recorded
reference resident set through the Rust patch/offload owner. Reports explicitly
set `diagnostic_memory_plan=true`; they do **not** establish automatic memory
selection parity. Both complete diagnostic outputs match the reference exactly,
which isolates placement-dependent rounding as an explanation of these runs.
This does not prove identical placement under every allocator input.

The observed pipeline seconds for the three passing automatic pairs were
9.696/5.932 (model INT8), 11.508/5.972 (text INT8) and 12.342/5.893 (stack INT8),
Rust/Python respectively. These first-process smoke timings include loading,
unloads and intermediate export, have no repeated performance control, and
precede the final copy-semantics correction. They establish no speed advantage.
The retained diagnostic timings include plan application and are not comparable
performance evidence.

## Initial memory gate and remaining scope

Approximately 3 GB was occupied before the automatic comparisons. Different
allocator/free-memory inputs selected different partial placements. Tomas chose
“Jag frigör GPU-minne själv” when asked to free roughly another 2–2.5 GiB.
At that capture he had not yet confirmed that memory was free, so new
automatic BF16/Turbo pairs waited. This gate was subsequently resolved below. No external process is stopped, inspected or changed.
The rerun must keep original automatic policy and exact comparisons. No fixed
placement, modified budget or relaxed arithmetic may be substituted as acceptance.

After the nonblocking-copy correction, three fresh final-source automatic INT8
pairs also pass: model LoRA, text LoRA and ordered stacking. Each matches all
tensor bytes, patch metadata and decoded PNG pixels in Python-free Rust
isolation. Their pipeline seconds are 9.430/6.157, 5.798/5.433 and 11.449/5.903,
Rust/Python respectively; these remain single functional runs, not performance
benchmarks. The two BF16/Turbo automatic failures and placement diagnostics
precede that correction. The automatic BF16/Turbo reruns were subsequently performed below with fresh
output names. Do not repeat the already
passing final INT8 controls unless a relevant source/input change justifies it.

General configurable-object/bypass patch integration, shape-changing patched
offload lifecycle, full model clone/shared-memory ownership and cache invalidation,
broader malformed loader inputs and arbitrary LoRA/model combinations remain
unaccepted. They are implementation/coverage gaps, not approved alternate
behavior. Broader Stage 2 precision, editing, sampler/CFG, OOM/tiling and product
integration requirements also remain open. No full LoRA or Stage 2 completion is
claimed, and no actual parity departure approval is pending.

## Replay and upgrades

With the existing local locked dependencies prepared, run bounded source oracles
through `inference/tools/launch_reference.sh` and `reference_lora.py`,
`reference_lora_weight.py`, `reference_lora_lifecycle.py`,
`reference_lora_reload.py` and `reference_lora_linear.py`, supplying `--site`,
`--comfy` and a new `--output`. Review source changes before capturing a new
reference; never overwrite an old expected result merely to pass an upgrade.
`check_qwen_image21.sh` runs the 29 focused tests and their standalone isolation.

Explicit trained runs are separate:

```bash
IMAGINE_PARITY_ROOT=/tmp/imagine-rust-comfy-parity-20261017 \
IMAGINE_REFERENCE_SITE=/home/thhel/git/ComfyUI/.venv/lib/python3.12/site-packages \
bash inference/tools/check_qwen_lora.sh \
  /home/thhel/git/ComfyUI /mnt/media/gemensam/AI/ComfyUI/models \
  FRESH_PREFIX int8 bf16 turbo text-int8 stack-int8
```

The comparator stops at the first failure. Ordinary tests do not load the real
Qwen weights. Keep this source, all three LoRA oracle fixtures, generators,
isolation rules, selected profiles and failed/successful evidence together for
future reference upgrades. Model weights remain external. No Git commit or
push is performed by this continuation without a new request.

## Freed-memory continuation and reload correction

Tomas confirmed: “Nu ska det vara frigjort så mycket jag kan”. Fresh unchanged
profiles observed 14,562,033,664 bytes free before BF16 and 14,564,327,424 before
Turbo, out of 16,769,744,896 total. Offload remained necessary. No process list
was read and no external process or service was changed.

Before the correction below, the new automatic BF16 pair passed all six tensor
boundaries and PNG pixels. Rust/Python retained 296/297 modules; their sole
physical-placement difference was the unpatched
`time_text_embed.timestep_embedder.linear_1`. This is exact output acceptance
for that execution, not a claim of identical physical allocator state.

The new automatic Turbo pair failed: 17,855 final latent values and 50,940 PNG
pixels differed. Its final resident-module lists were identical, yet loaded-byte
counters differed by 33,554,432 bytes. This exposed an additional Rust lifecycle
gap: physical residency alone does not describe ComfyUI's current patch path.

The unchanged original `ModelPatcher.load` can skip a formerly resident module
in a new fit calculation, retain its already merged GPU weight, and attach
`LowVramPatch` to that weight. A later `partially_unload` restores the backup
and **appends** another lazy function. Existing low-vram modules preserve their
function lists when `load` skips them. Recomputed loaded-byte accounting may
subsequently become negative after partial unload; it is not the physical
resident byte total.

Rust now retains lazy patch passes independently of physical residency, keeps
existing function order/count when original loading skips a module, appends on
partial unload and uses signed byte accounting. `Linear` applies those passes
in the original execution dtype. It preserves the original repeated application
rather than changing ComfyUI's arithmetic to make memory-pressure runs more
uniform. No quantizer, model strength, rounding, fitting threshold or autotuning
policy was changed.

Decision **S2-LORA-RELOAD-01**: these source-specific behaviors are required
compatibility fixes under Tomas's standing authorization. They are not approved
parity exceptions. If a future ComfyUI version changes them, capture and review
that new source separately; retain these old-reference cases.

`reference_lora_reload.py` executes original method bodies with three initialized
16×256 CPU weights, never a full model constructor. Three dtype sequences
(FP32/FP16/BF16), each containing initial load, reload, unload and another reload,
verify twelve states, raw/effective weights, function counts and signed counters.
All match exactly. The four focused LoRA tests pass; the complete 29-test Qwen
suite also passes in 154.19 s, and all-target warnings-as-errors Clippy passes.

The development memory-plan replay now requires recorded lazy-function state;
old resident-only plans fail clearly instead of silently omitting it. Earlier
successful diagnostics remain historical evidence for their earlier source.
Normal automatic profiles do not read a replay plan. Reference observation now
records the additional state without changing the original model methods.
Before-correction source/report snapshots and the successful BF16/failed Turbo
runs are retained beside the new regression evidence.

The first rerun after function-state repair passed automatic Turbo exactly, but
BF16 failed with a different lazy-function choice for
`transformer_blocks.15.attn.to_out.0`. Its initial resident set and counters
matched; during sampling Rust retained one lazy pass on that already resident
weight while Python did not. That failed pair is retained, not counted as
acceptance of the final implementation.

A further source review found a concrete allocation-order mismatch. Original
`load` first fits modules by offload cost, then separately sorts the regular
load list by **module bytes and name**, patches the complete selected list,
and finally moves modules in a second traversal. Rust had combined patching
and moving while traversing offload-cost order. With LoRA the two cost orders
can differ, affecting the allocator state seen by the next automatic load.
Partial unload likewise must restore/move/pin in ascending offload-cost order.

Decision **S2-LORA-LOAD-ORDER-02**: preserve the original two-phase regular-load
order and partial-unload traversal. This is another required source-fidelity
repair, with unchanged budgets, arithmetic and autotuning. A small original-code
case mixes three patched 16×256 weights with two initialized unpatched 32×256
weights. Its exact patch/move trace now drives a regression assertion against
the same action planner used by Rust loading. The earlier twelve expected
reload states are byte-for-byte unchanged. The intermediate source snapshots,
failed BF16 pair and successful Turbo pair are preserved separately.

The three prior INT8 model/text/stack controls exercise unchanged resident
quantized paths. They are reused as such; they are not relabeled fresh model
runs of the later offload correction. Fresh automatic BF16/Turbo checks and
bounded offload regressions validate the changed path.


## Final copy guard and selected acceptance

Decision **S2-LORA-COPY-03**: follow original `cast_bias_weight`'s
`copy=weight_has_function` / `copy=bias_has_function`. A resident lazy patch must
operate on a copy so that denoising calls do not mutate and progressively merge
into the stored weight. CPU-to-GPU offload already creates fresh storage. Rust
now requests copying on the same-device lazy branch as well; calls without lazy
functions retain their existing arguments. This is required original behavior,
not a parity exception or a new optimization policy.

`reference_lora_linear.py` captures original `cast_bias_weight`, LowVramPatch
and F.linear execution for twelve tiny CPU/CUDA × FP32/FP16/BF16 × one/two-pass
cases. Each performs two forwards and verifies that stored weights and biases
remain unchanged. The actual Rust Linear path matches all 24 forward outputs
and all retained parameters exactly. The final focused LoRA rerun passes four
tests in 13.55 s; all-target warnings-as-errors Clippy passes on that source.
The full 29-test suite passed after allocation-order repair in 152.86 s; the
other 25 unchanged cases are reused after this copy-only guard. Final standalone
isolation evidence is recorded in the checkpoint.

After the allocation-order correction, fresh automatic BF16 and Turbo pairs
pass the strict comparator with **no recorded-plan replay**. The six tensor
boundaries, patch metadata and all PNG pixels are exact; Turbo's sigma bytes
also match. Each pair patches the same 192 DiT parameters in both engines.

| Automatic pair | Steps / output | Rust / Python pipeline | Rust / Python sampling |
| --- | --- | --- | --- |
| BF16 DiT + LoRA | 4 / 320×224 | 8.945 / 17.554 s | 0.712 / 0.705 s |
| Turbo BF16 DiT + LoRA | 8 / 320×224 | 21.672 / 20.954 s | 1.190 / 1.228 s |

These are single functional pairs with variable loading times, not performance
or portrait-quality benchmarks. The text encoder stays INT8 ConvRot and VAE
BF16. No previous 1MP portrait or 1600×896 benchmark was repeated or replaced.

Both phases in each measured pair have matching physical resident sets and
matching lazy-pass counts for all patched modules. No measured resident patched
module has a lazy function. Therefore the final copy guard's new branch is not
executed by these measured profiles: their exact results and timings are reused
explicitly, rather than described as fresh runs of the guard. The new branch
has the separate twelve-case actual-Linear evidence above. The earlier three
INT8 model/text/stack controls likewise retain unchanged execution paths.

BF16's final reported loaded-byte counters still differ by 1,048,576 bytes
(13,121,904,640 Rust / 13,120,856,064 Python), despite equal physical resident
sets. Turbo's final counters are equal. Counter accounting, physical residency
and lazy functions are separate observations; these pairs do not prove identical
allocator inputs or unrestricted shared-memory behavior. The bounded original
counter/selection cases retain exact comparisons under identical inputs.

All five selected LoRA profile kinds now have automatic exact functional
comparison evidence. General clone/config/bypass, shape-changing offload and
broader memory/model acceptance remain open as listed above. No deviation,
relaxed tolerance, changed autotuning or production service change was adopted.
No new approval is pending for this continuation. Changes remain uncommitted.
