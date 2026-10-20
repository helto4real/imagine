# H3 trained comparison: phase and memory plan

This is a pre-load plan, not measured trained memory fit. The selected four
files remain those in `native-av-baseline-plan-20261020.md`; neither engine may
substitute the earlier AWQ encoder. Full-file SHA-256 locks must be captured
before loading their trained tensors. Header locks and quant records alone do
not establish full-file identity.

The actual original constructors and meta load/state inventories give:

| Component | File bytes | Loaded module bytes | Compute |
| --- | ---: | ---: | --- |
| Text/vision | 26,363,476,151 | 26,363,248,757 | Original full-precision text policy, FP32 |
| DiT | 20,970,379,616 | 20,967,513,920 | BF16/INT8 |
| Video VAE | 3,171,670,912 | 2,795,667,280 | FP16/INT8 |
| Audio VAE | 605,254,808 | 605,149,280 | FP32 |

These are physical source module accounting values from
`inference/resources/h3-component-modules.json`, including original buffers and
extra parameter owners. They are not activation, workspace or peak-RSS estimates.
For example, a text FFN matrix has 131,072,000 INT8 elements; its FP32 dequantized
temporary alone is 524,288,000 bytes. The original text full-precision setting
must be retained, rather than borrowing the DiT policy to fit memory.

## Process phases

Run each engine's phases sequentially under the canonical Ticket controller.
Use independent processes for text, optional reference encode, DiT, video decode
and audio decode. Only small explicit own tensor artifacts cross phases. Wait
for the owned PID namespace and launcher to exit before the next lease/phase.
Never retain text and DiT trained owner graphs simultaneously.

The native loader uses the existing read-only AIMDO mapping and common component
normalization, including legacy metadata precedence. Retain its storage and
metadata owners through preparation; do not duplicate entire checkpoints or
cast their raw payload wholesale. Use the H3 constructor and offload inventory,
with the actual sampling sigmas buffer added to the DiT owner. Reuse the common
static ModelPatcher/memory arithmetic and synchronous source cast path. Do not
apply Qwen's architecture-specific memory estimate to H3.

Query current available host memory/cgroup limits and source GPU free memory
inside the owned phase before placing a model. Record actual budgets, placement,
peak memory and elapsed time. An OOM is retained as a failed gate; no silent
dtype, resolution, model, sampler or backend fallback is permitted. Fully reap
the failed/cancelled phase before testing the next image's recovery.

## Initial bounded workload

The proposed first profile is batch 1, 384×256, 39 frames at 24 fps, seed 42,
four Euler/simple steps, source shifts 12/3 and CFG 1. There is no LoRA or second
stage. First use no references, then exactly one own synthetic image reference.
This is a technical probe; visual quality is unmeasured.

The original full video VAE's meta shape calculation confirms video latents
`[1,24,12,16,24]` decode to `[1,3,39,256,384]`. Original AV latent initialization
gives audio `[1,32,2,65]`. At the source 800 PCM samples per latent and 32 kHz,
audio has 52,000 samples: both streams last exactly 1.625 seconds. Verify these
actual output shapes and time bases after trained decode; this prediction does
not replace that gate.

The two-stream sampler carries audio scaled by 4 while denoising, undoes it at
the original boundary, and preserves the original CPU noise generator order.
Keep noise, sigmas, conditioning/tags, every sampler state, final latents, video
frames and PCM in separate provenance/comparison records. Export/container work
and product integration remain with the video owner.

## Acceptance sequence

Finish bounded synthetic CUDA comparisons before these model-sized loads. Then
compare identical SHA-locked files in both engines, phase by phase, retaining
failed evidence and without relaxing the numerical criterion. Exercise cold,
repeat, failure, cancellation, unload and subsequent-image recovery. A second
H3 baseline requires the actual short AV clip and synthetic-reference case, not
only constructor, CPU or interface tests.
