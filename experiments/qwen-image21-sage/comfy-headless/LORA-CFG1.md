# Matched ComfyUI LoRA comparison — 2026-10-16

The installed ComfyUI/AIO engine completed the same eight-request sequence.
Warm times with Lenovo LoRA were **20.360 / 21.192 s**, mean **20.776 s**.
This compares with the latest native candidate's **44.611 s** flag off and
**39.892 s** flag on. The latter is **1.920×** Comfy's time; Comfy took **47.9%
less time** in these measurements. These are local observations with different
measurement instrumentation, not a precision-equivalent engine comparison.

## Workload and boundaries

Same physical diffusion, encoder and VAE files under
`/mnt/media/gemensam/AI/ComfyUI/models`:

- `diffusion_models/qwen-image2.1/qwen_image_2.1_int8_convrot.safetensors`
- `text_encoders/qwen-image2.1/qwen3vl_8b_int8_convrot.safetensors`
- `vae/qwen-image2.1/qwen_image_2.1_vae_bf16.safetensors`
- Only `loras/qwen-image-21/lenovo_qwen21.safetensors`, strength 1.

1600×896, 40 ER-SDE/beta steps, CFG 1, seed 42, batch 1, empty negative,
no reference images. The woman-portrait fixture is taken from the native
benchmark's **source constant**, not from stored requests or history. Comfy's
logical `clip/` path resolves to the physical `text_encoders/` file.

Comfy uses the installed AIO adapter directly in one persistent headless process,
Kitchen INT8 attention, automatic memory/cache policy, default cache dtype and
Comfy sampling schedule. This is real Comfy model loading, conditioning, sampling
and VAE execution, without browser, HTTP server or graph-executor/UI overhead.
The AIO adapter is called anew per request, retaining process-local model and
encoding caches, matching the previous Comfy comparison method.

Comfy totals end after PNG encoding in memory; images are then discarded. Native
totals include PNG/base64 encoding and synchronous HTTP response transport.
Comfy phase timers synchronize CUDA but **no kernel profiler is attached**.
The reused native candidate numbers have CUPTI enabled identically off/on.
The earlier unprofiled frozen native run with LoRA was 44.493 s, consistent with
the 44.611 s profiled control; there is no new unprofiled fused-native run here.
Do not interpret the exact ratio as isolated kernel speedup.

Both engines receive the same seed and named sampler/scheduler; RNG, numerical
precision, quantization, attention and sampler implementations differ. Generated
images were not inspected or compared, so identical pixels/quality are not
claimed. No inference optimization or installed Comfy/AIO source was changed.

## Measured times

| Request | Comfy total, s | Comfy sampling, s |
| --- | ---: | ---: |
| Without LoRA, first | 27.428 | 21.064 |
| Without LoRA, warm 1 | 21.563 | 19.402 |
| Without LoRA, warm 2 | 21.621 | 19.475 |
| With LoRA, first | 20.492 | 19.044 |
| With LoRA, warm 1 | 20.360 | 18.904 |
| With LoRA, warm 2 | 21.192 | 19.148 |
| LoRA removed, first return | 21.719 | 19.758 |
| LoRA removed, warm return | 21.257 | 19.660 |

| Engine | Without LoRA, warm mean | With LoRA, warm mean |
| --- | ---: | ---: |
| ComfyUI/AIO, this unprofiled run | 21.592 s | 20.776 s |
| Native candidate, flag off, previous CUPTI run | 20.542 s | 44.611 s |
| Native candidate, flag on, previous CUPTI run | 20.297 s | 39.892 s |

Comfy's with-LoRA warm sampling mean is 19.026 s; without LoRA it is 19.438 s.
This small inverse difference is not evidence that LoRA accelerates Comfy.
The key observation is the absence of the native engine's large LoRA penalty.
The sample count is two warm repetitions per case, measured sequentially.

## Verification and reproducibility

All eight requests completed 40 steps and verified Kitchen INT8 selection.
An additional fresh-process LoRA run asserts that the incoming model has no
LoRA patches, the loader adds nonempty patches and changes the patch identity,
and the prepared diffusion model retains those patches. It completed all 40
steps in 25.326 s including first conditioning/PNG (excluded from warm means).
This verifies that the supplied LoRA was not silently skipped; it does not
establish pixel parity or complete per-key adapter coverage across engines.

Installed Comfy revision: `ba994879b88d0ebd9950b1e5c1e923d928f216ba`, clean tracked
source. AIO HEAD: `52ecbf87dda674f51e1e0a798f8f56d0d37d303d`, with preexisting local
changes, including Qwen adapter/service/dimensions. Selected executed source
hashes are retained in [numeric results](LORA-CFG1-RESULTS.json); nothing was
reset. Torch 2.13.0+cu130, CUDA 13.0, Kitchen 0.2.37, 23 Torch CPU threads.
Native used 11 threads and its unchanged frozen settings.

Run [compare-lora-cfg1.py](compare-lora-cfg1.py) using the installed Comfy Python:

```sh
/home/thhel/git/ComfyUI/.venv/bin/python \
  experiments/qwen-image21-sage/comfy-headless/compare-lora-cfg1.py \
  --comfy-root /home/thhel/git/ComfyUI \
  --aio-root /home/thhel/git/comfyui-all-on-one-image-generation-node \
  --model-root /mnt/media/gemensam/AI/ComfyUI/models --run
```

Omit `--run` for a source-only plan. Add `--verify-lora` for the extra application
assertion run. Both invocations wait for an idle GPU and stop no existing service.
Owned Python processes exited normally. GPU sandbox escalation was approved.
Stdout/stderr from libraries were discarded before imports; no server logs,
existing runtime prompts/history, generated image files or private configuration
were read. Only newly authorized timing measurements were exposed. PNGs were
encoded and discarded in memory, never saved or visually inspected.

Raw numeric runs, ignored by Git: `runs/comfy-lora-cfg1-paypkc8q/` and
`runs/comfy-lora-cfg1-3_rer5yt/` (verification). Python syntax and `git diff --check`
pass. This benchmark/report is uncommitted; no commit or push was requested here.

Native evidence: [LoRA output comparison](../profiling/LORA-OUTPUT.md) and
[unprofiled frozen control](../profiling/LORA-CFG1.md).
