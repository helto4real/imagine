# ComfyUI / AIO / Kitchen reference at 4096×2304

Date: 2026-10-14. This is a headless reference run, not a new optimization.

## Protocol

Call the installed QwenImage21Adapter directly in one process: one initial image
and one warm repeat. No ComfyUI HTTP server, graph executor or UI is started.
Use the frozen synthetic prompts, seed 42, 4096×2304, max side 4096, 16:9,
multiple 32, 40 ER-SDE/beta steps, CFG 2.5, batch one. No LoRA, reference image,
second pass or color processing. Kitchen INT8 attention, memory policy auto,
prefix cache auto/default, Comfy sampling schedule and real negative conditioning.

The three model files resolve to the same physical files as the native test.
The prompt hashes match the frozen reference. The installed ComfyUI/AIO revisions
remain 08ff3c11b3a85eb07c2c06f21cb378ac2e086840 and
52ecbf87dda674f51e1e0a798f8f56d0d37d303d. Torch is 2.13.0+cu130, Kitchen 0.2.37.
Comfy uses its normal 23 Torch CPU threads; the native profile uses 11. Comfy VAE
executes in BF16. Native uses its accepted F32/F16 VAE path. Identical settings
therefore do not imply identical kernels, precision or pixels across engines.

The timer starts at adapter invocation and ends after PNG saving returns. It
includes preparation, conditioning, sampling, decode and CPU conversion/save,
but excludes process startup and fsync. Native timers include local HTTP and
base64 transfer too. Coarse phase timers synchronize CUDA. Torch profiling is
disabled; lightweight existing dispatch counters remain enabled.

New compiler caches and GPU samples are confined to the experiment directory.
No private ComfyUI logs, workflow, history or settings are read. No installed
ComfyUI/AIO source is modified, and no other GPU applications are stopped.
GPU access required escalation for this explicitly authorized test.

Local run: `runs/20261014T212510Z-comfy-v2v_xd9q`.
Driver log: `runs/comfy4096-reference-build/driver.log`.

## Reproduction

```sh
"$COMFY_ROOT/.venv/bin/python" experiments/qwen-image21-sage/comfy-headless/run.py \
  --comfy-root "$COMFY_ROOT" --aio-root "$AIO_ROOT" --model-root "$QWEN_MODEL_ROOT" \
  --width 4096 --height 2304 --runs 2
```

The harness now takes explicit checkout/model paths and dimensions, and reads
frozen prompt copies instead of deleted historical run files. It checks 40
steps, 80 forwards, effective Kitchen mode, output size and repeat-pixel identity.

## Results

Both requests completed at 4096×2304 with 40 steps, 80 forwards and real
Kitchen CUDA dispatch: each recorded 2624 `sage_sdpa_prequantized` calls and
15744 successful CUTLASS INT8 dequant calls. Both saved images have identical
RGBA pixels, hash:
`aaf0d9f38c7a322c1f5567b3bc81d6067088d6ff1a83783a1d92b3d8b811f00b`.
They differ from the native engine's reference; this is expected across these
different execution paths and is not a claim of visual equivalence.

| ComfyUI phase | First | Repeat |
| --- | ---: | ---: |
| Prepare models | 0.386 s | 0.316 s |
| Conditioning | 3.008 s | 3.140 s |
| Sampling | 458.917 s | 450.974 s |
| VAE decode | 7.976 s | 7.470 s |
| Adapter entry to saved PNG | 473.873 s | 465.642 s |
| Torch peak allocated (whole generation) | 10617.80 MiB | 10616.92 MiB |
| Torch peak reserved (whole generation) | 13744 MiB | 12188 MiB |

**Cache qualification:** the repeat is a warm process, not a prompt-cache hit.
The installed AIO cache defaults to a 300-second TTL, and sampling alone exceeds
that. Both calls report two conditioning encodes and zero conditioning hits.
This differs from the native repeat, whose CPU conditioning entries are reused.
No TTL was changed to improve the benchmark.

Both decodes finished without the ComfyUI regular-decode-to-spatial-tiling
fallback warning. Each emitted an internal CUDA allocation-failure warning and
recovered; there was no failed image request. The inspected `VAE.decode` path
logs its spatial fallback explicitly. This evidence does not exclude internal
chunking inside the VAE or lower-level convolution implementations. Its BF16
VAE and Torch memory figures must not be equated to native F32/F16 graph-arena
measurements. Selected ComfyUI/AIO source hashes are unchanged after the run.

## Comparison with the immediately preceding native runs

| Repeat metric | Frozen native | Dynamic native | ComfyUI/Kitchen |
| --- | ---: | ---: | ---: |
| Sampling | 474.58 s | 638.29 s | 450.97 s |
| VAE | 32.19 s | 32.28 s | 7.47 s |
| Start to saved PNG | 508.18 s | 671.94 s | 465.64 s |

ComfyUI is 42.53 seconds faster than the frozen native repeat in this series.
Its VAE is 24.72 seconds faster (about 4.3× the native decode throughput for
this image). Sampling is also 23.61 seconds faster, partly offset in total time
by conditioning and conversion/saving overhead. Do not add nested phase timers
or treat unsynchronized individual operation counters as GPU durations.

The dynamic native series had much more diffusion segmentation and repeated
weight loading; its larger gap is not isolated evidence of a VAE-kernel slowdown.
These are sequential runs, one initial and one repeat per engine, not a
counterbalanced statistical benchmark. No new native runs were made here.
The result identifies VAE decoding as a concrete remaining difference; it does
not by itself establish which precision, convolution, memory-placement or
chunking choice causes it.

Compact evidence is in `evidence/comfy4096-reference/`. It contains sanitized
configuration, coarse timing results, dispatch counts, pixel hashes and selected
source hashes. Raw images/logs and compiler caches remain ignored locally.
Harness syntax and diff checks passed; live assertions passed. No installed
inference code, native baseline, commit or remote repository was changed.
