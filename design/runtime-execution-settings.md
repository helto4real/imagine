# Runtime execution settings

The operator chooses how jobs run; a change applies to jobs started after it.
Decided by Tomas on 2026-10-29, after Combo/Influence on GPU
([h3-boundary-memory-20261029.md](h3-boundary-memory-20261029.md)).

## Setting

`GET` and `POST /v1/runtime/execution` (session guard, as the other runtime
routes) read and replace
`{"attention": "auto" | "kitchen_int8" | "pytorch_sdpa", "dynamic_vram": true | false}`.
Unknown values or fields fail `invalid_execution_settings` (422). The server
keeps the choice in its private storage (`library/runtime-execution.json`), so
it survives restarts and applies to every client: the native UI, MCP and API
clients. The native UI shows it under Settings, Execution.

## Attention

Modes follow ComfyUI's own behavior:

| Mode | Base attention | ComfyUI equivalent |
| --- | --- | --- |
| `auto` (default) | Comfy Kitchen INT8 where the GPU supports it, else PyTorch SDPA | started with `--use-ck-attention` |
| `kitchen_int8` | Comfy Kitchen INT8; an unsupported GPU fails the job | started with `--use-ck-attention` |
| `pytorch_sdpa` | PyTorch SDPA | started without the flag |

The base applies to every component of a job (text encoders, VAEs and the
diffusion model), as ComfyUI's global attention does. A checkpoint's own
`ComfyAttention` preference still applies per block in every mode, as in
ComfyUI. Of the current models only MOSS carries one (42 blocks,
`comfy_kitchen_int8`); Qwen and H3 carry none, so `pytorch_sdpa` runs them on
PyTorch SDPA throughout. Kitchen INT8 needs compute capability 7.5 or higher.
An unsupported explicit choice fails `attention_unavailable` (images) or
`h3_attention_unavailable` (video) before any model loads.

## Dynamic VRAM

On (the default, as in ComfyUI), the diffusion model of an H3, Qwen or MOSS job
keeps its weights on the host and brings each cast module (`Linear`) into a
comfy-aimdo VBAR range when it runs, releasing ranges under memory pressure, as
ComfyUI's `ModelPatcherDynamic`. Modules register in its load order: a model's
dynamic units, then `_load_list(for_dynamic=True)` sorted; these diffusion
models define no units. LoRA patches are bound as low-VRAM patches that `Linear`
applies where it uses the weight; a patch on any other weight fails the load by
name rather than being dropped. With it on, the Qwen and MOSS text encoders
are dynamic too, as ComfyUI's text encoders are with dynamic VRAM; H3's text
encoder always was. Off places the diffusion model and the image text encoders
statically before use, as before. VAEs stay static in both modes.

The general preparation lives in the inference core (`checkpoint/dynamic.rs`),
so `native/product/build-lock.json` moved `core_revision` to the commit that
contains it and updated that file's hash. The parity fixes below changed more
core files and added four to the lock's core list (`qwen_image21/denoiser.rs`,
`qwen_image21/host_cache.rs`, `moss_image/text.rs`, `moss_image/execution.rs`),
each unchanged from `source_base` before the fix.

## Each job

A job reads the choice once, when it starts on the worker, and that value is
both sent to the worker and recorded: as `execution` in an image job's `output`
and in a Take's `provenance`. The H3 worker protocol is version 18 for the new
request field and error code; the image worker reads the field from its request
header.

## Verified

885 workspace tests (synthetic), Clippy and formatting pass. Through the proxy
on the RTX 4070 Ti SUPER, with one synthetic prompt and seed: Qwen with
`pytorch_sdpa` reproduced the earlier PyTorch image byte for byte (74.2 s);
Qwen and MOSS with `auto` reproduced the earlier Kitchen images byte for byte
(Qwen 48.1 s with loaded models, 62.1 s as the first job after a restart); an
H3 scene with `auto` succeeded (80.1 s). Each job's `output` or Take
`provenance` carried the choice it ran with.

Dynamic VRAM, through the proxy on the same GPU: an H3 scene and the Qwen and
MOSS images were byte-identical with dynamic VRAM on and off. Combo and
Influence at 640×352 with a 56-frame boundary tail succeeded with it on (120 s
each) and ran out of memory with it off, as before.

Cold and warm starts on 2026-10-29, same GPU, `auto` attention, one synthetic
prompt and seed: cold is the first job after a server restart, warm the same
job straight after it. Times come from the engine's phase changes; free GPU
memory is the lowest `nvidia-smi` reading during the job and RSS the worker's
peak resident memory.

| Model | Dynamic VRAM | Start | Total | Load and text encoding | Placing the DiT | Sampling | Free GPU | Worker RSS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen | on | cold | 71.8 s | 13.4 s | — | 56.1 s | 4195 MiB | 9798 MiB |
| Qwen | on | warm | 57.1 s | 5.1 s | — | 50.1 s | 4184 MiB | 11176 MiB |
| Qwen | off | cold | 64.9 s | 13.2 s | 6.9 s | 42.4 s | 4086 MiB | 10008 MiB |
| Qwen | off | warm | 48.2 s | 5.4 s | 1.0 s | 40.4 s | 4074 MiB | 11243 MiB |
| MOSS | on | cold | 54.6 s | 14.4 s | — | 39.1 s | 3664 MiB | 10418 MiB |
| MOSS | on | warm | 40.1 s | 5.7 s | — | 34.1 s | 3628 MiB | 11362 MiB |
| MOSS | off | cold | 48.6 s | 13.9 s | 5.1 s | 27.8 s | 3664 MiB | 10404 MiB |
| MOSS | off | warm | 32.7 s | 5.4 s | 0.8 s | 26.2 s | 3628 MiB | 11372 MiB |

Each model gave the same image in all four runs. Dynamic VRAM cost 6.0–8.9 s
per image (11–23 %), in sampling rather than loading, and the cost did not
shrink when warm, while free GPU memory and worker RSS stayed within about
200 MiB of the static runs. These jobs fit statically, so dynamic VRAM bought
them nothing. The cause was a defect, not residency or staging: see
[Parity with ComfyUI](#parity-with-comfyui-2026-10-30).

ComfyUI's own code for the same profiles, prompt and seed
(`tools/image-python-reference.py`, checkout `426f5eea`, `--use-ck-attention`,
under the coordinator's reservation without network): one process loads the
profile's three files and runs the graph twice, cold and then warm with the
models kept, as a repeated ComfyUI prompt. Its encoding phase includes loading
the text encoder; the static diffusion model is placed inside sampling.

| Model | Dynamic VRAM | Start | Encoding | Sampling | Decode | Free GPU | Process RSS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen | on | cold | 2.9 s | 37.4 s | 0.9 s | 262 MiB | 2547 MiB |
| Qwen | on | warm | 0.8 s | 35.4 s | 0.7 s | 593 MiB | 2553 MiB |
| Qwen | off | cold | 3.7 s | 39.4 s | 1.1 s | 1122 MiB | 8897 MiB |
| Qwen | off | warm | 3.4 s | 37.6 s | 0.8 s | 1070 MiB | 11254 MiB |
| MOSS | on | cold | 3.2 s | 25.3 s | 0.5 s | 272 MiB | 2927 MiB |
| MOSS | on | warm | 0.5 s | 23.3 s | 0.2 s | 265 MiB | 2935 MiB |
| MOSS | off | cold | 3.9 s | 26.0 s | 0.4 s | 2806 MiB | 7788 MiB |
| MOSS | off | warm | 2.5 s | 25.7 s | 0.2 s | 1708 MiB | 8781 MiB |

In ComfyUI dynamic VRAM is not a cost: it samples as fast as static placement
or slightly faster (0.7–2.4 s), uses the device down to a few hundred MiB and
keeps the process at 2547–2935 MiB resident against 7788–14730 MiB static
(the last at its peak). Each
model's decoded pixels were identical with it on and off. Imagine's dynamic
sampling took 42–55 % longer than ComfyUI's (Qwen 56.1/50.1 s against
37.4/35.4 s, MOSS 39.1/34.1 s against 25.3/23.3 s), while Imagine's static
placement and sampling were within
1–10 s of ComfyUI's static sampling. The gap is in Imagine's dynamic port, so
an automatic choice would hide it; the port should be brought to ComfyUI's
behavior instead. The reference sets comfy-aimdo's log level to warning, as
`main.py` sets one; with it, decoding completed in every run. The RSS columns
are not comparable across the two: Imagine's counts the mapped checkpoint
pages it touches, and how comfy-aimdo's host buffers appear in RSS is
unverified.

## Parity with ComfyUI, 2026-10-30

A temporary build (not committed) wrote integer counters to a file of my own
choosing: VBAR acquires, resident hits, copies and bytes per denoising step,
and the time of each phase. It showed the Qwen diffusion model fully resident
after step 1 (168 modules, 5.4 GB, no copies afterwards), so neither
residency nor host staging explained the slower steps. Two runs per step of
one row each matched ComfyUI, which does not batch these conditions either
(their contexts have 26 and 56 tokens).

A small CUPTI kernel tracer, preloaded into the image worker and into the
ComfyUI reference, showed identical kernels with identical durations in both.
ComfyUI's GPU was busy 34.6 s of its 36.1 s sampling window; Imagine's 33.0 s
of 43.9 s. The gaps were host-to-device copies of exactly 48 MiB, one per
block per forward (2600 copies, 8.3 s): `Linear::input_act`, the fused SwiGLU
INT8 path, read the module's host weight instead of its VBAR lease, so each
call copied the weight to the GPU. With dynamic VRAM off the weight is
resident and the path had no gaps. It now takes the lease, as `forward` and
ComfyUI's `cast_bias_weight` do.

Other changes, each measured on the same synthetic prompt and seed:

- **Text encoders on the dynamic path.** Qwen's static text encoder cost 1.3 s
  to place and 3.2 s to unload per job; as a VBAR model it encodes both
  prompts in about 1.4 s and drops in a fraction of a second. MOSS's text
  embedding and gated-delta `conv1d` now take their leases, so its text
  encoder runs dynamically as well (warm encoding 6.7 s before, 1.5 s after).
- **Free memory counts what AIMDO could release** (`vbars_analyze`), as
  `ModelPatcher.get_free_memory` does, for CFG batching (Qwen, MOSS) and the
  Qwen prefix cache. It does not change these text-to-image jobs, whose
  conditions cannot be concatenated; it follows ComfyUI where they can.
- **comfy-aimdo 0.6.0** (ComfyUI's current version) was tried in place of the
  locked build: no speed change for images, and the H3 worker failed every job
  with `h3_protocol_error`. The lock keeps the earlier library.
- **ComfyUI's compiler and prefetch queue** were ruled out on ComfyUI's side:
  with `--disable-comfy-compiler` or `--disable-async-offload` its Qwen
  sampling stayed at 35.7–36.0 s.

Results through the proxy on the RTX 4070 Ti SUPER, Kitchen INT8 (`auto`),
after the fix. Cold is the first job after a server restart with the model
files in the OS page cache; warm repeats it.

| Model | Dynamic VRAM | Cold | Warm | ComfyUI cold / warm |
| --- | --- | --- | --- | --- |
| Qwen | on | 41.8 s | 40.8 s | 41.6 / 36.8 s |
| Qwen | off | 47.5 s | 45.0 s | 44.7 / 41.8 s |
| MOSS | on | 25.8 s | 25.8 s | 29.5 / 24.0 s |
| MOSS | off | 32.5 s | 31.5 s | 30.8 / 28.4 s |

Imagine's totals include its job overhead and PNG output, ComfyUI's only
load, encoding, sampling and decoding. Every image was byte-identical to the
earlier ones (Qwen `75ba1036`, MOSS `a4c2e962`), and dynamic VRAM is now faster
than static placement, as in ComfyUI. A first job whose files are not in the
page cache reads them from the model disk: 65.3 s for Qwen and 36.3 s for MOSS
on this machine; with 64 GB of RAM the page cache does not hold Qwen, MOSS and
H3 together, so switching families rereads them. The 80 s H3 scenes measured
earlier were such jobs. With its files cached, the H3 source scene took 35.4 s
(text 5.0 s, sampling 27.3 s; ComfyUI's code: conditioning 7.0 s, sampling
25.6 s) and Combo 73.5 s (text 8.2 s, reference video 4.7 s, sampling 57.3 s;
ComfyUI: 11.6 s and 61.5 s), with the earlier source video (`4e19f788`).

The fixes left the results of the inference parity suite (`tests/primitives`,
on the GPU under the coordinator) unchanged: 442 passed and the same 17 failed
before and after, all for parity fixtures this environment does not provide.

## Not yet included

Keeping models, VBARs and host pins between jobs, as ComfyUI does between
prompts: its warm text encoding is 0.5–0.8 s against Imagine's 1.5 s, because
its text encoder stays loaded. Pinned host staging for dynamic layers matters
once models stay loaded and evict each other; on its own it would only speed
up the first step. A test matrix
of both attention modes for Qwen, MOSS and H3 at Draft quality, comparing time,
memory and output.
