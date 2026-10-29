# Runtime execution settings

The operator chooses how jobs run; a change applies to jobs started after it.
Decided by Tomas on 2026-10-29, after Combo/Influence on GPU
([h3-boundary-memory-20261029.md](h3-boundary-memory-20261029.md)).

## Setting

`GET` and `POST /v1/runtime/execution` (session guard, as the other runtime
routes) read and replace `{"attention": "auto" | "kitchen_int8" | "pytorch_sdpa"}`.
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

## Not yet included

Dynamic VRAM (comfy-aimdo) for the H3 and image diffusion models, as a second
choice in the same setting, on by default; and a test matrix of both choices
for Qwen, MOSS and H3 at Draft quality, comparing time, memory and output.
