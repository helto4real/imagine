# Qwen INT8 single-LoRA comparison — 2026-10-16

Follow-up: [kernel-level fusion attribution](LORA-FUSIONS.md) repeats this exact
workload with authorized diagnostic profiling and unchanged optimizations.

User-requested investigation of roughly doubled generation time with one small
LoRA. This is a new workload, not a replay of the no-LoRA CFG 2.5 acceptance.
No native source, production setting or frozen executable was changed.

## Setup

- Frozen `qwen-image21-1600x896-v2` server; SHA-256
  `9fd49b0e5cb3426043ebd85271ad167555551ac65252590cde5ac25bed44bbb0` verified.
- Qwen Image 2.1 INT8 ConvRot, INT8 encoder and the frozen Qwen VAE.
- 1600×896, 40 ER-SDE/beta steps, **CFG 1**, seed 42, CPU RNG, batch 1.
- One source-defined photographic portrait prompt of an adult woman in a blue
  shirt; identical prompt in every request, empty negative prompt, no references.
- Only `lenovo_qwen21.safetensors`, strength **1.0**, versus an empty LoRA list.
  File size: 79,743,896 bytes (76.05 MiB). The file contents were not inspected.
- Frozen optimization environment, auto-fit and conditioning cache capacity 4.
  Diagnostic logging was suppressed in both cases.
- One dedicated native process throughout all eight requests. The Rust API/UI
  and its process-replacement logic were bypassed. No existing service was stopped.
- GPU free-memory/utilization checks preceded startup, after the user stopped
  their GPU job. The test-only startup gate is not a production memory setting.

## Results

Client wall time for the synchronous native HTTP request, including inference,
image encoding and response transport. Responses were discarded to `/dev/null`.
No server logs, generated images, job records or their metadata were read.

| Case, in execution order | Seconds |
| --- | ---: |
| Without LoRA, first request | 22.877 |
| Without LoRA, warm 1 | 19.907 |
| Without LoRA, warm 2 | 19.946 |
| With LoRA, first request | 45.307 |
| With LoRA, warm 1 | 44.095 |
| With LoRA, warm 2 | 44.891 |
| LoRA removed, first return request | 20.758 |
| LoRA removed, warm return request | 19.904 |

Warm means: **19.926 s without**, **44.493 s with**, or **2.233× / +123.3%**.
All eight requests returned HTTP 200. Native startup to HTTP readiness was
0.416 s, measured separately. The owned native process was stopped afterwards.

The slowdown persists on repeated requests without restarting the engine and
reverses when the LoRA is removed. It cannot be explained solely by the Rust
wrapper restarting the native process when the selected files change.

This small series does not identify the extra time by stage, measure peak VRAM,
validate image quality, or establish behavior for other LoRAs/reference editing.
Source inspection suggests an additional candidate beyond LoRA matrix work:
the optimized Qwen gate/up-SwiGLU and residual paths require specific immediate
graph operations, while LoRA application adds an output sum. Affected layers
can therefore miss those fusion conditions. Kernel/graph dispatch was not
measured here, so the contribution of that mechanism is not established.

## Reproduction and evidence

`compare-lora-cfg1.py` is dry-run by default. Supply `--model-root` and `--lora`;
add `--run` to execute. It checks frozen hashes, uses a new loopback server,
discards native output and image responses, and records only its own timings,
HTTP status and numeric GPU observations. Python is the test client only.

Local client-generated measurements and the source-defined plan are under
`experiments/qwen-image21-sage/runs/lora-cfg1-9sagwp_5/` (ignored by Git).
No application/runtime logs or generated images were retained.
