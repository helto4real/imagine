# H3 terminal component release

Status: implemented and validated with controlled GPU phase probes. No product
package activated; no commit or push performed by this task. The test server,
proxy and proxy-owned backends have been stopped.

## Cause and change

H3 creates a fresh model, patcher and manager for each component. Previously,
`component<T>` called `Manager::unload_all` immediately before dropping them.
The reusable unload path reaches `StaticOffload::unload` in
`inference/src/models/qwen_image21/offload.rs`, which moves resident parameters
to CPU. These newly copied weights were then discarded. During that copy,
anonymous RAM and the mapped checkpoint pages coexist.

`inference/src/h3_product/lifecycle.rs` now releases this terminal ownership
directly: explicitly unpin weights, drop patcher and model, then clean up the
manager's weak registrations. Explicit unpin failures remain returnable; the
existing finalizer handles destruction without the reusable CPU transfer.
The caller retains synchronization, cache release and error precedence.
The source overlay includes the helper, and the GPU probe imports that exact
helper rather than a separate implementation.

This is an H3-only lifetime change. Shared reusable unloading, Qwen inference,
budget calculations, sampling, dtypes, pinning configuration, asynchronous
offload and dynamic residency are unchanged.

## Matched measurements

Both arms use the separate H3 minimum from the preceding experiment. Only
terminal cleanup differs. Three timing pairs ran in old/new, new/old, old/new
order with the same compiled probe. The larger workload is 768 x 512,
124 frames at 24 fps (approximately 5.17 seconds), four Euler/simple steps,
CFG 1, without a reference.

| Median of three runs | Reusable unload then drop | Terminal release |
| --- | ---: | ---: |
| Process peak RSS | 30.457 GiB | 20.855 GiB |
| Sampling | 28.518 s | 28.764 s |
| Measured DiT phase | 40.190 s | 36.512 s |

Peak RSS falls by 9.602 GiB (31.53%). Sampling remains in the same range;
these three pairs do not establish a sampling speedup. The measured phase
falls by 3.678 seconds. It includes loading, inference, output materialization
and cleanup, and is not end-to-end product request latency.

Separate CUDA traces report zero dropped records. After sampling, device-to-host
copies fall from 10,279,572,992 bytes across 186 copies to 5,508,864 bytes across
two output copies. Summed copy duration falls from 3.315 seconds to 0.000474
seconds. The removed volume matches resident model weights; the residual output
transfer remains. Sampling host-to-device traffic remains approximately
42.17 GB in both traces. This confirms removal of the terminal weight transfer.

After each timed DiT run, CUDA allocated bytes return to 8,519,680; reserved
bytes are 12,582,912 or 20,971,520. These include retained output/runtime buffers,
so zero allocated bytes is not the criterion. The terminal model is gone,
manager registrations are zero and tracked pinned bytes are zero.

The earlier static synchronous ComfyUI comparison measured 21.314 GiB peak RSS
and 28.099 seconds median sampling. That historical arm was not rerun here;
the fresh matched native comparison above isolates the cleanup change. The
previous native RAM disadvantage is removed in this bounded DiT workload.

## Correctness and lifecycle evidence

- 65 exact artifact comparisons pass: text/context, reference encoding, noise,
  sigmas, sampling observations, final AV latents, decoded frames and PCM as
  applicable. Short workloads use 384 x 256, 39 frames at 24 fps (1.625 seconds),
  with and without the experiment's deterministic reference. Large DiT/video/
  audio results match the prior candidate. A post-failure short run also matches.
- All 23 Ticket reservation cases pass, using the user-selected
  `/run/user/1000/imagine/gpu.sock`. Namespace exit and reservation finish are
  verified. There is no coordinator fallback.
- Controlled allocation failure after loading exercises terminal cleanup with
  no surviving model, registrations or pins. Forced cancellation after step
  zero checks namespace termination and reservation release; it does not claim
  graceful in-process destructor execution. The next run succeeds exactly.
- The isolated worker and probe build with locked offline dependencies.
  Targeted warnings-denied Clippy passes. Helper rustfmt, modified Python AST
  parsing and focused whitespace checks pass. The current component body and
  helper match the tested source. Earlier unchanged CPU checks are retained;
  the previously reported unrelated broad CPU Clippy failure is not repaired
  or represented as passing here.

Evidence lives under the ignored `comfyui/h3-memory-20261023` root:
`terminal-release-results.json`, `terminal-release-source-inputs.json`,
`terminal-release-checks.json`, `run-terminal-release.py`,
`summarize-terminal-release.py`, the `release-*.json` comparisons and
`runs/release-*` captures. Earlier controls, results and the pre-change source
snapshot remain intact. The selected bridge and checkpoint identities are
unchanged from the preceding report.

## Limits and operational state

These are isolated source-based phase probes using synthetic inputs, not a
live product API, complete-request RAM, browser, or quality acceptance test.
RSS includes mapped pages; PSS was not measured. Existing pin-error warnings
remain. The full set of unrelated current product changes was not validated
by this frozen source build.

Only the expressly authorized test server and proxy were started. Both were
stopped after validation, and the test coordinator socket disappeared.
Sandbox escalation was used for the authorized GPU/service operations; no
automatic approval rejection occurred in this follow-up. No persistent
service configuration or active product package was changed.
