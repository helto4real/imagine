# API v1 implementation validation

Date: 2026-10-14. Scope: the first Rust API slice, with the unchanged local
`qwen-image21-1600x896-v1` native executable and environment.

## Static and simulated checks

Completed successfully:

- `cargo fmt --all --check`
- `cargo clippy --locked --offline --all-targets -- -D warnings`
- `cargo test --locked --offline`: seven tests (two unit, five integration)
- `cargo build --release --locked --offline --bin imagine-server`
- `./start-server.sh --help`
- `bash -n start-server.sh`
- `python3 -m py_compile tests/live-native.py`
- `git diff --check`

Controlled fake-engine tests cover metadata without model loading, strict input
validation, ordered references and reference preprocessing, upload ownership,
job ownership, admin-only unload, idempotent submissions and conflicting keys,
FIFO dispatch, queued cancellation, active cancellation, unload/wait preserving
queued jobs, resume, engine failure recovery and terminal SSE. Persistence checks
cover graceful interruption, simulated crash rows, pending unload operations,
retained queues and pause state. Credential checks cover file permissions,
distinct stable tokens, loopback-only binding and the data-directory lock.
These tests do not simulate CUDA kernels or certify image quality.

## Real native/GPU acceptance

Command: `python3 tests/live-native.py --run`.

Hardware: RTX 4070 Ti SUPER, 16 GB. Before the test: 1351 MiB used, 14643 MiB free;
no other inference process was running. The test launched the Rust service via
`start-server.sh`, using its own private data directory. It used only the frozen
synthetic benchmark prompt and a newly generated reference; no ComfyUI logs,
private prompts, user photos or chat data were read.

| Case | Execution to saved asset | Submit to downloaded PNG | Result |
| --- | ---: | ---: | --- |
| First reference job, including lazy native startup | 41.067 s | 41.098 s | Matches frozen RGBA hash |
| Reference edit, 1 MP per image | 47.267 s | 47.299 s | Valid 1600×896 output |
| Reference job after full unload/resume | 41.346 s | 41.380 s | Matches frozen RGBA hash |

Both reference outputs match:
`473bfa4ab0c2230258a168afbe07523bd6bc7d81a4a2daf458500b6df036132d`.

The edit output RGBA hash is
`a4479f90c72bb0b631d972855f69258a16d5258b9ac29434221c7037978ec544`.
Its reference preprocessing was 1600×896 → 1344×736 (989,184 pixels), as recorded
in the effective job. This is a functional API edit test; it is not a new
portrait-likeness study or a native-versus-ComfyUI comparison.

Unload/wait completed the active edit, stopped/reaped the native process and
left a queued image untouched. That queued job was then cancelled without
execution. Resume started a new native PID and reproduced the reference image.
Unload/cancel terminated another active generation, marked it cancelled and
confirmed native PID disappearance. Metadata requests before the first job did
not start the worker. The harness stopped its API process after the test. A final
GPU query returned to 1351 MiB used / 14643 MiB free, the pre-test values.

Local evidence (ignored):
`.imagine/test-runs/native-api-lhx_ef_d/results.json`, three PNGs, the private test
database and a startup-only server log. Secrets and media are not published.

These are single observations, not repeated performance measurements; timings
include API validation, persistence and transfer as indicated. They do not
replace the native experiment baseline or establish a new ComfyUI speed gap.

## Deliberate limits and unrun checks

No browser UI or Dreamchat integration has been implemented or tested. H3,
video/audio, multi-reference quality beyond one image, LoRAs and other output
resolutions remain outside this acceptance. There is no remote/TLS deployment
check. The Linux parent-death safeguard is implemented but no live SIGKILL/crash
injection was performed; database crash recovery was tested synthetically.

No native source, CUDA settings or frozen executable was changed. Network access
was elevated to fetch Cargo dependencies, and GPU access was elevated for the
live test. No commit or push was performed for this implementation.
