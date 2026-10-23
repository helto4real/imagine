# H3 separate inference minimum

Status: implementation and CPU/source validation complete; trained comparison
is pending coordinator availability. No product package has been activated.

## Change

`inference/src/h3_product/memory_budget.rs` returns the original two-batch
requested budget and a separate one-batch minimum for packed H3 AV, BF16 and
the selected PyTorch flash-attention path. The H3 DiT `Manager::load` call now
passes that minimum. Other components retain `None`. The shared memory policy,
400 MiB reserve, 0.8 GiB inference floor, cast buffers, dtypes, kernels,
sampling and host admission are unchanged.

The current runtime shape is used, including the existing continuation shape
expansion. This does not add accounting for reference/text conditioning absent
from the original H3 estimator. Reference workload acceptance is still required.

The product source overlay explicitly includes the new module. All unrelated
pre-existing changes in `native.rs`, `mod.rs` and the source preparation script
are preserved. No D-05, Qwen, async-offload or dynamic-residency change is included.

## Source and synthetic evidence

The local ComfyUI revision is `446ef4e005fc7134d53ce6869ebd77d2134cd39c`.
`tools/check-h3-memory-budget.py` executes only the AST-selected original pure
`BaseModel.memory_required`, `extra_conds_shapes` and `estimate_memory` functions
with shape stubs. It imports neither Torch nor ComfyUI and records source hashes.
The no-override and H3 factor assumptions are asserted against the source.

| Workload | Original minimum with reserve | Candidate minimum with reserve |
| --- | ---: | ---: |
| 384 x 256, 39 frames | 1.190625 GiB | 1.190625 GiB |
| 768 x 512, 124 frames | 6.523540 GiB | 3.4570825 GiB |

These values are arithmetic, not observed memory use. Four focused CPU tests
cover packed video/audio, portrait symmetry, the common floor, unchanged desired
budget, constrained free memory, loaded bytes, reserve overrides and policy
gates. All pass. All 15 existing H3 CPU transport/deployment tests also pass.

`inference/examples/h3_memory_placement.rs` runs the real `StaticPlacement` on
the documented H3 module inventory, without model tensors or CUDA calls.
Twelve synthetic placements pass. At the deliberately synthetic 16 GiB free
capacity, short-profile residency is unchanged at 15,824,353,408 module bytes;
the large profile changes from 10,059,579,792 to 13,352,163,448 module bytes.
Its cast buffer remains 115,691,592 bytes. This is not actual free capacity,
physical VRAM, RAM savings or measured speedup.

The isolated product worker, phase probe and placement checker build. Targeted
warnings-denied Clippy passes for them and the new CPU tests. The broader
H3 CPU `--all-targets` Clippy check fails on pre-existing `useless_vec` at
`inference/tools/h3-product-cpu/tests/product_cpu.rs:80`; that unrelated line
is left untouched. New Rust files pass rustfmt; task files pass whitespace checks.

`tools/compare-h3-memory-probe.py` reuses the frozen exact artifact comparator,
restricted to this experiment's successful capture runs and matching workload,
profile and checkpoint identities. Synthetic self-tests accept equal payloads
and reject both changed payloads and checksum corruption. These self-tests
have no GPU or reservation evidence and are explicitly labelled as synthetic.

## Isolation and reproducibility

The ignored experiment root is `comfyui/h3-memory-20261023`. Its source is
archived from `cd0ef53be69dd539ef0ffdf0cb03f4c1c5835fc7` with the explicit current
H3 product overlay. `source-inputs.json` records that overlay's hashes. A single
separate `target` is reused. Locked Cargo packages, Torch, CUDA/Kitchen and
the accepted source-built bridge are reused; no new backend/toolchain is installed.
The source archive is approximately 947 MiB, including source fixtures; no old
multi-gigabyte Cargo targets or complete backend environments are copied.

The accepted bridge SHA-256 remains
`ab7c807cd5f9aac244f58a302d9500a6ab04b0314b7223404881ce407ed0d4a3`.
The existing isolation preflight passes its source, library and dependency
checks. Frozen executables, evidence and existing product/source/build roots
remain unchanged. New worker snapshots and reused own tensor inputs use
`cp --reflink=auto`, never writable hardlinks.

`tools/prepare-h3-memory-probe.py` generates explicitly bounded derivatives of
the frozen Stage 4 phase probes in the isolated source. Original probe files
are unchanged. Only `short` (384 x 256 / 39 frames) and `large` (768 x 512 /
124 frames), the same locked Ref2VA weights and four Euler/simple CFG-1 steps
are admitted. `control` retains the old minimum; `candidate` uses the new one.
The Python candidate invokes the original `estimate_memory` directly.
The frozen source prompt and an optional deterministic source-created reference
are the only inputs. This is unrelated to the unknown screenshot workload.

`tools/run-h3-memory-probe.py` acquires a Ticket and runs one phase in an owned
PID namespace with selected read-only dependencies/model files. Only this
experiment's newly created tensor inputs can be reused. No private product
logs, prompts, jobs, media or reference data are read. A missing coordinator
fails before namespace/model start; there is no fallback or service startup.

Tomas explicitly selected `$XDG_RUNTIME_DIR/imagine/gpu.sock` after the old
checkout socket was unavailable. `start-proxy.sh` creates this endpoint in the
private-storage configuration; `start-server.sh` selects it as a client. Initial
All six initial attempts failed before any GPU worker: the first on sandbox
permissions, subsequent attempts with `coordinator unavailable` / ENOENT.
Sandbox escalation was needed to reach the coordinator; no auto-review rejection
occurred. The failed attempts are retained separately and are not GPU evidence.

A seventh reservation attempt also stopped before namespace start. Tomas then
explicitly authorized starting the proxy with `--log` to investigate startup.
The selected product was `20261023T054300Z-f71a3541`. Its LLM dependencies became
ready, but the proxy exited with the generic `startup or shutdown failed` error;
the supervisor subsequently stopped the dependencies. A separate direct proxy
startup trace limited to `bind`, `connect` and `getsockopt` confirmed successful
binds on 5002, 8189 and 8190, followed by `ECONNREFUSED` on 127.0.0.1:8080.
No HTTP bodies or file contents were traced. The selected package's source at
`comfyui/ui-groups-20261021/source/proxy/src/mcp.rs` requires Imagine's
`/v1/media-destination` response during MCP construction, before the coordinator
socket is bound. Starting the separate Imagine server awaits authorization.
No service configuration or product executable was changed during diagnosis.

## Remaining acceptance

Use separate capture, timing and profiling runs. The test-only probe records
resident module bytes, placement/cast buffer/pinned bytes, per-step boundaries,
sampling time, phase allocator peaks, and own RSS/HWM/anonymous/file-backed RAM.
The original probe records allocated and reserved peaks too. CUPTI uses the
existing diagnostic `activity.cpp`, built separately, without modifying it or
the production bridge. It requires zero dropped records. Profiling adds a
per-step synchronization to delimit GPU-copy windows; its explicit sync wait
is not a measurement of all host-side waiting and its timing must not be mixed
with ordinary timing runs. PSS is not currently captured.

Pending gates:

- Short no-reference and own-reference exact comparisons across control,
  candidate and static/synchronous ComfyUI: dtype, shape, payloads, token entries,
  noise, sigmas, every sampler observation, final latents, frames and PCM.
- Larger-profile exact comparison, then counterbalanced timing repetitions;
  physically resident module sums and H2D/D2H bytes/times per step.
- CPU/GPU peak and after-release accounting; no claim that module sums equal
  actual physical memory or that mmap size equals RSS.
- Controlled OOM (native probe only), cancel after sampling begins, clean
  release and subsequent successful recovery. The probe's OOM allocation is
  explicitly test-only and never affects product execution.
- Compare the instrumented results separately from uninstrumented timings.
  Retain failures and do not relax numerical, identity or lifecycle gates.

Until these gates run successfully, this is a built source candidate, not
memory/performance acceptance or authorization to activate it. RAM ownership
changes, async offload, dynamic residency, service changes, commit and push
remain outside this slice.
