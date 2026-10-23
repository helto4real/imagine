# H3 separate inference minimum

Status: implementation, CPU/source checks and controlled GPU validation complete.
Exact output and lifecycle checks pass. The larger workload uses more VRAM and
less host-to-device traffic, with a small measured sampling-time improvement;
its native unload-time RAM peak increases. This is not a RAM reduction.
No product package has been activated or committed.

Follow-up: [H3 terminal component release](h3-terminal-release-20261023.md)
identifies and removes the unload-time CPU copy. Fresh matched GPU probes
reduce the larger DiT peak RSS from 30.457 to 20.855 GiB with exact outputs.
The measurements below are the preserved pre-fix budget experiment.

## Change

`inference/src/h3_product/memory_budget.rs` returns the original two-batch
requested budget and a separate one-batch minimum for packed H3 AV, BF16 and
the selected PyTorch flash-attention path. The H3 DiT `Manager::load` call now
passes that minimum. Other components retain `None`. The shared memory policy,
400 MiB reserve, 0.8 GiB inference floor, cast buffers, dtypes, kernels,
sampling and host admission are unchanged.

The current runtime shape is used, including the existing continuation shape
expansion. This does not add accounting for reference/text conditioning absent
from the original H3 estimator. The bounded own-reference workload below passed.

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
socket is bound. Tomas then explicitly authorized a new server instance for the
tests. `start-server.sh` followed by `start-proxy.sh --log` succeeded and the
supervisor reported a healthy proxy. All GPU probes subsequently used that
coordinator. The test proxy/backends were stopped using `--stop`, then the
verified test server executable received SIGTERM and exited successfully.
No service configuration or product executable was changed during diagnosis.

## GPU evidence

Capture, timing and profiling runs were separate. The test-only probe records
resident module bytes, placement/cast buffer/pinned bytes, per-step boundaries,
sampling time, phase allocator peaks, and own RSS/HWM/anonymous/file-backed RAM.
The original probe records allocated and reserved peaks too. CUPTI uses the
existing diagnostic `activity.cpp`, built separately, without modifying it or
the production bridge. It requires zero dropped records. Profiling adds a
per-step synchronization to delimit GPU-copy windows; its explicit sync wait
is not a measurement of all host-side waiting and its timing must not be mixed
with ordinary timing runs. PSS was not captured. The Python profiler derivative
removes `LD_PRELOAD` from the child environment after startup: the parent profiler
remains loaded, while compiler subprocesses cannot overwrite its trace. The
earlier generated Python source and its hashes are retained as version 1.

### Exact outputs and lifecycle

All **97 exact artifact comparisons** pass across short no-reference, short
own-reference and large no-reference cases. They cover token entries, context,
reference latent, noise, sigmas, every sampler observation, final latents,
video frames and PCM. Each geometry uses four Euler/simple CFG-1 steps.
Unchanged text/reference inputs are reused from this experiment only, after
their independent native/original comparison. Both native budget variants and
the static, synchronous original ComfyUI path produce identical captured results.

All **48 reservation cases** satisfy their success or expected-failure criteria,
including namespace exit and `finish_sent`. The intentionally failed 64 GiB
allocation produced OOM with zero loaded/pinned bytes after native cleanup.
Cancellation after the first sampler callback terminated and reaped the worker;
a subsequent candidate run succeeded. These tests belong to the phase probe,
not the deployed product API. Earlier unavailable-coordinator attempts remain
retained failures and are not counted as GPU evidence.

### Larger-workload measurements

Three independent uninstrumented runs per implementation, with ordering
control/candidate/original, original/candidate/control, candidate/control/original.
Times are sampling only; file identity checks, checkpoint loading and decoding
are excluded. GiB values below are medians of phase measurements.

| Metric | Native control | Native candidate | ComfyUI original |
| --- | ---: | ---: | ---: |
| Sampling seconds, median | 29.411 | 28.460 | 28.099 |
| Sampling seconds, range | 29.128–29.782 | 28.154–28.691 | 28.094–28.713 |
| Resident module bytes, GiB | 6.519 | 9.608 | 9.622 |
| CUDA peak allocated, GiB | 8.254 | 11.357 | 11.256 |
| CUDA peak reserved, GiB | 8.980 | 12.098 | 11.938 |
| Process peak RSS, GiB | 27.384 | 30.471 | 21.314 |
| Process RSS after release, GiB | 1.276 | 1.274 | 1.786 |

The candidate's measured median sampling duration is **3.23% shorter** than the
native control. Three repetitions do not establish statistical significance or
a general speedup. GPU peaks include loading and inference, not sampling alone.
All three paths retain 8,519,680 allocated and 20,971,520 reserved CUDA bytes
after graph release in these DiT runs; process exit then ends the CUDA context.
Native pinned accounting returns to zero on cleanup. Pin-error warnings occurred
in the unchanged environment; no pinning settings were adjusted to improve a run.

The RAM regression is localized by phase markers: in repetition 1, native RSS
at sampling end is about 20.870 GiB for both variants. After `unload_all`, it is
27.371 GiB for control versus 30.473 GiB for candidate. Anonymous RSS rises from
about 0.946 to 7.447/10.548 GiB while file-backed RSS remains about 19.849 GiB.
The larger peak is therefore observed during native unload, not steady sampling.
This does not justify a RAM-saving claim or an unapproved ownership refactor.
The original's intermediate `unloaded` patcher counters remain nonzero until
graph references are released; final allocator readings above are measured
after that release, separately from those intermediate counters.

One separate CUPTI run per variant completed with **zero dropped records**.
Sampling H2D traffic totals 50.720 GiB for control, 39.166 GiB for candidate and
39.150 GiB for original: **22.78% less** for the candidate than control. The
four callback-delimited H2D windows are:

| Window | Native control, GiB | Native candidate, GiB | Original, GiB |
| --- | ---: | ---: | ---: |
| Begin to step 0 callback | 13.046 | 9.942 | 9.938 |
| Step 0 to step 1 | 12.558 | 9.741 | 9.738 |
| Step 1 to step 2 | 12.558 | 9.741 | 9.738 |
| Step 2 to step 3 | 12.558 | 9.741 | 9.738 |

No H2D record crosses these synchronized boundaries. Summed H2D copy durations
are 5.130/3.110/3.280 seconds; these sums are not wall-clock savings. D2H totals
inside sampling are only 44/44/37 bytes. Per-window copy durations, explicit
synchronization waits, host counters and trace hashes are in `gpu-results.json`.
The trace shows about 3.10 GiB more resident module bytes for the candidate;
that is a module sum, distinct from measured allocator usage.

### Evidence and decision boundary

`comfyui/h3-memory-20261023/gpu-results.json` indexes comparisons, all timing
samples, trace summaries and reservation outcomes. Each run retains selected
source/binary/model hashes and its own phase report. `run-numerical.py`,
`run-measurements.py` and `summarize.py` record the executed experiment sequence.
Failed attempts and synthetic comparator self-tests are kept separately.

Numerical equivalence and tested lifecycle behavior pass. The bounded workload
shows a modest throughput benefit with higher VRAM utilization, but **the RAM
peak worsens by about 3.1 GiB during unload**. Broad RAM/performance acceptance or
product activation should not be inferred from passing execution checks.
RAM ownership changes, async offload, dynamic residency, product activation,
commit and push remain outside this slice. No browser/manual UI acceptance or
live product API generation was run.
