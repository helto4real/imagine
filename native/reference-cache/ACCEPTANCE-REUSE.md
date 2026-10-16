# Qwen reference cache: corrected budget acceptance

Decision: enable hybrid + asynchronous prefetch for Qwen Image 2.1 passes with
references, independently of ordinary images or character sheets. Frozen v2
remains the text-only profile and explicit/verification fallback. This supersedes
the blocked decision in [the first acceptance report](ACCEPTANCE.md).

## Final live results

All GPU work was sequential, with isolated API data collected in
`test-generations/reference-cache-reuse-final-20261016-093146-1cd7be/summary.json`; each case records its own directory. Each test waits for its own API, wrapper and native child
to exit before starting the next case. Only authorized test status, numeric
diagnostics and result hashes were read. No visual assessment was performed.

| Workload | Control, seconds | Corrected candidate, seconds | Result |
| --- | ---: | ---: | --- |
| Reference sheet, CFG 1 | 39.888 (fresh v2) | 39.620 | All three PNGs identical |
| Reference sheet, CFG 2.5 | 210.839 (historical v2); 82.167 (same-session previous prefetch) | 80.843 | All three pinned PNG hashes match |
| Ordinary image, three references, CFG 2.5 | 205.457 (v2); 57.331 (previous prefetch) | 61.358 | PNG identical |
| Reference sheet after cancellation | 81.312 (previous prefetch) | 81.393 | Cancel cleanup and immediate recovery pass; all hashes match |

The final ordinary-image and recovery totals include the benchmark-only startup
memory stabilization wait, unlike the older controls. Ordinary-image sampling
is 45.836 seconds versus the previous 45.394 seconds; the raw total difference
is not attributed to inference. No end-to-end speedup over previous prefetch is
claimed. CFG 1 and the CFG 2.5 repeat did not add this startup wait.

Settings remain 40 ER-SDE/beta steps, the approved seeds and prompts, and the
1,000,000-pixel cap per reference. Portrait is 704×864, turnaround 832×864,
composed sheet 1536×864; the ordinary image is 832×864. No LoRA, sampler or
precision changes. These are individual observations on one RTX 4070 Ti SUPER,
not throughput guarantees across workloads or machines. The original text-only
comparison remains applicable because normal text-only passes still select v2.

CFG 1 uses one actual guidance branch. After the initial store, the smaller
reuse graph allows the remaining host K/V cache into VRAM; both passes then
have no host prefetch. CFG 2.5 retains two branches, partial VRAM placement and
78 steady prefetch forwards in the turnaround. The ordinary three-reference
image also has 78 prefetch forwards. All final passes have zero steady weight
loads and no cache/prefetch fallback. See each case's `result-metadata.json`
for exact allocations and sampling times.

Cancellation was requested after turnaround step 3, while prefetch was active.
No partial output was published, the old native PID exited, and an immediately
submitted job in the same isolated API server succeeded with a different PID.
This exercises process termination/restart, not graceful in-process CUDA abort.

## Implementation and rejected trials

The planner shares the sampler's exact branch-use predicates and unions them
over the complete guidance schedule. Branch counting alone gave 43.667 seconds
at CFG 1 against 39.888 seconds for v2, so it was not sufficient.

The runner now rechecks capacity once after all planned branches are stored,
using the measured steady graph workspace. It protects weights, K/V staging,
existing runtime reserve and 256 MiB additional margin. A dry capacity check
preserves the existing workspace if promotion is rejected. If accepted, it
releases the oversized store workspace, rechecks physical capacity, copies all
remaining host entries transactionally, and rebuilds the graph while retired
host inputs remain alive. Failed admission leaves hybrid/prefetch in place.
No cache tensors are persisted to disk and no per-step promotion is attempted.

An earlier trial discarded workspace even when promotion was rejected. It gave
40.248 seconds at CFG 1 but 90.449 seconds at CFG 2.5, versus a subsequent
82.167-second run of the previous prefetch binary. That trial was not promoted.

Separate intermittent 117–119-second trials chose an all-RAM cache. Scoped
capacity diagnostics showed a startup graph limit of 6221.44 MiB even though
13895.81 MiB was free later. Source inspection confirmed auto max-VRAM is a
startup snapshot. The test harness waited for the API but not its native
grandchild, which was a possible source of transient low startup budgets.
Waiting for both recorded PIDs alone did not eliminate
the transient low startup limit: a subsequent recovery still took 119.340 seconds.
The final cancellation/multi-reference harness also requires four half-second
samples with at least 13000 MiB free before launching the native test process.
This is benchmark isolation only, not a production minimum-memory restriction.
The precise source of every startup allocation was not established; other
applications and their data were not inspected. No native auto-memory policy
was relaxed. Low-budget trials are retained as fallback observations, not used
as matched performance evidence. The production selector launches the native
binary directly and its profile-switch path kills and waits for that child.

## Source and synthetic verification

- Native CUDA build and assertion-enabled BF16 cache tests pass: GPU/RAM/hybrid,
  whole K/V pairs, one/two branches, exact copies, retired input lifetime,
  failed-graph rollback and per-pass reset.
- Guidance predicates match the old skip decisions, including epsilon,
  non-finite values, forced negative guidance and scheduled branch unions.
- Clean restoration from pinned upstream through the base plus all five
  reference-cache overlays matches eleven source files and every lock hash.
- Rust workspace tests, Clippy with warnings denied, formatting and diff checks
  pass. Selector tests cover reference counts, other models, explicit off,
  missing/corrupt executable and actual process replacement across profiles.
- Frozen v1/v2 binary hashes, base patches and workflow compatibility lock are
  unchanged. GPU and local-socket test execution used approved escalation.

Candidate SHA-256: `b3188ab2dee7297f4a9060dea2155dcdc7206c0a7e1031c5fdf25e21e810615c`.
Frozen v2 SHA-256:
`9fd49b0e5cb3426043ebd85271ad167555551ac65252590cde5ac25bed44bbb0`.

The native candidate is packaged in the branch/reuse patch chain and pinned by
`reuse-budget-source-lock.json`; the compiled executable is local and ignored
by Git. The Rust API verifies it before use. `IMAGINE_QWEN_REFERENCE_CACHE=off`
selects v2. A missing or mismatched candidate also selects v2; this is a startup
selection fallback, not automatic regeneration after an inference failure.
Turbo and future model families do not inherit this Qwen-specific profile.


## Installed API verification

Installed the pinned candidate at `native/reference-cache/bin/sd-server` and
rebuilt `target/release/imagine-server`. A separate isolated API smoke test
used the actual project and installed binaries, without a diagnostic wrapper
or benchmark startup wait. All three two-step jobs succeeded:

- `auto` with a reference launched the pinned reference executable with both
  hybrid placement and prefetch flags.
- A subsequent text-only job in the same API stopped that child and launched
  frozen v2 with no reference-cache flags.
- `off` with a reference launched frozen v2.

Only these test children's executable paths, hashes and cache flags were
inspected. All test children exited. Evidence is in
`test-generations/reference-cache-selection-434487d8/selection.json`.
This checks integration and selection, not image quality or performance.
No existing production API process was inspected or restarted. The installed
profile takes effect when the rebuilt server starts. No commit or push.
