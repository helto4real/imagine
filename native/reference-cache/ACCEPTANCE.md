# Qwen reference-cache acceptance — 2026-10-16

**Decision: do not promote yet.** Functional acceptance passed, but CFG 1
regressed by 11.5%. Frozen v2 remains the normal Rust API executable.
Hybrid + prefetch remains opt-in. No native source, sampling setting, frozen
manifest, production binary or normal API configuration was changed by this
acceptance run. An activation draft was not installed.

## Live API results

All cases used separate test data under
`test-generations/reference-cache-acceptance-20261016-084051-fafcf4`. Only these tests' scoped
status, output checksums and numeric diagnostics were inspected. No visual
quality assessment was performed. API servers were fresh per case; cancellation
and recovery intentionally shared one API server. GPU work was sequential.

| Workload | Frozen v2, seconds | Hybrid + prefetch, seconds | Output equality |
| --- | ---: | ---: | --- |
| Reference character sheet, CFG 2.5 repeat | 210.839 (earlier control) | 80.121 | All 3 PNGs match pinned control hashes |
| Reference character sheet, CFG 1 | 41.271 | 46.025 | All 3 PNGs identical |
| Text-only ordinary image, CFG 2.5 | 42.998 | 42.762 | PNG identical |
| Ordinary image with 3 references, CFG 2.5 | 205.457 | 57.331 | PNG identical |
| Full reference sheet immediately after cancellation | — | 81.312 | All 3 PNGs match pinned control hashes |

Times are server-reported execution-to-saved output. The CFG 2.5 repeat is
consistent with the preceding 79.754-second candidate run. Other table pairs
are contemporaneous. Each pair has one sample, clocks/desktop load were not
locked, and source checks ran during part of the CFG 1 comparison; these are
acceptance observations, not general throughput guarantees.

The sheet retained 40 ER-SDE/beta steps, seeds 998622113751077/78, portrait
704×864, turnaround 832×864 and final sheet 1536×864. Only CFG changed for
the CFG 1 pair. All references retained the 1,000,000-pixel per-image budget.
The ordinary three-reference image used the original approved identity card,
plus the portrait and turnaround from this suite's CFG 2.5 repeat, at 832×864,
seed 424242. The new text-only still life used 1600×896 and seed 424242.
Neither workload used LoRAs. The preserved test harness contains exact requests.

## Why promotion is blocked

At CFG 1 the portrait sampling was effectively unchanged (11.466 vs 11.634 s),
but turnaround sampling increased from 17.715 to 21.777 s. The hybrid planner
reserved **two** branches even though only one was computed: its diagnostic
requested 9306 MiB and admitted a 3200 MiB total GPU cache budget. The actual
single branch held 1599.47 MiB in VRAM and 3053.53 MiB in RAM, requiring 39
prefetch forwards. Frozen v2 kept that prefix cache in VRAM and sampled faster.
The source explicitly calls `cache_.set_hybrid_storage(2)` in the Qwen runner.
This is a concrete planning issue to address before blanket activation; faster
CFG 2.5 results do not make the CFG 1 regression a pass.

Recommended next change: derive the cache reservation from the branches actually
used by the generation, preserving weights, workspace, staging and safety
reserves. Recheck CFG 1 and the established CFG 2.5 control before promotion.
This report does not authorize that implementation or another model run.

## Ordinary multi-reference generation

The common runner already supports the ordinary image path. In this test,
steady weight-load events fell from **1413 to 0**, and sampling fell from
192.700 to 45.394 s. Prefetch executed in all 78 steady forwards, with no cache
or prefetch fallback. Combined cache placement was 3096.16 MiB VRAM plus
5910.84 MiB RAM; two staging buffers used 282.22 MiB.

Future API activation should depend on model capability and actual references
in each inference pass, independent of `character_sheet` versus `image_edit`.
GPU-resident caches need no host prefetch. Other models and Qwen Turbo retain
their own inference profiles until separately validated; this is not evidence
for universally enabling the Qwen implementation across models.

## Cancellation and recovery

The sheet was cancelled through its API endpoint after turnaround step 3.
Four prefetch events had already occurred. The API reported `cancelled`,
published no partial output, and the old native process was no longer running.
The next job was submitted immediately in the same API server and completed
with a different native PID and the exact expected three PNGs. Its 78 steady
turnaround forwards used prefetch with no fallback or repeated weight loads.

This validates the API's process-termination/restart path; it is not a test of
graceful in-process CUDA cancellation. All eight isolated API servers stopped,
and their recorded wrappers/native children were verified no longer running.
GPU/local-socket execution required approved sandbox escalation.

## Artifacts and source checks

Each case records `result-metadata.json`, scoped `status-events.jsonl`, numeric
`memory-diagnostics.jsonl` and output PNGs. `cancel-recovery` separately preserves
its cancelled diagnostics. `summary.json` collects completed jobs;
`FUNCTIONAL_PASSED` is explicitly distinct from the failed promotion decision in
`acceptance-decision.json`. Test harness and diagnostic wrapper are preserved.

The prospective Rust selector passed focused tests, full `cargo test --workspace`,
Clippy with warnings denied, formatting and diff checks. Socket tests initially
hit sandbox permission restrictions and passed after approved escalation. Those
selector changes were withdrawn after the performance gate failed; `src/engine.rs`
is byte-identical to its pre-acceptance contents. No release rebuild, activation,
commit or push occurred. No native source changed, so prior overlay restoration
and synthetic CUDA evidence still apply to the unchanged candidate.

Frozen v2 SHA-256:
`9fd49b0e5cb3426043ebd85271ad167555551ac65252590cde5ac25bed44bbb0`.
Candidate SHA-256:
`19545cb052e9248d8d0076874a06acc45308b4253ba1f6f274388721f72032bd`.
Both executable hashes were verified unchanged.
