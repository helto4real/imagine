# Stage 2 follow-up after the approved reference upgrade

## Scope and accepted baseline

Tomas confirmed on 2026-10-18 that the Rust backend works in Imagine and
requested a baseline commit before completing the remaining Stage 2 work.
Commit `f6fc1cd` preserves the approved ComfyUI/Kitchen upgrade, Qwen/MOSS
implementation, default Rust server integration, original-source controls and
regression tooling. Its exact staged tree passed 148 workspace tests with one
deliberately ignored live test. Unrelated native optimizations and UI layout
work remain outside that commit. No push or ordinary service restart occurred.

The original selected Qwen Stage 2 checkpoint remains accepted against its old
reference. This continuation concerns the upgraded reference
`446ef4e005fc7134d53ce6869ebd77d2134cd39c`, Kitchen 0.2.38 and the early product
adapter. It does not relabel old comparisons or weaken their requirements.

## Remaining work and decisions

1. Expose the existing ordinary Qwen BF16 inference path through the model
   library and Rust server, alongside INT8 and Turbo. Reuse the Qwen component
   group, ordered LoRA settings, sampling defaults, references and two-pass
   character-sheet mapping. The explicit BF16 model ID must retain the BF16
   checkpoint and cannot silently select INT8 or Turbo. Other backends must
   reject the new Rust-only product selection before admission.
2. Investigate the retained automatic BF16/Turbo LoRA comparison failures.
   Observe the original memory-query inputs during DiT initial loading and
   sampling without overriding them. Preserve the automatic memory policy,
   source patch arithmetic, exact comparison requirements and autotuning.
   Recorded-placement replay is diagnostic evidence only.
3. Validate changed admission/discovery/UI request paths with synthetic tests
   and the real BF16 Rust worker in a fresh isolated development project. Trained
   comparisons run serially, after the existing aggregate host-RAM guard.

The new BF16 control separates adapter correctness from the existing LoRA
placement investigation: it uses the same bounded reference geometry and
ER-SDE/beta settings as the INT8 server control, with the ordinary BF16 DiT and
no LoRA. This is a new precision/adapter workload, not a replacement for the
failed BF16/Turbo LoRA controls. The selected INT8 text encoder and BF16 VAE
remain unchanged. No new dependency, production memory override or numerical
tolerance is introduced. New builds use a separate target directory so the
working baseline's deployed worker and manifest remain intact during validation.

All samplers/schedulers/guiders and general conditioning/loading/patching
contracts remain required under master-design Stage 3. MOSS variant expansion
is outside this Qwen Stage 2 continuation. No requirement is removed.

## Results and acceptance

**The selected Stage 2 follow-up is complete.** Ordinary BF16 is exposed through
the Rust server/model library, and fresh BF16/Turbo automatic LoRA controls pass
the strict numerical and observed-memory checks against the upgraded reference.
The [evidence directory](stage-2-upgraded-followup-evidence/README.md) retains
the new results and the failed intermediate controls separately.

| Control | Result |
| --- | --- |
| BF16 HTTP/model-library/reference control, 320×224, four ER-SDE/beta steps, CFG 2.5, no LoRA | Exact decoded PNG pixels through the real queue/Rust-worker path; no Python mapping in worker; shutdown reaps worker and removes session media |
| BF16 automatic LoRA/reference control, 320×224, four ER-SDE/beta steps, CFG 2.5 | All seven Float32 boundaries and PNG pixels exact; all five component-memory states and three DiT lifecycle states exact |
| Turbo automatic LoRA/reference control, 320×224, eight ER-SDE/beta steps, CFG 2.5 | Same exact numerical/component/DiT checks; original-first run, no recorded-placement replay |
| Workspace | 152 passed, two deliberately ignored live tests; workspace and independent inference all-target Clippy pass |
| Focused source export, excluding unrelated native/UI work | 150 passed, one deliberately ignored live test |
| Strict automatic acceptance gate | Positive trained control passes; all six negative controls rejected, including stale tensor/memory reports, placement replay and an actual failed pair |

Both accepted automatic controls actually offload: BF16 initially has
12,888,072,192 loaded bytes and 1,342,177,280 pinned bytes; Turbo has
12,485,419,008 loaded bytes and 1,744,830,464 pinned bytes. Original and Rust
agree on resident leaves, lazy patch passes, loaded/buffer/pinned byte counts,
cached model size, low-VRAM state and sampler-buffer device/dtype/values.
Unload returns loaded/pinned counts to zero and retains the original registered
Float32 sampler buffer. No physical placement is injected.

The ordinary BF16 model uses the existing Qwen INT8 text encoder and BF16 VAE,
with unchanged standard 40-step/CFG 2.5 product defaults. Its distinct model ID
cannot silently choose INT8/Turbo. Synthetic library and headless UI tests cover
LoRA retention, model precision identity, request construction, mismatched
checkpoint rejection and non-Rust admission rejection. The trained adapter
check is the bounded BF16 reference/no-LoRA case above; it is not a newly
trained acceptance of every BF16 character-sheet or pressure combination.

## Memory investigation and decisions

The first BF16 pair and two intervening Turbo pairs remain recorded as failed:
contexts, references and schedules match, but different live free-memory inputs
lead to different placement and resident/lazy LoRA arithmetic. The first BF16
pair differed by about 344 MiB before DiT loading. Tomas then reported that he
had generated concurrently and stopped the ordinary Imagine server himself.
He subsequently confirmed no other model was running and supplied a dashboard
showing approximately 2.5 GB of 16 GB in use.

An aggregate-only idle observation still varied by 65 MiB over 20 seconds.
No process list was read. The exact source of every historical fluctuation was
not attributed; the user report does not retroactively establish the cause of
all old failures. The new observation records distinguish ATen active/reserved
memory from global available memory. The accepted BF16 pair and subsequent
original-first Turbo pair independently reach identical observed placement and
exact outputs. Test order changes scheduling only, not operands or policy.
No failed result is relabelled, and no inference algorithm or memory policy was
changed to obtain acceptance.

The new `check_qwen_automatic_acceptance.py` gate verifies current tensor-file
hashes, complete component ownership and every DiT lifecycle field, rather than
accepting an old report or numerical equality alone. It rejects replayed
placements and requires actual automatic offload. The normal upgrade runner
invokes it automatically; explicit placement replay remains diagnostic only.
`test_qwen_automatic_acceptance.py` preserves the six rejection checks for
future upgrades without allocating any model weights. `--original-first`
provides the separate execution-order diagnostic used here.

This closes the selected fresh upgraded-reference controls; it does not promise
identical images under different live memory/patch operands, establish equal
physical peak VRAM or replace the wider model/pressure matrix. Earlier 1600-pixel
benchmarks and old-reference cases retain their original scope. The small new
timings include loading/export/transport and are not performance benchmarks.

## Preservation and remaining project scope

The baseline commit is `f6fc1cd`. The subsequent implementation, tests and this
report remain uncommitted. The working baseline's deployed worker and manifest
were preserved during all new validation; candidate binaries use a separate
target and manifest. The ordinary server was neither inspected nor restarted
by the agent. The next normal `./start-server.sh` rebuilds the worker/server and
publishes the corresponding verified runtime manifest, including BF16 support.

The evidence also retains a focused source patch and input fingerprints for the
independently tested source export. This proves the new changes do not depend
on the unrelated native/UI work still present in the main checkout. The actual
Git index was left unchanged after the initial baseline commit.

Only selected local model assets and own isolated outputs were read. Test
socket/GPU/namespace sandbox escalations were granted; none was rejected. No
new dependency, cloud service, numerical tolerance, autotuning change or design
deviation was needed. The reference locks, frozen native comparison executable
and unrelated dirty native/UI changes remain intact.

Full sampler/scheduler/guider and general conditioning/loading/patching parity
remain Stage 3. Broader model variants and production-release coverage remain
Stages 4–5. MOSS is still the separately documented selected SFT INT8 T2I path.
