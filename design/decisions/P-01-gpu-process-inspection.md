# P-01: GPU memory and optional process-list inspection

Date: 2026-10-17. Status: **resolved: Tomas freed GPU memory; unchanged validation passed; no new access authorized**.
This is a development-data permission request, not a parity exception.

## Observed blocker

The new INT4 implementation passes its differential test in both ordinary and
Python-free processes: 81 successful cases with exact payload/scale bytes and
171 expected ValueErrors with matching messages.

The complete test suite initially failed under GPU memory pressure. The normal
process passes 18 primitives and fails three; a fresh isolated process passes 19
and fails two. Both fail allocating the unchanged original autotuning benchmark's
268,435,456-byte (256 MiB) cache-clear buffer. Other failures also report CUDA OOM.
These are failed aggregate checks, not accepted numerical results.

The only GPU-state query was total/used/free memory, with no process information:
16,376 MiB total, 15,621 MiB used and 373 MiB free as reported by nvidia-smi.
No owner, PID, process name or command line has been inspected. The consumer of
that memory is unknown.

## Boundary and requested choice

The project [AGENTS.md](../../AGENTS.md) says “Never read runtime data”. Existing
authorization permits the new engine's own isolated development data. GPU process
information about ordinary workloads is treated as outside that scope; this is
the privacy interpretation behind the request, not an explicit instruction in
AGENTS.md naming GPU process lists.

Tomas can either free GPU memory himself, or explicitly approve this narrow read:

- GPU-compute process PID, program name and GPU-memory usage only.
- Use it to prepare a concrete proposal for freeing memory.
- No prompts, logs, histories, media, databases, environment variables or command
  lines; no process/service stop or other mutation is authorized by this read.

Tomas answered on 2026-10-17:

> Jag frigör GPU-minne själv

Tomas subsequently confirmed on 2026-10-17:

> Nu är GPU-minnet frigjort. Fortsätt valideringen och steg 1 enligt designen. Dokumentera resultaten och stoppa bara om något kräver mitt uttryckliga godkännande.

The unchanged aggregate suite was resumed using `inference/tools/check.sh`;
results are recorded in the Stage 1 evidence. Neither answer authorizes process-list
inspection.
Stopping/unloading an ordinary service would require a separate concrete approval.

## Conditions retained

Do not shrink the benchmark buffer, shorten measurement budgets, skip required
tests, change autotuning/cache policy, pin a winner or introduce a tolerance to
work around memory pressure. Retry the unchanged aggregate suite after resources
are available. Keep its failure evidence and the historical 28/20 success
separate. The unchanged recovery suite subsequently passed 29 crate tests and
21 isolated primitives; the ConvRot continuation then passed 32/24. P-01 is
resolved without process inspection or service changes.

No ordinary service has been inspected or changed. No automatic approval review
rejected an action; the access boundary remains unchanged after Tomas
chose to free memory himself.
