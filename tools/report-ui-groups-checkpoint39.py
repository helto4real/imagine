#!/usr/bin/env python3
"""Persist known CP39 source/test outcomes; never reads runtime/job/media data."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/home/thhel/ChatGPT/imagine')
OWN = Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')
PACKAGE = ROOT/'comfyui/ui-groups-20261021/packages/cp39-components-f1c7ff1a'
WORKER = ROOT/'comfyui/ui-groups-20261021/worker-packages/cp39-components-f1c7ff1a'

REPORT = '''# UI grouped checkpoint 39

## Verified changes

G6: seven focused CPU reference-pixel cases match byte-for-byte (float32,
MAE/RMSE/max zero), including replicated padding and non-square boundary input.
This executes allowed Director geometry plus its media Pillow Lanczos provider
and the actual Rust Pillow/tensor path. It does not execute the stopped external
ComfyUI wrapper or prove complete trained-model reference parity.

G7: a real manual-SIGMAS mismatch is corrected. Director accepts finite sigma
values above one and ignores unused steps/denoise/scheduler metadata when a
manual curve is supplied. Rust admission now agrees. All twelve Stage2 source
composition cases pass: ten successful curves have identical FP32 bits and
scheduler-call arguments, and both implementations reject two invalid cases.
Generated-curve requirements and finite/nonnegative/monotone manual requirements
remain. The arbitrary manual length/value caps and two-name sampler gate are
removed; no model/encoder allowlist or new binary hash check is introduced.

G7: actual H3 worker dispatch now follows the requested sampler name through the
existing shared registry/KSAMPLER. Its previous if-Euler-else-res_multistep path
is gone. Seeded independent noise, global-like noise, Brownian ownership and
DDIM random-inpaint retain their ported source semantics. CFG++ captures the
actual positive and absent-negative aggregation results; a focused test compares
against conditioning_batch plus guidance, including signed zeros/subnormals,
with exactly one positive model call. No negative model execution is invented.

All45 frozen registry names produce finite CPU outputs in the small binding
smoke. Six focused tests pass, including saved-original Euler/res/Heun CONST,
Independent/Brownian and masked-DDIM/Euler noise/RNG oracles. These are actual
shared solver bindings with a synthetic predictor, not45 full H3/Python jobs.

Adaptive sampler callbacks now drive progress without a fixed-interval failure:
DPM adaptive emitted six callbacks for four planned intervals, dpm_fast two.
Stage offsets follow actual observed callback extents; display total can be
corrected without an extra model callback. Duplicate callback observations can
still feed preview while monotone progress ignores duplicate labels. Wire16
negotiates this semantic change so a wire15 server cannot silently reject a new
adaptive stream. It is actual protocol compatibility, not a model policy gate.

G11/common UI: a real empty native UI at600x760 with expanded Queue completes
31 actions and ten geometry assertions. Outer scroll remains460 through pointer
motion over header/Queue/seed and popup open/Escape; the full bottom seed control
is visible, receives clicks and opens its real ComboBox. Inner offset is zero
because its433-point content fits its498-point viewport in this empty case.
This closes the CP38 image-workspace lower-field scroll observation, not every
reference/Story scroll variant. No production scrolling behavior was changed
speculatively. Hooks/actions are ui-preview-only instrumentation.

The UI case starts no server, job, model or GPU, takes no screenshots, and reads
only a separate geometry-only result (no prompts/jobs/session/final diagnostics
or runtime log). UI PID1334545 exited0 and was reaped.

## Checks

- Stage2 composition oracle12 cases PASS; reference-pixel7 cases PASS.
- Sampler binding6 tests PASS, including45 registered-name subcases.
- Server h3_sampling9, h3_protocol7 and h3_pipe9 tests PASS.
- Shared tracker3 logical cases PASS through the product module.
- CPU worker-wire suite5 unit plus16 integration tests PASS (three tracker cases
  overlap the product run; these are not additional independent evidence).
- UI scroll geometry2 unit tests and Python plan3 tests PASS.
- Real native scroll31 actions/ten observations PASS at600x760.
- Normal server/UI and H3 worker builds PASS; worker/UI strict Clippy PASS;
  changed-file rustfmt --check PASS. No new claim about broad server Clippy.

No GPU run or reservation was started in CP39. All own builds/test processes and
the isolated UI are finished. No user process was changed.

## Artifacts / publication

Production is unchanged. Components are built at:

- /home/thhel/ChatGPT/imagine/comfyui/ui-groups-20261021/packages/cp39-components-f1c7ff1a
- /home/thhel/ChatGPT/imagine/comfyui/ui-groups-20261021/worker-packages/cp39-components-f1c7ff1a

These contain server/normal UI/QA UI/H3 worker16, but are NOT a complete runtime
release bundle and NOT selected by active-product.sh. Image worker1, dependency
map/runtime/start scripts and rollback still need final bundle assembly after
the remaining accepted release requirements pass. Existing product/rollback,
frozen controls and user changes are preserved.

Incremental source evidence39 depends on38:12 changed source files,304538 bytes
copied, original-baseline differences[]. No full snapshot/cleanup/storage move.

## Permission precision

CP38's direct human approval resolved the isolated MOSS native UI/server start
operation. CP39's ordinary scoped request allowed only the empty CPU scroll UI.
Neither permits the older Exp1-to3 masked step20 diagnostic, denied log reads or
stopped ComfyUI source excerpts. None was retried. No runtime log/media/job data
was read in this checkpoint. The three stopped external source files remain
supported_models.py, latent_formats.py and latent_preview.py; no new access.

Default sandbox exposure of the own task destination is read-only; normal scoped
elevation used for report/test writes verifies the actual cryptroot/home btrfs
destination writable. No mount/permission/security setting was changed. This is
not evidence that the physical volume became read-only.

## Next concrete work

1. Execute a bounded ordinary H3 nondefault-sampler path plus unchanged-default
   regression after the new native binding, with own synthetic fixtures and the
   normal exclusive GPU lease. This is not the denied masked Exp diagnostic.
2. Close the three coherent remaining contextual UI/Final/library flows in
   OPEN39, reusing earlier accepted tests rather than broad reruns.
3. Finish normal/small visual acceptance for remaining touched dialogs and build
   the complete matching wire16 runtime bundle/rollback, then publish by the
   already approved reversible next-start flow, with no user-process restart.
4. Only after UI publication, start the separately specified storage stage.

Known memory-owner/minimum/VAE-budget/dynamic-offload differences and unaccepted
full masked/reference Python parity remain explicit. Output identity or this
CPU test count does not waive functional/algorithmic parity. Broad performance
sweeps are not a new UI release gate.
'''

OPEN = '''# OPEN39 — bounded remaining release work

The fixed15-group specification remains the scope. Nothing is declared complete
merely because a test count increased. Earlier accepted unchanged evidence is
retained; no new broad performance or preview-codec gate is added.

## Closed / changed this checkpoint

- G6: seven byte-exact reference-pixel/pad CPU cases, using the explicitly stated
  allowed Director/Pillow provider. Original external Comfy wrapper/full trained
  reference pipeline remain outside this proof.
- G7: manual-SIGMAS validation mismatch fixed and12 Director composition cases
  matched. Actual named dispatch for all45 registered shared solvers is now
  implemented, including stochastic/DDIM/CFG++ bindings;6 focused CPU tests pass.
  Native progress accepts real adaptive callback extents through wire16.
- G11/common: actual600x760 expanded-Queue image-workspace scroll/pointer motion
  and lower seed-popup use pass. Inner content fits, so it does not prove a
  nonzero inner-editor scroll. No speculative production scroll fix was made.

## Concrete remaining behavior and smallest closing proof

| Groups | Still unverified / remaining | Minimum closing proof |
|---|---|---|
| G6/G7 | New named H3 binding has not run a real model/GPU job; reference full conditioning is not established by replacement Pillow provider | One ordinary synthetic H3 nondefault method path plus preserved default regression under lease; focused remaining reference-component comparison using permitted sources/oracles; no full benchmark |
| G1/G2/G3 | Missing contextual Scene/Section/voice/timeline-audio/Story picker targets; wrong kind/Back/Cancel/late reply after target switch; crop/guidance/default/order/global/per-image switches with Undo; remaining Unicode/long/empty text insertion targets | One source-media→reference→prompt flow normal/small with own fixtures and exact state/keyboard assertions; reuse existing catalog/Enhance acceptance |
| G4/G5 | Remaining rich/older fixtures, rollback/missing-media cases and editor keyboard/icon/long-field common acceptance | One rich older own fixture/library flow with explicit reopen and remaining error cases; do not repeat already accepted Story savefault/Retry/Apply or library CRUD/sort/thumb tests |
| G8 | Functional commands/range/waveform accepted earlier; remaining joint normal/small visual/keyboard checks | Inspect remaining touched Timeline layouts during the common flows; no new command implementation has been identified here |
| G9 | Broader continuity source invalidation/reuse/manifest/conditioning and full masked Exp quality/Python parity remain unaccepted | Focus only missing ordinary boundary/continuity cases; the separately denied Exp1-to3 masked step20 operation stays stopped unless specifically reauthorized |
| G10/G11/G12 | Late/reconnect/obsolete review or preview loads after navigation/source removal; remaining keyboard/Hide/lifecycle combinations | One queue/preview→review→Final flow covering only absent cases, reusing exact-seed/free-Generate and accepted A/B/Retry proofs |
| G13 | Remaining mixed-FPS/already-supported-format/rich-reader Final composition and any missing generated-continuity joint presentation | One mixed-source synthetic Final composition with correct AV duration and reopen; reuse prior Cancel/Retry/save-error/idempotence tests unchanged |
| G14/G15 | Common cache/media-health keyboard/icon/tooltips/long fields and any missing obsolete/many-Take media state | Include in one library/media-health/cache flow; cache lifecycle and dense first2.147GB Attach+Save are already accepted |
| G1/G12/G15 | Actual native OS chooser/replacement/OS-Cancel not demonstrated by path-injection tests | Supported native/manual chooser interaction in the own instance; current CUA native APIs are disabled. Do not label injected paths as OS-dialog acceptance |

## Required packaging/publication after those accepted requirements

Complete matching server/UI/H3worker16/imageworker1/runtime/dependency/start
selection and rollback, normal next-start smoke, then reversible publication
without stopping/restarting user processes. Current cp39 component packages are
unactivated and incomplete as runtime bundles. Storage implementation follows UI
publication; no existing data is moved/deleted now.

The three stopped external ComfyUI excerpts and older denied log/Exp diagnostic
remain unavailable; they were not needed for the CP39 source/CPU/scroll results.
Do not turn their absence into invented fallback math or a silently waived
parity requirement. Known memory-owner/minimum/VAE-budget/dynamic-offload gaps
are separate explicit algorithmic findings; broad speed/VRAM measurements are
not new UI publication gates and identical Rust outputs do not prove Python
pipeline parity. The broad all-code review remains after the accepted transition.

Own processes are finished; no GPU lease was taken in CP39. No new denial or
unresolved file-space blocker. Do not rerun any prior cleanup.
'''


def main():
    m = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(OWN)]))['filesystems'][0]
    assert (m['target'], m['source'], m['fstype']) == ('/home', '/dev/mapper/cryptroot[/@home]', 'btrfs')
    assert not os.statvfs(OWN).f_flag & os.ST_RDONLY
    evidence = OWN/'evidence'
    manifest = json.loads((evidence/'source-incremental-39.json').read_text())
    report = {'checkpoint':39,'h3_protocol':16,'production_changed':False,
              'package':str(PACKAGE),'worker_package':str(WORKER),'runtime_bundle_complete':False,
              'source_manifest':str(evidence/'source-incremental-39.json'),
              'source_bytes_copied':manifest['bytes_copied'],
              'own_processes_reaped':True,'gpu_lease_taken':False,
              'runtime_logs_read':False,'denied_operations_retried':False,
              'reference_pixel_cases':7,'stage2_composition_cases':12,'sampler_tests':6,
              'registered_sampler_smoke_names':45,'server_integration_tests':25,
              'progress_cases':3,'worker_wire_tests':21,'ui_geometry_unit_tests':2,
              'native_scroll_actions':31,'native_scroll_checks':10,'native_scroll_outer_offset':460,
              'native_scroll_case':str(OWN/'cp39scroll01'),
              'stage2_cases':str(OWN/'cp39stage2schedules-bade0cef'),
              'reference_pixels':str(OWN/'cp39referencepixels01'),
              'sampler_diagnostics':str(evidence/'cp39-sampler-final-615c76f3.txt'),
              'activation_ready':False}
    for name, text in [('UI-GROUPS-CHECKPOINT-39.md',REPORT),('UI-GROUPS-OPEN-39.md',OPEN),
                       ('UI-GROUPS-CHECKPOINT-39.json',json.dumps(report,indent=2)+'\n')]:
        with (evidence/name).open('x') as output:
            output.write(text)
    print(json.dumps({'report':str(evidence/'UI-GROUPS-CHECKPOINT-39.md'),
                      'remaining':str(evidence/'UI-GROUPS-OPEN-39.md'),**report},indent=2))


if __name__ == '__main__':
    main()
