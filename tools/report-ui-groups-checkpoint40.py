#!/usr/bin/env python3
"""Static checkpoint plus numeric synthetic-component results, never runtime data."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/home/thhel/ChatGPT/imagine')
C = ROOT/'comfyui/ui-groups-20261021'
OWN = Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')

REPORT = '''# UI grouped checkpoint 40

## Real H3 sampler boundary

Four focused real-DiT GPU cases pass with deterministic numeric conditioning
(four tokens), explicit seed42 noise and empty AV streams. Shape416x224,
39frames24fps; Ref2VA DiT BF16, actual product component loader/owner, Kitchen
bridge/Aimdo and PyTorch attention; Triton disabled. No tokenized text, submitted
prompts, jobs, decoder, model-generated images/audio, latent dumps or export.

| Method | Manual SIGMAS | Observed progress | Seconds |
|---|---|---|---:|
| res_multistep | 1,.6,0 | 1/2,2/2 | 2.283925536 |
| heun | 1.2,.6,0 | 1/2,2/2 | 2.480698898 |
| euler_cfg_pp | .9,.6,0 | 1/2,2/2 | 1.686719896 |
| dpm_fast | 1,.8,.6,.3,0 | 1/4,2/4,2/2 | 3.322839320 |

Both streams are finite in every case. Dynamic progress closes at the actual
two callbacks instead of failing against four planned intervals. Component
total15.741457229s includes load/unload and is not a performance comparison.
The numeric result contains exact stream shapes/RMS/peak. A latent peak above
one is not evidence of decoded clipping or picture quality.

Two earlier own test attempts failed before completing the four rows. The
structured source-stage marker isolated CFG++: CONST half_log_snr is -logit(sigma)
and the harness had used1.2 outside that algebra's finite domain. Only that test
curve changed to.9; Heun still exercises1.2. A CPU regression proves the original
unclamped algebra (NaN at1.2, -inf at1, finite at.9). No product sigma clamp,
sampler policy gate or silent fallback was added. Console logs remain unread.

The successful own worker PID2211165 exited0 and was reaped; normal exclusive
coordinator ticket finished/closed. Transient nonzero aggregate utilization
postponed admission; no other process was stopped. All GPU resources are released.
This closes the changed named-dispatch/SIGMAS/progress model component smoke,
not a45-sampler benchmark, a full Generate quality test or Python pipeline parity.
CP39's exact saved CPU solver/RNG oracles remain the numerical control.

## Final: a real VFR soundtrack-range defect fixed

Two new procedural source fixtures execute actual Rust freeze/prepare/render and
allowed pure Director media/audio/sequence functions. No existing runtime
project, job, prompt, generated model media or log is read.

MOV24fps plus MP4 30fps to30000/1001 delivery already matched. The mixed
VFR-Matroska/MP4 case exposed an audio difference: the Matroska container reports
2s, but its24 decoded frames at24fps define Director's1s source interval. Rust
used the container duration for the soundtrack and fitted2s of sound to the
picture. Direct PyAV/FFmpeg decode of that own Matroska source was byte-identical;
the differing trim interval, rather than gain or the AAC decoder, was causal.

Rust now derives the soundtrack's source start/end from the same selected-frame
range as the picture. A missing declared frame count uses a cancellable one-pixel
decode pass instead of buffering full RGB solely to count it. The existing
bounded process/reader and errors are reused. This adds a decode/count pass only
when the source has no declared frame count; no new arbitrary format/model gate.

Before fix the VFR raw PCM MAE/RMSE/max was
.0237682581/.0382385418/.1054864749. After fix both cases have raw RGB float32
MAE/RMSE/max0, integer RGB max0 and PCM max1.4901161193847656e-8.
MOV mixed case PCM MAE6.79003e-10/RMSE1.86104e-9; VFR mixed case
MAE6.75501e-10/RMSE1.84616e-9. Export is direct H264/yuv420p/AAC stereo44.1k,
TV-range BT.709, with48frames1.601610s and45frames1.875011s respectively.
Expected durations are1.601600s and1.875000s. No lossless intermediate or gain
compensation was introduced.

## G1/G2/G3 coherent native editor flow

Each normal1100x760 and small600x760 run completes246 actions. Project-assets
selection/Imagine-Cancel, exact correct-reference image/voice IDs, wrong-target
reference remaining unchanged, Section image Auto-anchor and video, Section
switch, Timeline audio and Story source choice are exercised. No provider,
Generate, GPU or user data is involved.

Per-image enabled/default behavior, reference enabled/global visual switch,
Kind/Relationship live update without Apply, Undo/Redo, voice language/guidance,
save/reopen and image order survive. Scene caret completion preserves its suffix:
the own template 'å🙂 (@kee rest' becomes 'å🙂 (@keeper rest'. Long/empty/Unicode
Cast fields are preserved. Existing catalog/Enhance tests are reused, not rerun.

This is Project-assets transport and native Imagine controls, NOT real OS picker
interaction. Video-Cancel, Story-source-count assertion after Cancel, malformed
file and superseded in-flight target replies were not separately asserted in
this246-action sequence. Prior independent component/Relink proofs are retained;
the absent native/late-target cases remain explicitly bounded below.

Visual QA exposed duplicate egui Custom value IDs for Kind and Relationship,
then a real small-window clip. The ID now has the existing semantic field's own
push_id scope. Rows wrap, Alias and Kind are separate, window/content widths
follow the viewport, and Transfer target/ID are distinct label+field rows.
No reference semantics or allowed values changed. Full semantic flows were not
repeated for each layout revision; focused36-action editor cases verify the
changed layout. Final normal/small screenshots are inspected, not inferred from
the assertions. Emojis remain intact codepoints but show the installed font's
missing-glyph box; no font installation or new product restriction was attempted.

The fixed G2 specification does not require new crop/reorder controls. The vague
crop/reorder remainder in OPEN39 is corrected rather than implemented as extra
scope. Import/default order preservation is actually tested.

## OS picker: prepared, not accepted

The own manual CPU fixture and six procedural files are at cp40-picker-manual.
CURRENT-PACKAGE.md points to immutable cp40-manual-6831df7b, own loopback
ports18491/18492 and separate state/cache/output. start-cp40-* scripts pass four
shell syntax checks. No manual instance was started by preparing these files.

Test-feature-only --qa-manual-picker-root starts the real OS chooser in our own
fixture directory without installing a QA Runner/input filter. It is absent from
the normal product UI. Current supported CUA native APIs are disabled, while
injected paths skip rfd and cannot prove OS Cancel/selection. No alternative
portal/IPC/input/security route is used. The manual click list covers reference
image/voice, Section, audio, Story, external Take and Relink Cancel/Back/confirm.

The human OS-dialog check remains before publication under the accepted order.
Moving exactly that point to Tomas's first sharp test requires his explicit
decision; this report does not make that decision or waive other acceptance.

## Verification and packaging precision

Actual changed-source checks: strict inference LIB-UNIT-TEST Clippy via
cargo clippy --lib -- --test -D warnings PASS; five affected CPU cases PASS;
focused existing trim/blend/stereo/duration regression PASS; server library
Clippy PASS without warnings; normal server/UI build and strict QA UI Clippy
PASS. Changed-file rustfmt check is recorded separately. The initial broader
--tests invocation also selected unrelated integration targets with two absent
static stage3 include fixtures; it did not pass. Three unit-test lint findings
were corrected (one new needless borrow, two older test-only loop/discard uses).
The narrower lib-test Clippy is not presented as a full inference integration run.

Production active-product.sh remains unchanged. cp40-context-qa-04 is a frozen
QA/CPU component package, not a complete selected runtime bundle. Current
normal server/UI target binaries include the Final/Cast changes. H3 protocol
remains16. Final matching imageworker1/H3worker16/runtime/dependency/start and
rollback bundle still follows remaining acceptance. No publication, process
restart, storage change, deletion, commit or push occurred.

Incremental40 depends39; actual manifest records changed paths and hashes.
No full source snapshot or cleanup. Only normal scoped elevation wrote to the
own moved task root; no mount/security setting changed. No denied log/source
read or old Exp operation was retried. Own workers/UI/server/builds are finished.

## Exact old Exp denial — no new approval request

See EXP-APPROVAL-SCOPE-40.md. The old proposal extended one early masked DiT
callback to three early callbacks in the same20-interval Exp schedule. It was
not a request to run only the20th step. The recorded refusal was a scope/stop-
condition decision, not a stated privacy or model-weight risk. One first step
had already matched byte-for-byte. The proposed three-step patch was not applied
and no three-step worker ran. After CP38's separate ordinary Exp/Stage2 proof,
the exact old diagnostic is not demonstrated necessary. Its absence alone is
not a newly invented release gate; the remaining functional/algorithmic Exp
requirements are still not silently accepted. A question about it is not approval.
'''

OPEN = '''# OPEN40 — remaining accepted release work

Scope is the fixed15 groups, not an expanding permutation matrix. Previously
accepted unchanged proofs remain valid. No full sampler/performance benchmark
or new preview codec/quality selector is added.

## Closed by this checkpoint

- G7: actual BF16 DiT executes default res_multistep plus Heun/manual sigma>1,
  CFG++ in its finite domain and dpm_fast dynamic progress on numeric fixtures.
  Output quality/whole Python pipeline are not implied by finite latents.
- G1/G2/G3: coherent Project-assets→Cast/voice/Section/audio/Story flow, live
  properties/default/enabled/Undo/reopen/Unicode and correct target IDs in both
  sizes. No new crop/reorder implementation is required by the fixed G2 text.
- G2/common: actual duplicate ID and600-wide Cast clipping fixed and visually
  rechecked; correctness is not inferred merely from246 action assertions.
- G13: mixed MOV/MP4 and VFR Matroska/MP4 Final readers now match Director raw
  picture/sound and export clock/color after fixing soundtrack source-range use.

## Concrete remaining minimum, not generic 'partial'

1. Contextual import lifecycle: preserve target/project identity on a late import
   reply after navigation, wrong-kind actual decoder error, and any missing
   Story/Section cancel assertion. Reuse the shared picker/Relink component tests
   where they already prove the exact case; do not repeat every picker permutation.
   Real native OS chooser/Cancel/replacement remains a distinct human point.
2. G4/G5: one rich/older OWN library fixture through missing-media/rollback and
   remaining editor keyboard/icon/long-value presentation. Keep CP35 accepted
   Story savefault/Retry/Apply and library CRUD/sort/thumb; no duplicate full suite.
3. G9: close only absent ordinary continuity source-invalidation/reuse/manifest
   cases with their actual conditioning behavior. CP34's four boundary GPU cases,
   CP38 ordinary Exp+trained Stage2 and prior component masks remain evidence,
   not complete masked/reference Python quality proof. Old denied1→3 diagnostic
   stays stopped; do not make it a mandatory new operation without necessity.
4. G8/G10/G11/G12/common: one remaining coherent preview/review→navigation or
   source removal/obsolete response→Final presentation, with normal/small
   keyboard/Hide/icon/tooltips. Retain existing commands/waveform/A-B/Retry/exact-
   seed/free Generate proofs instead of implementing unrequested commands.
5. G13/G14/G15/common: remaining joint Final/media-health/cache presentation and
   keyboard checks in that flow. Dense first2.147GB Attach+Save, stable Relink,
   cache lifecycle/budget/clear and existing Final cancel/retry/save-error are
   already accepted. CP40 closes the newly exercised VFR/MOV raw-AV issue.
6. Human OS-dialog point G1/G12/G15: follow cp40-picker-manual/CURRENT-PACKAGE.md
   and README.md with six own procedural files. CUA native APIs are disabled;
   path injection is not OS acceptance. This remains before publication unless
   Tomas explicitly moves just this test to his first sharp test. Independent
   acceptance/packaging preparation must continue while that decision is pending.
7. Assemble matching normal server/UI, H3worker16, imageworker1, runtime libraries,
   dependency/start selection and verified rollback; next-start smoke using own
   state; then approved reversible publication with NO user-process restart.
   Storage implementation starts only after the UI publication.

The known memory-owner/minimum/VAE-budget/dynamic-offload algorithmic findings
remain open and must not be called accepted on the basis of output identity.
Full reference/masked Python parity is not implied by CP40's numeric model smoke
or Director's pure CPU montage oracle. These are explicit parity findings, not
a newly added broad speed/VRAM release benchmark. No requirement was waived here.

No user data/logs/generated media or stopped external ComfyUI excerpts were read.
No cleanup or global storage change. Current production still points at the old
selected product; cp40 QA packages are not activated. Own test processes and
normal exclusive GPU lease are finished. The next task is item1 followed by the
remaining two coherent library/preview/Final visual flows, then final packaging.
'''

EXP = '''# Exact scope of the older Exp denial

Source: this task's static CHECKPOINT-12.md and CHECKPOINT-13.md, already read.
No denied diagnostic, runtime log, external source or private file was reread.

- Purpose: identify where the historical near-silent Exp paired-audio output
  first diverged after a byte-exact first masked DiT step.
- 1→3 means expand the existing one-step component diagnostic to the FIRST THREE
  early sampling callbacks within the SAME20-interval res_multistep/simple
  schedule. 'step20' referred to that schedule, not the20th callback alone.
- Masked means existing Exp prefix/tail video+audio context is preserved by the
  actual denoise mask while the target portion is sampled. The prior component
  used exact physical source/tail, context/tags, mask, noise and21 sigma values.
- Own data: h3-e2e-08 saved source/tail (source originally h3-e2e-06), numeric
  video [1,24,47,14,26]/audio [1,32,2,263] prefixes; original20step shape416x224,
  visible124/sampled158frames24fps, seed42/shifts12/3. Exact prior instrumented
  source is archived under exp-step-01/artifacts/instrumented-source; future
  one-step code under diagnostics/exp-step-01/source. All are inside our known
  moved N=/home/thhel/tmp/.codex-video-night-20261020-a6428d9e task directory.
  The static record does not name an exact new three-step patch filename or
  configurable variable; none is invented here.
- Processes: it would use only our instrumented Rust worker and direct Python
  reference sequentially under the shared exclusive GPU reservation. No service
  restart, user media, model-weight change, VAE decode or export was involved.
  It would create additional raw latent/conditioning evidence under our task root.
- Earlier one-step proof: same actual nested-mask API,20 checked boundaries
  byte-identical, callback0 x/denoised video+audio MAE/RMSE/max0. An initial wrong
  Python packed-mask API attempt was NOT counted as parity.

Recorded refusal, verbatim:

> The patch extends the explicitly authorized single-step component probe into a multi-step diagnostic workflow despite the user's stop condition, so this scope expansion is not authorized.

Outcome: proposed three-step extension NOT applied; verified callback0 stop
remained; no three-step worker started and no alternative path was used.

Current necessity: not established after subsequent independent fixes and CP38's
ordinary Exp+trained Stage2 component/export proof. The exact old diagnostic is
not itself an identified necessary release operation now. Complete Exp/reference
quality and functional/algorithmic parity are not automatically accepted by that
statement. Continue the remaining permitted functional tests first. Only if a
remaining failure actually needs this isolation should a fresh narrow question
name exactly three early masked steps, identical own inputs, sequential Rust/
Python workers, exclusive GPU and new raw evidence in this task root. The user's
question about the denial is NOT that approval. No new request is made merely
to reopen old work.
'''


def main():
    m = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(OWN)]))['filesystems'][0]
    assert (m['target'], m['source'], m['fstype']) == ('/home', '/dev/mapper/cryptroot[/@home]', 'btrfs')
    assert not os.statvfs(OWN).f_flag & os.ST_RDONLY
    evidence = OWN/'evidence'
    manifest = json.loads((evidence/'source-incremental-40.json').read_text())
    gpu = json.loads((OWN/'cp40sampler03/evidence/result.json').read_text())
    final = json.loads((OWN/'cp40finalformats02/component-result.json').read_text())
    assert gpu['passed'] and gpu['own_process_reaped'] and gpu['own_ticket_closed']
    assert final['passed']
    report = dict(checkpoint=40, h3_protocol=16, production_changed=False,
                  runtime_bundle_complete=False, own_processes_reaped=True,
                  gpu_lease_released=True, runtime_logs_read=False,
                  denied_operations_retried=False, activation_ready=False,
                  gpu_component=str(OWN/'cp40sampler03/evidence/result.json'),
                  final_component=str(OWN/'cp40finalformats02/component-result.json'),
                  context_cases=[str(OWN/p) for p in ['cp40contextn01', 'cp40contexts01']],
                  source_manifest=str(evidence/'source-incremental-40.json'),
                  source_bytes_copied=manifest['bytes_copied'],
                  os_picker_accepted=False, manual_only_prepared=str(OWN/'cp40-picker-manual/CURRENT-PACKAGE.md'))
    for name, text in [('UI-GROUPS-CHECKPOINT-40.md', REPORT), ('UI-GROUPS-OPEN-40.md', OPEN),
                       ('EXP-APPROVAL-SCOPE-40.md', EXP),
                       ('UI-GROUPS-CHECKPOINT-40.json', json.dumps(report, indent=2)+'\n')]:
        with (evidence/name).open('x') as output:
            output.write(text)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
