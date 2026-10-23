#!/usr/bin/env python3
"""Persist source-fixture acceptance only. Never reads logs or runtime jobs."""
import json
import os
from pathlib import Path
import subprocess

OWN = Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')

REPORT = '''# CP41 — remaining import/library/continuity boundaries

## Implemented and verified

- G1/G2: a source-fixture unit reproduction FAILED before the fix: an older
  same-library reply replaced a newer contextual-picker list. Library queries
  now carry their own monotonically advancing request token across picker
  reopen/refresh. The stale reply cannot replace the current list or selection.
  This is asynchronous result ownership, not a model/file allowlist.
- G1: malformed own PNG actually reaches the upload/decoder path. Failure leaves
  the Cast/Story/Section bindings intact, then a valid Project-assets choice
  recovers. Visual inspection found that the stored import error was behind the
  picker; the picker now shows the error belonging to that captured target.
  Success clears it. No fallback, size quota or new model restriction.
- G1/G3: separate Story empty/nonempty source Cancel assertions and Section
  video-Cancel now pass in real native1100x760 and600x760 UI. Final cases complete
  55 actions each,3.2153s/2.9344s; error/recovery screenshots are inspected.
- G5: new own handwritten older CharacterDocument without `reference` and richer
  reference/voice/relationship fixture open without rewriting input manifests.
  Cancel, keyboard Tab editing, long Unicode, Save/reopen and missing-media
  rollback/catalog retention pass. The source v1/v2 profile-resolution regression
  remains a distinct existing test, not a live provider or Python-quality claim.
- G5: a missing library blob previously displayed a misleading project-Save
  error. Only the Open reply's known storage/file failures now describe opening
  library files; Save/F05 and session error handling stay distinct. Focused unit
  test passes. Native afterfix verifies readable exact error, preserved two-entry
  catalog, Refresh and successful recovery Open in normal/small windows.
- G5/common: simultaneous Library and character editor emptied the single-owner
  thumbnail cache each frame. In-memory thumbnail keys now include the actual
  project/library namespace and asset ID; version and pending-reply ownership
  still apply. Library callers share their library namespace. Actual project/
  session replacement clears the retained map. No persistent thumbnail storage,
  model change or new budget is introduced. Native screenshots now show the own
  blue PNG in the library and editor in both window sizes. Two focused cache
  tests and existing revision/duplicate behavior pass.
- G10/G11/G12: four source-only lifecycle tests exercise actual event handlers:
  late preview/comparison success cannot reopen a closed/superseded viewer;
  project replacement cancels the prior decoder and rejects old chooser/import/
  preview replies; a removed captured target never falls back to the current
  picker; partial failed import retains available resources without binding.
  These use unsubmitted source templates and tiny procedural arrays, no network,
  disk/media decode, job history, model result or live user document.
- G9: two direct typed ProjectSession/Service/freeze tests verify separate reuse,
  style and audio-off invalidation, preserved candidates/resources, frozen source
  bytes/tail/manifest, pending receipt, own pre-submission prompt template and
  Save/reopen. Director's normal seam uses reuse prompt phrases; Combo/Influence
  early returns do not invent a picture-conditioning toggle. A stale tracked Take
  remains Outdated even after explicit Accept, matching Director normalization;
  restoring its actual dependency enables explicit Accept and reopen. No product
  inference change, scheduler/worker/GPU or new full Exp diagnostic was needed.

## Checks and limits

Normal server/UI build PASS. Strict QA UI Clippy PASS. Picker unit suite,
four lifecycle tests, Open-error unit and two thumbnail tests PASS. G9 two direct
server-lib tests PASS. Changed-source rustfmt and Python syntax checks are recorded.
The first G9 attempt had an incorrect expectation that explicit Accept overrides
tracked stale dependencies; the test was corrected against Director, not the
product. Library harness first had an unstable post-Save row index and then a
missing active fixture project for a generic QA assertion; both are known test
errors, not hidden product failures. No blind timeout increases or retries.

Core library native flows: normal98actions15.49s, small98actions14.39s. Focused
afterfix:37actions5.087s/4.806s, four screenshots inspected. Full prior Story
provider/savefault and catalog suites were not repeated unchanged. All own
CPU UI/server/build processes are finished. No GPU reservation was taken in CP41.
No runtime logs, user data, actual submitted prompts/jobs or model-generated media
were read. No new denial, cleanup, storage move, commit, push or product restart.

## Runtime packaging source mismatch

The standard publisher formerly read the outer checkout's protocol8 even when
staging the isolated candidate's protocol16. It now accepts explicit --source-root
for the source from which supplied artifacts were built. Direct source check
distinguishes8 and16. The candidate staging command must pass
/home/thhel/ChatGPT/imagine/comfyui/ui-groups-20261021/source. No staging or
activation was executed here. This checks the actual worker/server protocol
contract, not model names or historical binary hashes. A small separate patch and
source hashes preserve the publisher change; the incremental support copy includes
the current tool. Current normal binaries are built, but no complete matching
runtime/imageworker1/H3worker16/dependency/start/rollback bundle exists yet.

## Exp/Python: exact release distinction

Full masked Exp/Python numerical/quality parity is an OPEN algorithmic porting
requirement, not accepted or cancelled. CP38's short ordinary Exp+trained Stage2
proves that path can execute; it does not prove every20step full-scale output.
CP41's remaining UI reuse/invalidation/freeze decisions are deterministic typed
operations and are now tested without changing Exp model arithmetic. They do not
technically require rerunning the old denied first1-to3 masked diagnostic.

For this UI delivery, no NEW full-Python sweep is justified as a new acceptance
gate solely by these UI changes. A demonstrated functional Exp regression would
still block G9 and require a permitted targeted test/fix. The broader existing
memory-owner/minimum/VAE-budget/dynamic-offload and full masked/reference parity
findings remain release qualifications; a UI test candidate must not be called
generally algorithmically equivalent. A decision to publish for Tomas's testing
does not constitute acceptance of those porting requirements. They must remain
explicit in the candidate/publication report. No denied operation is reopened.

## Manual OS chooser — actual click list

The supported CUA surface has native APIs disabled. Browser automation cannot
operate egui/rfd OS dialogs. Injected paths skip the OS selector/Cancel and cannot
prove it. No portal/IPC/input workaround or system-setting change is used.

Use the separate own CPU instance, not production. Prepared scripts:
normal/start-cp41-server.sh then normal/start-cp41-ui.sh under
/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01/cp40-picker-manual.
They select the immutable cp41-ui-picker-cdff395e QA package, port18491, own state,
no QA input Runner, initial OS directory set to the own fixture folder.
Do not start UI if the own server failed. No Generate/provider/Final render needed.

Open project folder:
/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01/cp40-picker-manual/normal/saved-project

Exact six own fixture paths:

```
/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01/cp40-picker-manual/fixtures/original.png
/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01/cp40-picker-manual/fixtures/voice.wav
/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01/cp40-picker-manual/fixtures/clip.mp4
/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01/cp40-picker-manual/fixtures/replacement.png
/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01/cp40-picker-manual/fixtures/replacement.wav
/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01/cp40-picker-manual/fixtures/replacement.mp4
```

1. Video > References/Media > Cast & references > Lighthouse keeper > Images >
   Add reference images > Folders > Import from files. Select original.png then
   OS Cancel: no binding/busy remains. Repeat with OS Open: exactly one image on
   this reference; Visitor unchanged. Imagine Cancel is a separate button/test.
2. Same reference > Voice profile > Add voice references: OS Cancel, then select
   voice.wav. Check exactly the intended voice target, original image unchanged.
3. Selected Section > Attach images/Attach videos: Cancel then select original.png
   and clip.mp4. Check the selected Section, not another Section/reference.
4. Attach Audio > Import from files: Cancel then voice.wav. Story > Story outline
   > Cast & source images > Add source images: OS Cancel without provider call.
5. Takes > Open Captures > Attach video as Take: Cancel then clip.mp4. One external
   Take on the chosen Scene, honest imported-video provenance, no generation.
6. Project > Check Media / Relink > Relink > Choose replacement file: OS Cancel;
   then replacement.png > impact > Back; then confirm. Check existing bindings
   now show the replacement. Repeat Cancel/type choice for replacement.wav/mp4.
7. Save and reopen only this own project. Repeat image Cancel and Relink/Back at
   small600x760 using small/start-cp41-* (port18492, small/saved-project). Check
   Tab/Escape/focus, reachability/scroll and correct target. Close only own UI and
   own server. Logs need not be read.

The OS point is still unaccepted and stays before publication unless Tomas
explicitly moves ONLY that manual test to his first sharp test. Preparing the
scripts, or passing injected-path/native Imagine controls, is not that approval.
'''

OPEN = '''# OPEN41 — bounded finish, no new general gates

Closed in41: source-owned picker library replies/import-error presentation;
Story/video Cancel and malformed-file recovery in native normal/small; older/rich
library/missing-media rollback and keyboard/long fields; library Open error and
simultaneous editor thumbnail; late preview/comparison/project/import handler
ownership; ordinary G9 reuse/style/audio invalidation→freeze→receipt/reopen.
Existing accepted command, sampler, Stage2, A/B, Final AV, cache/Relink and dense
Attach+Save proofs remain valid; do not rerun them just to increase test counts.

Actual next release work:
1. Complete the JOINT normal/small visual pass for remaining Review/Final and
   media-health/cache panels, including requested consistent repeated tool icons,
   tooltips, labels/long fields and navigation/Hide. Examples still needing that
   shared presentation check are Cast Reload/Remove and cache Refresh/Clear;
   this is an existing requested UI criterion, not new feature scope.
2. Assemble and smoke the matching NORMAL source-built server/UI, H3worker16,
   imageworker1, runtime libraries/dependency map/start selection and rollback.
   Publisher must use --source-root C/source (outer source is8, candidate16).
   QA component packages are not production bundles; do not publish a QA build.
3. Human native OS chooser/Cancel/replacement G1/G12/G15, exact click list in
   CHECKPOINT41. It remains before publication unless Tomas explicitly moves
   that point to his first sharp test. Continue independent package preparation.
4. Publish only after actual accepted UI finish/decision, without stopping any
   user process; then the separate storage stage. No current selection changed.

Full masked/reference Python parity and known memory-algorithm gaps remain open
porting requirements. CP41 does not waive them or prove them through a UI test.
The specific old Exp1-to3 diagnostic stays stopped and is not demonstrated
necessary for the typed UI/freeze decisions just closed. Broad performance
measurement is not a new UI gate. Clearly qualify any test-candidate publication;
do not label all H3 algorithms accepted. No additional product functionality gap
was proven in41 beyond the concrete UI defects fixed above.

All own processes are finished, no GPU lease in41, no new denial or file-space
blocker. No cleanup/user-data/storage move or process restart. Incremental41
depends40; no full snapshot. Next main task is item1 plus package preparation.
'''


def main():
    m = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(OWN)]))['filesystems'][0]
    assert (m['target'], m['source'], m['fstype']) == ('/home', '/dev/mapper/cryptroot[/@home]', 'btrfs')
    assert not os.statvfs(OWN).f_flag & os.ST_RDONLY
    evidence = OWN/'evidence'
    manifest = json.loads((evidence/'source-incremental-41.json').read_text())
    # These files contain only source-fixture test assertions/geometry and own
    # process exits. They contain no submitted prompts/jobs/model media/history.
    cases = [OWN/name/'evidence/result.json' for name in ['cp41contextcanceln02', 'cp41contextcancels02']]
    for path in cases:
        value = json.loads(path.read_text())
        assert value['passed'] and value['own_processes_reaped'] and not value['jobs_submitted']
    data = dict(checkpoint=41, production_changed=False, activation_ready=False,
                h3_protocol=16, runtime_bundle_complete=False,
                own_processes_reaped=True, gpu_lease_taken=False, runtime_logs_read=False,
                denied_operations_retried=False, os_picker_accepted=False,
                source_manifest=str(evidence/'source-incremental-41.json'),
                source_bytes_copied=manifest['bytes_copied'],
                cancel_cases=[str(p) for p in cases],
                library_afterfix=str(OWN/'cp41library-report01/AFTERFIX.md'),
                manual_prepared=str(OWN/'cp40-picker-manual/CP41-CURRENT.md'))
    for name, text in [('UI-GROUPS-CHECKPOINT-41.md', REPORT), ('UI-GROUPS-OPEN-41.md', OPEN),
                       ('UI-GROUPS-CHECKPOINT-41.json', json.dumps(data, indent=2)+'\n')]:
        with (evidence/name).open('x') as output:
            output.write(text)
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
