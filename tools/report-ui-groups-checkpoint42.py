#!/usr/bin/env python3
"""Record the own procedural UI and package assertions, not runtime job data."""
import json
from datetime import datetime, timezone
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[1]
OWN = Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')
PACKAGE = ROOT/'comfyui/products/20261022T221439Z-a81a1f6f'


def main():
    tests = []
    for name in ['cp42visualn02', 'cp42visuals02']:
        path = OWN/name/'evidence/result.json'
        value = json.loads(path.read_text())
        assert value['passed'] and value['own_processes_reaped']
        tests.append(dict(path=str(path), width=value['width'], actions=value['actions_completed'],
                          seconds=value['seconds'], ui_exit=value['ui_exit'], server_exit=value['server_exit']))
    smoke_path = OWN/'cp42packagesmoke01/smoke-result.json'
    smoke = json.loads(smoke_path.read_text())
    assert smoke['passed'] and smoke['own_processes_reaped']
    metadata = json.loads((PACKAGE/'package.json').read_text())
    assert metadata['selected_for_next_start'] and metadata['h3_protocol'] == 16
    active = {k: shlex.split(v)[0] for line in (ROOT/'comfyui/active-product.sh').read_text().splitlines()
              for k, v in [line.split('=', 1)]}
    assert active['IMAGINE_PRODUCT_RUNTIME'] == str(PACKAGE/'runtime.json')
    source = OWN/'evidence/source-incremental-42.json'
    manifest = json.loads(source.read_text())
    report = dict(checkpoint=42, recorded_utc=datetime.now(timezone.utc).isoformat(),
                  published_for_next_manual_start=True, version=metadata['version'], package=str(PACKAGE),
                  selected_runtime=str(PACKAGE/'runtime.json'), runtime_bundle_complete=True,
                  protocols=dict(h3=16, image=1), normal_ui_qa_entry_absent=True,
                  rollback_selector=str(PACKAGE/'rollback-active-product.sh'),
                  user_processes_restarted=False, user_data_read=False, runtime_logs_read=False,
                  denied_operations_retried=False, own_processes_reaped=True, gpu_lease_taken=False,
                  storage_stage_started=False, os_picker='handed to user first sharp test; NOT tested',
                  os_picker_approval='Sentinel_94c7c7cc47d0819193f457b61caa580a',
                  native_tests=tests, package_smoke=str(smoke_path),
                  visual_report=str(OWN/'cp42visualreport01/REPORT.md'), source_manifest=str(source),
                  source_files_copied=len(manifest['copied_files']), source_bytes_copied=manifest['bytes_copied'],
                  original_baseline_differences=manifest['baseline_files_that_differ_in_original_checkout'],
                  full_exp_python_parity_accepted=False, broad_memory_algorithm_parity_accepted=False)
    text = f'''# CP42 — UI candidate published for the next manual start

Version: `{metadata['version']}`

Package: `{PACKAGE}`. The active selector was read back and matches this exact
server/UI/runtime. All four staged binaries were checked against their package
provenance digests after selection. This is explicit release evidence; it does
not restore historical runtime/model hash gates. Existing user processes were
not inspected, signalled, stopped or restarted. They continue their loaded
version until the user's own next start.

## Final UI fixes and acceptance

- Repeated Cast Reload/Remove, cache Refresh/Clear and media-health Check now use
  the existing 32px icon component, accessible label and tooltip. Review Refresh
  uses the same Retry icon. Primary Apply/Save/Cancel/Generate remain clear text.
- Actual normal/small native inspection exposed Final's filename/content while
  Hide was enabled and the window unfocused. Final now uses the existing private
  scope, keeping layout and scroll geometry while hiding private content.
- Final's right edge exceeded the narrow viewport; it now follows the existing
  viewport width rule. One duplicate Clear tooltip was removed.
- Final options, long filename, frozen plan, Review bindings/refresh tooltip,
  Hide, media-health/Relink Back, cache Apply/Clear/Refresh, Cast Reload/Remove,
  keyboard and Undo/Redo passed in actual isolated native UI, using only new
  procedural fixture documents/media. No inference, provider or Final render
  jobs were started by this visual suite.
- Afterfix normal1100x760:127/127 actions,22.3883s; small600x760:127/127,
  20.2245s. Both UI/server exits0 and reaped. Critical afterfix small screenshots
  `final-hidden.png`, `final-options.png`, `review-refresh-tooltip.png` were also
  inspected by the integration owner. Remaining inspected panels/screenshots
  are enumerated in `cp42visualreport01/REPORT.md`.
- QA build, two focused cache tests, strict all-target QA UI Clippy, changed-file
  rustfmt and final normal server/UI build passed. A lint-only QA test-module
  ordering correction moved the module unchanged to the end of qa.rs.
- Source evidence is incremental42→41:6changed source files,215869bytes copied;
  no additional complete source snapshot or deletion. Original baseline
  differences were empty. Previous unchanged GPU/component acceptance was reused
  rather than rerun for test counts.

## Matching runtime and actual package smoke

The package contains normal server/UI, imageworker1 and H3worker16. Image Kitchen
remains7fb6c097…e051b; H3 Kitchen remainsab7c807c…d4a3. Their actual accepted
dependencies and existing frozen library locations are preserved. The imageworker
is the CP37 artifact used by CP38's targeted GPU preview acceptance; it was not
replaced with the older active imageworker. H3 was rebuilt from current candidate
source. Embedded Rust tokenizers/templates require no separate prompt file.

The publisher now takes the actual candidate source root for protocol16 and can
include the matching imageworker; rollback also retains a copy of its imageworker.
No test observer destination is present in the production runtime manifest.

Isolated CPU smoke of the FROZEN package passed: real H3 Ready16 deployment
identity equals the packaged declaration; malformed zero-length framing returns
invalid_request before native construction/model access; normal server loads the
manifest and reports Rust/H3 available; five source profiles load; normal UI CLI
works and rejects QA-only entry. The smoke used a new own empty model/data root,
no jobs, no GPU and no runtime-log reads. Own processes exited/reaped. Selection
remained unchanged during smoke, then was atomically published after visual pass.
Imageworker Ready itself initializes CUDA and was intentionally NOT claimed by
this CPU smoke. Its unchanged CP38-tested artifact was verified when packaging.

## Start and rollback

At the next USER-controlled start, use the existing normal launch flow from
`{ROOT}`: `./start-server.sh` and `./start-ui.sh`. Existing proxy/API configuration
is unchanged. No user process was restarted automatically.

To select the preserved previous version for a later manual restart:

```sh
python3 tools/publish-video-product.py --rollback {PACKAGE}
```

Rollback binaries/manifests/selectors are inside `{PACKAGE/'rollback'}` and
`rollback-active-product.sh`; the original active selection is also retained.
Rollback was checked structurally, not activated (which would undo this release).

## Explicitly remaining

The user explicitly handed the native OS chooser to his first sharp test
(Sentinel_94c7c7cc47d0819193f457b61caa580a). It is NOT marked tested. For a neutral
test, create a new test project in the UI and use the six existing procedural
files under `{OWN/'cp40-picker-manual/fixtures'}`. Their exact paths exist. The
separate CP41 manual-server scripts create their own saved-project only when
started; that prepared project is not incorrectly presented as already existing.

1. Cast→Images→Add reference images→Folders→Import: Cancel, then original.png.
   Repeat voice Add with voice.wav; verify the chosen character, not another.
2. Section Attach images/videos and audio: Cancel, then original.png/clip.mp4/
   voice.wav. Story source Cancel should leave existing sources unchanged.
3. Captures/Takes Attach video as Take: Cancel, then clip.mp4.
4. Check media/Relink: Cancel; replacement.png/wav/mp4→Back should preserve the
   original; confirm only when the displayed impact is intended.
5. Save/reopen the own project; check Tab/Escape and scroll at a smaller window.

Full masked/reference Exp-Python numerical/quality parity remains an OPEN porting
requirement, as do separate memory lifecycle/budget/offload algorithm questions.
They are not accepted by native UI success, Rust/Rust equality, or this smoke.
The old specifically denied Exp1→3 diagnostic and external ComfyUI-file reads
were not retried. No new performance gate was introduced.

There is no newly known remaining UI implementation blocker in OPEN41's bounded
finish. This is the published UI TEST candidate, not a claim of complete H3
algorithmic parity. The separate approved storage stage starts only after this
verified publication and its full specification is handed over. It has NOT
started here. No cleanup, installation, commit, push or user-data migration.
'''
    evidence = OWN/'evidence'
    for name, value in [('UI-GROUPS-CHECKPOINT-42.json', json.dumps(report, indent=2)+'\n'),
                        ('UI-GROUPS-CHECKPOINT-42.md', text),
                        ('UI-GROUPS-OPEN-42.md', 'UI candidate published for next manual start. Native OS chooser handed to user, not tested. Full Exp/Python and memory-algorithm parity remain open. Next: separately specified approved storage stage after publication; no user data migration or new access implied.\n')]:
        with (evidence/name).open('x') as output:
            output.write(value)
    (PACKAGE/'RELEASE.md').write_text(text)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
