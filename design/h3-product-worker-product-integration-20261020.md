# H3 short AV product integration — 2026-10-20

Worktree `/home/thhel/.codex/worktrees/h3-product-worker/imagine`, branch
`h3-product-worker`, frozen core base `40ff91114a6d71782aa1a79a76df6940546570bb`.
All changes are uncommitted. The frozen `be07` checkout remains clean. No later
main/core work, native patch changes or inference policy changes are imported.

## Authorization and implementation

Tomas directly instructed “Gör produktkopplingen jag godkänner”. The subsequent
permission requests succeeded, resolving the earlier tool-review block.
Historical CPU/GPU milestone files retain the earlier blocked status as evidence;
this document and the product milestone supersede it. No extra consent is missing.
No chat was messaged and the other product worktree's staged/dirty work was
preserved. Its source adapter was copied with hashes, then completed here.

The flow is user prompt → existing public job API → common FIFO/Ticket →
verified dedicated H3 child → complete RGB/PCM → lossless Matroska → in-memory
client preview → explicit Save. The one child slot reaps image/H3 workers on
switching. Scheduler cancellation/loss/shutdown ownership remains the existing
implementation; no second reservation, broker, queue, service or GPU child exists.

The only profile is `h3-short-av-noref-v1`: four locked models, no references,
384×256, 39 frames at 24 fps, stereo 32 kHz/52,000 samples, seed 42, four
Euler/simple steps, CFG 1 and shifts 12/3. Image controls, negative text, LoRAs,
model assignments, resolution changes and alternate profiles cannot enter H3.
The private transport contract and native worker bytes are unchanged from the
passed independent GPU milestone.

H3 admission requires a configured coordinator, including direct API/scheduler
calls; it never falls back to the image runtime's optional standalone mode.
CPU tests cover no-deployment and no-coordinator rejection before worker launch.
The server requires an explicitly hash-verified H3 deployment. Old image-only
manifests stay valid and report H3 unavailable. Ready must match the actual
worker and canonical dependency map; pretty JSON whitespace is immaterial.
Dependency verification streams files and excludes model-like paths. Model
identity checks stay inside the admitted worker before trained allocations.

AV output is published with the existing transaction/file rollback mechanism.
Partial/error/cancelled output is never published. Download retains session/no-store
controls and adds bounded streaming with single byte ranges. CPU ffprobe checks
codec, dimensions, fps, sample rate and channels; decoding checks exact frame/
sample counts and finite PCM. Codec input is restricted to pipes, not external
file/network protocols. No user media/prompt is put in argv or diagnostics.

The Video desk has prompt, Generate/Cancel, queue status, playback/frame control
and explicit Save. Preview remains in memory. Save uses the selected output
folder, date folder and independent `video_0001.mkv` numbering. Save failure
retains the result and does not resubmit a GPU job. Video results do not use the
image client's automatic saving. FFplay receives PCM only over stdin; its child
is stopped/reaped on pause, concealment, workspace leave and teardown. A hidden
preview also stops sound. Text concealment now clears accessibility values as
well as pixels, preserving editing/clipboard and existing reveal behavior.

## Explicit deployment and launch

The prepared test deployment is:

`comfyui/h3-product-20261020/manifests/product-runtime.json`

It combines the existing image deployment with the H3 deployment without
modifying an active manifest or starting any service. Its paths bind retained
local build/dependency files; keep those artifacts or prepare a new identity.
A new deployment can be assembled with the built server's CPU-only command:

```sh
imagine-server --prepare-h3-runtime IMAGE_RUNTIME.json H3_DEPLOYMENT.json OUTPUT.json
```

`H3_DEPLOYMENT.json` contains `identity`, `binary`, `model_root`, `environment`,
`files` and `dependency_map`. `files` includes the actual worker and dependency
hashes; the map contains only dependencies and hashes, serialized canonically
for the Ready digest. Neither map contains the four models. The environment
supplies verified Kitchen/AIMDO paths and the native library search path.

To opt in at the next user-controlled launch, pass the explicit manifest:

```sh
IMAGINE_COORDINATOR_SOCKET=/home/thhel/ChatGPT/imagine/.imagine/gpu.sock \
  ./start-server.sh --backend rust --rust-runtime /absolute/path/to/product-runtime.json
```

The ordinary launcher does not automatically select an experimental deployment.
No existing product/proxy/coordinator was restarted or reconfigured in this task.

## Verification

The final CPU workspace run passed 238 tests, zero failed, two intentionally
ignored (the existing live test and this task's explicit trained GPU test).
Warnings-denied all-target Clippy, formatting and workspace build passed.
A final display-only active-model header correction was followed by all 89 UI
unit/headless tests, all-target UI Clippy, formatting and the standalone UI build;
all passed. It changes no request, scheduler, worker or media behavior.
The UI includes headless video controls/privacy, explicit Save/failure retention
and playback interval tests. FFplay's real command/kill/reap path passes with
synthetic silence and dummy SDL devices, with no window or real speakers.
Server tests cover fixed-profile rejection, image/video FIFO, atomic AV, session
headers/ranges, cancellation, image↔H3 switching and failure cleanup.

Two test-environment failures were resolved without changing product guards:
long temporary socket names exceeded Linux AF_UNIX limits; a short symlink then
correctly failed the existing no-symlink storage tests. Final tests use a short,
private physical `/tmp/h3pt-*` directory. The first live-test setup also rejected
an empty profile catalog before job admission; it now supplies its own valid
synthetic profile. No private image-profile defaults are resolved.

The explicit trained product-path GPU test passed separately: one actual H3 job,
39 frames/52,000 stereo samples, explicit Save and Save-failure retry without a
second generation. The final alternate-prompt run after mandatory-coordinator admission finished in
205.52 seconds including setup/identity checks. The earlier own fixture also
passed in 205.43 seconds; its pre-guard evidence is retained separately.
The worker was confirmed reaped and the disposable server shut down. The result
is recorded separately in
`h3-product-worker-evidence/product-milestone.json`. It uses only a fresh test
server/store, the actual UI client, the existing canonical coordinator and the
unchanged locked worker. The test shuts down/reaps its owned worker before
propagating a failure and discards its temporary media on exit.

Native window rendering, real-device audio/video synchronization and perceptual
quality remain manually unverified. No Computer Use was authorized or used.
Intermediate tensor equality/peak VRAM and a trained next-image recovery test
remain outside this product wiring evidence. The earlier independent GPU cases
remain valid for their unchanged worker inputs; they are not rerun or conflated
with the product-path test. No private runtime history/logs/media were read,
no models downloaded, and no commit, push, main merge or product restart occurred.
