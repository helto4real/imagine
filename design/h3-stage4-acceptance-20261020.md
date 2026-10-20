# Isolated H3 Stage 4 acceptance

The selected standalone native H3 core now completes trained text/vision,
Ref2VA DiT sampling, video VAE and audio VAE execution. No-reference generation,
one own synthetic image reference, and repeated generation after cancellation
and failure match the original exactly. This accepts the explicit technical
profile in `inference/profiles/h3-short-av.json`: 384×256, 39 frames at 24 fps,
four Euler/simple steps, CFG 1, shifts 12/3, seed 42 and the four selected
SHA-locked models. It is not quality or performance acceptance.

The [acceptance index](h3-stage4-evidence/acceptance.json) verifies 25 retained
gates and the [source snapshot](h3-stage4-evidence/final-source-inputs-v3.json)
locks 1,381 implementation inputs and 230 direct original-source inputs.

The local freeze on `h3-stage4-isolated` is based on
`901c847929cc9a68d046910fefdda1dba30e8214`. See the
[version-bound runbook](h3-stage4-runbook-20261020.md) and package lock for the
local commit contents, replay commands and ignored-artifact restoration.
No main/video checkout, product
server, engine, scheduler, UI or ordinary worker was changed or restarted.
No push, merge, other-chat message, Computer Use, network download or private
runtime-data read occurred. All model and media reads were the four explicitly
selected files and this task's own synthetic probe artifacts.

## Exact trained comparisons

| Probe | Native selection prefix | Original selection prefix | Compared artifacts |
| --- | --- | --- | ---: |
| Initial no-reference | `trained-noref-final-01-native` | `trained-noref-cold-01-original` | 18 |
| One synthetic reference | `trained-reference-final-01-native` | `trained-reference-cold-01-original` | 19 |
| Full-source repeat/recovery | `trained-noref-final-repeat-01-native` | `trained-noref-cold-01-original` | 18 |

These are 13 component-phase comparisons and 55 exact artifact comparisons.
The criterion is payload SHA-256, shape, dtype, byte count and actual token
entries. Text conditioning/tags, reference VAE latent, noise, sigmas, each Euler
state/denoised result, final video/audio latents, decoded frames and PCM all
match. No numerical tolerance, alternate weights or dtype fallback was added.
All three final native runs use the complete source rebuild and final worker.
Earlier diagnostic runs remain historical evidence and are not the final source
acceptance.

All selections and comparisons are under `design/h3-stage4-evidence/`.
`selected-model-sha256.json` locks all 51,110,781,487 checkpoint bytes.
`compare_h3_trained_phase.py` verifies the retained artifact payloads before
comparing them. Lifecycle probes are explicitly excluded from inference success.

Two real viewing exports are retained under the ignored own root:

- `comfyui/h3-stage4-20261020/trained/noref-final-repeat-01/native/no-reference.mp4`
- `comfyui/h3-stage4-20261020/trained/reference-final-01/native/synthetic-reference.mp4`

Each MP4 has H.264 video and AAC stereo audio with an actual 1.625-second stream
duration, verified by ffprobe. Original/native source tensors have 39 frames
and exactly 52,000 PCM samples per channel. The viewing-only pixel conversion
and lossy encoding remain outside exact tensor acceptance and product export.
Each clip has its own source/codec/hash manifest. No visual quality review ran.

## CUDA recovery fix and reproducible build

The first trained native text attempts failed at the first `[8192,5120]` ConvRot
projection with `invalid argument`. A bounded regression proved the concrete
cause: after a refused zero-byte own host registration, the original async-error
discard cleared LibTorch's shared CUDA runtime but left the standalone bridge's
separately linked static runtime error slot uncleared. Its next original kernel
reported that stale error.

The initial shared-runtime relink fixed the trained H3 text phase, but the
whole primitive replay found a separate failure: 414 passed, one failed, seven
ignored. The original Kitchen extension statically links its runtime too. Its
Sol empty-chunk failure stays in that private runtime until the original kernel
wrapper checks it. Unifying runtimes changed the exception's class and timing.
That candidate was rejected and its source/binary/provenance retained.

The final fix preserves original static Kitchen linkage. Only host register /
unregister resolve from LibTorch's already loaded shared `libcudart.so.13`,
matching the actual `torch.cuda.cudart()` source boundary. The existing async
error-discard therefore clears the correct runtime, while numerical Kitchen
kernels retain their original error behavior. No kernel, pin policy, expected
error, numerical tolerance or model setting changed.

`pin-discard-cuda-06` passes all 24 bounded dequantization cases before and after
a pin refusal. `sol-chunked-cuda-01` passes the original Sol payload and exact
error expectations, including the previously failing empty-chunk case. The
read-only allocator peak counters are checked by the pin regression too.

The immutable control bridge still has SHA-256
`7fb6c097f1804d163390ff620f2c8ee4fb8f59370f5ec8f4225f50a030be051b`.
The complete source-built H3 bridge has SHA-256
`ab7c807cd5f9aac244f58a302d9500a6ab04b0314b7223404881ce407ed0d4a3`.
Its source/generated hashes are recorded by `build_native.sh` in the separate
own build directory's `provenance.json`. The control build directory was never
made writable. No files in the ignored stable-diffusion.cpp checkout were used
or changed; this is the tracked LibTorch/Kitchen bridge.

`build_native.sh` accepts `IMAGINE_NATIVE_BUILD_ROOT` to keep generated files,
objects and the library outside the immutable dependency root. The pinned
manylinux compiler image and existing locked sources/dependencies are reused
offline. For these H3 probes, explicitly set `IMAGINE_H3_KITCHEN_LIBRARY` and
`IMAGINE_H3_KITCHEN_PROVENANCE`; isolation verifies the chosen library and its
source hashes. Default control selection remains available.

## Validation

- `complete-cpu-v4`: all twelve H3 CPU selectors pass with the final executable
  and complete source-built bridge. Formatting and warnings-denied all-target
  Clippy pass.
- The six `*-cuda-source-build-02` selections pass all 101 synthetic H3 CUDA
  histories, rerunning native against the final bridge and reusing hash-verified, unchanged original oracles.
- The full primitive replay is retained separately at
  `full-primitives-torch-registration-01`. It passes **415 tests, zero failures, seven ignored**, in
  592.33 seconds. The seven explicitly ignored CUDA tests are each selected
  and passed separately; they are not silently counted as full-suite execution.
- The unchanged reservation crate retains seventeen passing lifecycle tests
  and warnings-denied Clippy.
- Current implementation, fixture and original-source fingerprints are frozen
  and verified separately. The final machine-readable acceptance index links
  each selected gate by SHA-256.

An earlier conditioning oracle omitted `**kwargs` and therefore did not execute
vision. That false-positive result is explicitly superseded: the corrected
oracle asserts vision counts `[0,0,1,2,1,3]`; corrected CPU, CUDA and actual
trained image conditioning pass. All failed evidence is retained. An old test
executable initially ran zero tests in the new regression harness; it is marked
failed and the harness now verifies executed test counts. A separate setup-only
provenance check failed before any GPU reservation.

## Measured memory and lifecycle

Actual LibTorch allocator peak-allocated bytes, expressed as GiB below, include
each component's model preparation/forward lifetime. These are allocator peaks,
not sampled maxima of free memory or a controlled performance comparison.

| Phase | Native no-reference | Original no-reference | Native reference | Original reference |
| --- | ---: | ---: | ---: | ---: |
| Text/vision | 11.973 | 12.320 | 12.219 | 12.344 |
| Reference encode | — | — | 2.673 | 2.673 |
| DiT | 11.481 | 11.776 | 11.736 | 11.801 |
| Video decode | 2.841 | 2.800 | 2.841 | 2.800 |
| Audio decode | 0.630 | 0.630 | 0.630 | 0.630 |

The index retains exact byte counts, allocator state after release, and hashes
of the phase records. Successful native unloads report zero loaded model bytes
and zero pinned bytes. Allocator residue is 0–8,781,824 bytes, also present in
the original; it is not represented as zero. Every worker process/PID namespace
then exits before the reservation controller sends finish.

`trained-lifecycle-cancel-after-loaded-02` sends SIGTERM to the owned reservation
controller only after the trained text worker reports real GPU placement.
The controller records `cancelled`, reaps the namespace and then sends finish.
This tests process cancellation and resource release, not graceful in-process
model unloading under a fatal signal.

`trained-lifecycle-missing-input-02` loads the actual trained video VAE and then
fails on its deliberately absent own latent input. The worker performs cleanup
before returning the original error: loaded model/pinned bytes and allocator
allocated/reserved bytes all become zero. A subsequent own reference-image
encode and the complete `noref-final-repeat-01` generation succeed under the same
canonical coordinator. No unrelated Qwen product worker was exercised.

All GPU work used `/home/thhel/ChatGPT/imagine/.imagine/gpu.sock` and the same
Ticket protocol. Native workers expose no Python runtime. Original Python
workers expose locked source/dependencies, their one selected checkpoint and
own probe files. Outer sandbox escalation was required for namespaces and owned
socket/GPU jobs; no automatic approval rejection occurred in this track.

Finish sends have no broker acknowledgement. Controller SIGKILL/fatal-crash
resilience, adversarial sandboxing and broad Stage 3 registry completeness are
not claimed. Product orchestration/export/UI integration remains with the
separate video owner. The selected core is ready for that handoff.
