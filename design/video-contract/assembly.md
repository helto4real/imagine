# Frozen montage and delivery

`POST /v1/video/projects/{id}/assembly` captures selected Takes, imported clip
instances, scene order, output clock/canvas, seam settings and timeline audio.
The response contains identities and provenance, never media paths. Source
containers are copied to the project's selected encrypted staging volume;
raw frames and PCM remain in RAM. Edits after capture do not change that plan.
An older snapshot remains renderable with an explicit UI notice.

`POST /v1/video/assemblies/{id}/jobs` uses an Idempotency-Key and the existing
FIFO. This work kind does not touch inference or register a GPU ticket.
Cancellation joins the CPU pipeline before advancing the queue. A failed or
cancelled job does not publish a Delivery. Retry with a new key uses the same
frozen source copies; retrying an uncertain admission uses its original key.
Completed compressed MP4 bytes are retained by that plan in session memory
until publication succeeds or the session ends. A save retry takes those bytes
through a new destination reservation and the existing publication transaction,
without decoding or rendering again. Successful publication releases the cache.
The plan API exposes whether encoded output is available; it exposes no bytes
or media paths. A server restart does not retain that in-memory result.

Frame counts use Python ties-to-even rounding and endpoint-inclusive fitting.
Scene gaps are removed. Hard Cut, Continuous Shot luminance matching and Blend
Seam are implemented. A generated Transition needs a bridge Take and is
explicitly unavailable rather than silently becoming a cut. Imported clip
trim/speed intent is distinct from inference; changing speed in the inspector
updates authored duration. Attaching a clip preserves the authored scene length
and records its explicit trim and fitted speed from the video's frame clock.
Canvas fitting uses a native RGB8 Lanczos implementation with Pillow-compatible
coefficient quantization and intermediate byte rounding. Fourteen resize cases
match the installed Pillow reference exactly; other image modes are not covered.

The direct MP4 encoder uses libx264, CRF 19, yuv420p (optional 10-bit), BT.709
limited range and AAC stereo. Montage audio follows Director's 44.1 kHz clock;
single-scene H3 remains 32 kHz. Generated, Timeline and Combined routes,
volume/fades, trim/fit, gain, ducking and optional final RMS normalization are
separate explicit choices. The final normalization follows Director: −18 dB
RMS, −1 dB peak ceiling and at most 2× gain. Like Director's sequence mixer,
this path leaves per-clip normalization overrides unapplied and normalizes the
final mix only. The authored overrides remain available to other workflows.

Publication uses the common output transaction and project-owned resource
copies. Delivery provenance/media survive Save/Open and server-session changes;
Accepted Takes are not replaced. An autosave failure leaves the complete
candidate available in the session, as for Takes, and must remain visible.

New controls are limited to format/arithmetic and resident-buffer safety,
resource identity/revision consistency, duplicate admission and actual missing
implementation paths. No model/binary allowlist or input-binary hash gate was
added. Media hashes are the existing project-resource integrity mechanism;
the plan fingerprint describes authoring/source choices. Test binary hashes
are evidence only. The mandatory separate Director review must cover ALL code,
including UI and any older restrictions, after UI implementation is complete.

CHECKPOINT-18 verifies 199 CPU component cases against AST-selected actual
Director functions and six complete raw montage comparisons, including resized
sources, full/cross-clip seams, 10-bit, 30 fps, 30000/1001 fps and half-sample
audio trims. Frame fitting follows Torch FP32 endpoint arithmetic. Audio sample
indices are captured before display-time rounding; optional fields keep older
plans readable. Gain/fade order and tone thresholds follow the reference.
Rendering and raw probes share the same prepared frame/PCM stream.

Five whole RGB comparisons are byte-identical. Continuous Shot differs by at
most 1.192093e-7 in float RGB and one quantized level; PCM maximum error is
2.980233e-8. These measured tolerances do not establish full execution-policy
parity. The native UI import/edit/Undo flow passes 45 actions; explicit missing
transition -> Hard Cut -> successful delivery passes 34 actions. Two earlier
viewport timeouts remain recorded; the passing test scrolls the dialog.

CHECKPOINT-19 adds four passing native UI flows: encoding cancel/retry (3594
frames / 149.75 seconds), save-failure recovery with the retained MP4, retrieval
of an admitted job after a lost response, and the specific transition error
followed by explicit Hard Cut. The save regression makes re-rendering impossible
by removing its own frozen decoder descriptors before retry; publication still
succeeds with the retained digest and no GPU calls. A real transaction failure
rolls back the output file and does not publish a Delivery.

Ten raw Director comparisons now pass, including MKV, actual variable-PTS input
and Peak/LUFS overrides. The prior numeric bounds remain unchanged. This covers
specific procedural inputs, not all VFR/media variants or full H3 execution.
Like the Director reference, source selection/fitting uses decoded frame order
and average frame rate; it does not preserve the source's variable timestamps.
Remaining acceptance includes larger clips/buffer limits, small windows,
project-copy/library delivery handling, generated transition runtime and real
multi-Take H3 montage. See encrypted CHECKPOINT-19 and
[parity deviations](parity-deviations.md).
