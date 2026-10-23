# Native Take A/B review

Comparison is transient UI state: selecting A/B, seeking, looping, choosing audio
and closing never author or accept a Scene/Take. Both sources must be distinct
Takes in the selected Scene. Project/Scene replacement invalidates stale loads.
Two bounded CPU previews share one wall clock at the preview's 24 fps. Source
decode remains separate from full-resolution export.

As in Director, common duration is the shorter Take. Both panels use the same
frame index, capped to their common last decoded frame, including rounded
container-duration metadata. Seek/end/restart and editable loop in/out use this
same position. Playback does not start automatically.

The approved Timeline design exposes Muted, Audio A and Audio B. Exactly one
audio child is owned; changing source, seek, loop restart or pause drops/reaps
the previous child before creating another. PCM is stereo f32 at 32 kHz, sliced
at the common seek sample and common end. No gain/normalization is applied.
Privacy concealment, focus loss, workspace leave and comparison close pause
playback and stop audio. Returning requires explicit Play.

The Differences menu exposes selected non-sensitive provenance fields and marks
differences. Prompts, media paths, URLs, secrets and raw setting trees are not
displayed. Metadata stays available when media loading fails; Compare retries
the same selected pair. Cold-open does not restore a playing comparison.

CPU regression covers common duration/range, rounded last frame and exact PCM
source/sample selection. Native fixture QA separately covers real MP4 decoding,
play/seek/loop, audio process transitions, privacy/focus, cold-open/workspace
leave and HTTP media-failure/retry. It checks unchanged authored project content
and revisions during comparison. This proves presented-frame synchronization
and process/source control, not subjective listening or sample-exact device A/V
latency. Full Take removal, all provenance fields and broad size/keyboard/model
acceptance remain part of unfinished P06/V28.
