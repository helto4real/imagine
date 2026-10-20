# Isolated H3 Stage 4 progress

The selected native core now passes trained no-reference generation, one own
synthetic-reference generation and repeated generation after failure/cancellation.
The [acceptance report](h3-stage4-acceptance-20261020.md) describes the final
scope, measurements and limitations. The final whole primitive replay passes 415 tests, with zero failures and
seven separately executed CUDA tests explicitly ignored in that broad replay.
The machine-readable acceptance index verifies all twenty-five selected gates.
The local freeze on `h3-stage4-isolated` is based on
`901c847929cc9a68d046910fefdda1dba30e8214`.

## Core implementation

The dedicated `inference/src/models/minimax_h3/` modules implement original
packed layout/masks, checkpoint preparation/physical owners, 50-layer
unnormalized Qwen3-VL text/vision conditioning, complete dense DiT, causal video
VAE, FP32 audio VAE and binding to the common CFG1/Euler AV sampler. Actual
references, keyframes, masks, DeepStack, modality tags and audio scale boundaries
are retained. The existing Qwen constructor keeps its norm-required behavior.
The standalone development worker is `inference/src/bin/h3-stage4.rs`.

The four exact selected model files have full SHA-256 locks, totaling
51,110,781,487 bytes. Trained forward execution and SHA verification now read
actual weights; earlier header-only statements are historical. No AWQ weights,
model substitution, dtype fallback, numerical tolerance or new dependency was
introduced. Native inference workers expose no Python runtime.

All build/cache/model-probe outputs are under the ignored own root
`comfyui/h3-stage4-20261020/`. Existing pinned source/dependencies remain controls.
No main/video checkout, product server/engine/scheduler/UI, ordinary worker,
other chat or private runtime data was changed or inspected. No Computer Use,
push, merge, network download or product restart occurred.

## Final selected gates

- `complete-cpu-v4`: twelve exact CPU selectors against the final bridge; format
  and warnings-denied all-target Clippy pass.
- Six `*-cuda-source-build-02` selections: 101 synthetic H3 CUDA histories,
  including full text/vision/DeepStack, actual-format INT8, complete learned
  tiny AV sampling and large causal audio SDPA/state restoration.
- `pin-discard-cuda-06` and `sol-chunked-cuda-01`: both the reproduced pin-refusal
  regression and the original Sol payload/error expectations pass.
- Final trained runs `noref-final-01`, `reference-final-01` and
  `noref-final-repeat-01`: thirteen phase comparisons and 55 exact artifact
  comparisons against the original's selected, identical files.
- The two `trained-lifecycle-*-02` selections: real cancellation after trained
  GPU placement, then failure cleanup with zero model/pinned/allocator bytes,
  followed by successful reference-image encode and complete generation.
- Both own MP4 exports have 39 frames at 24 fps and stereo 32 kHz audio lasting
  1.625 seconds. Source PCM has exactly 52,000 samples per channel. They are
  technical viewing artifacts; no visual quality review ran.
- The unchanged reservation crate retains its seventeen passing CPU lifecycle
  cases and warnings-denied Clippy. Actual GPU probes additionally confirm
  namespace exit and child reaping before each finish send.

The original common baseline's twenty selected CPU tests passed before H3 edits.
The broader retained primitive suite and broader Stage 3 registry completeness
are separate claims; the latter remains open.

## Retained failures and decisive corrections

The first complete conditioning oracle accidentally omitted `**kwargs`, so its
six apparent image/video cases were plain text. It was corrected to forward
those arguments and assert actual vision counts `[0,0,1,2,1,3]`. Corrected CPU,
CUDA and trained image conditioning pass. The initial false-positive evidence
is not vision acceptance.

A one-ULP text RoPE difference came from enabling Kitchen Triton in Rust when
the source selected profile disables it by default. Only the H3 profile/test
policy was aligned; native engine defaults and numerical kernels were unchanged.
The initial video CUDA adapter called the generic registry for INT8 attention;
connecting the existing `Runtime::int8_attention` surface fixed it. Reference
layout timing also required CPython 3.12's compensated sum. Constructor load-kind,
legacy metadata precedence and source-specific buffer ownership were verified
rather than replaced with a common approximate recipe.

The first trained native text attempts failed at the first `[8192,5120]`
ConvRot projection after refused host pins. A tiny own zero-byte registration
reproduced it. Host registration incorrectly used Kitchen's private static CUDA
runtime, whereas Comfy's async-error discard executes through LibTorch.

An initial candidate unified the runtime by shared linking. H3's trained
comparisons passed, but the whole replay reported 414 passed, one failed, seven
ignored: Sol's empty-chunk failure changed exception class/timing. Read-only ELF
inspection confirmed original Kitchen itself statically links its runtime.
That candidate was rejected; its sources, library and provenance are preserved
under the own `shared-runtime-rejected` directory. Failed test expectations were
not weakened or changed.

The final narrower fix preserves static Kitchen linkage and resolves only host
register/unregister from LibTorch's already loaded shared `libcudart.so.13`,
matching `torch.cuda.cudart()`. Both isolated regressions and the final trained
comparisons pass. A new full-source build and final whole replay use this fix.
The frozen control bridge and ignored stable-diffusion.cpp checkout are unchanged.

An early regression probe selected an old executable and ran zero tests; it is
marked failed. The harness now checks actual test counts and full-suite totals.
A separate provenance setup check incorrectly matched the package directory's
name before GPU start; no reservation or GPU worker ran in that failed setup.

## Handoff boundary

Source/generated hashes, model hashes, exact comparisons, true allocator peaks,
clip time bases and reservation reports are retained separately. No sampled
free-memory value is called a true peak. Successful model unloading leaves
0–8,781,824 allocator bytes before process exit, matching the original's residue;
it is not represented as universally zero.

Sandbox escalation was required for reviewed namespaces and owned sockets/GPU
jobs. No automatic approval rejection occurred in this track. Finish sends have
no broker acknowledgement. Controller SIGKILL resilience, adversarial sandboxing,
quality/performance acceptance and a separate Qwen product worker's recovery
are not claimed. Product orchestration/export/UI remain with the video owner.

The [runbook](h3-stage4-runbook-20261020.md) records the version-bound start and
restore paths. Freeze packaging uses only read-only/CPU checks and starts no
new GPU jobs or heavy acceptance runs.
