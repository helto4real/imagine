# Early Rust backend integration in Imagine

Tomas explicitly requested a separately selectable Rust inference backend and
made it the server default before continuing general parity implementation.
He additionally approved making MOSS selectable. This advances the selected
product integration work without declaring the later full parity gates complete.
No numerical tolerance, alternate model algorithm or autotuning change was
introduced. No ordinary Imagine/Dreamchat service was restarted or inspected.
All live validation used new isolated projects and synthetic inputs.

## Implementation decisions

- `--backend rust` is the default; `native` and `comfy` remain explicit choices.
  Missing runtime artifacts and unsupported capabilities fail without fallback.
- Keep the inference crate/build separate from the server/UI workspace.
  Its dedicated Rust child preserves fault/cancellation isolation and avoids
  linking CUDA/LibTorch into the public HTTP server or UI.
- Reuse the existing private worker's bounded protocol, parent-death handling,
  progress validation, colour matching and stop/reap lifecycle. The Rust branch
  transfers raw RGBA references and RGB/RGBA results; image codecs stay in the
  existing server. This avoids a new inference codec dependency or temporary
  image/prompt files.
- New `inference::server` orchestration consumes the existing validated model,
  CFG, sampler, memory, VAE and LoRA components. The immutable registered
  development drivers keep their existing profiles/export behavior. Product
  requests are explicit validated settings, not edits to a fixed test profile.
- Use the source static patcher, disabled dynamic compiler/asynchronous streams,
  automatic CPU/GPU management, two CPU tensor threads, BF16 model arithmetic,
  PyTorch fallback attention and checkpoint-specific attention preferences.
  Kitchen Triton stays disabled as in the selected Comfy startup policy.
  Reference preprocessing retains Imagine's shared shrink/crop and resolution=0
  Qwen encoding. Output canvas is independent of reference geometry.
- Initial product coverage retains the existing Qwen INT8/official Turbo IDs
  and adds MOSS SFT INT8. The general BF16 development model is not a new public
  model ID in this change. MOSS is text-to-image only; unsupported reference,
  LoRA, layer-skip/decode settings fail. Components have separate model groups.
- No base-weight reuse across jobs is claimed. The worker keeps its native
  runtime/kernel caches; managed models unload between phases. This is the
  already validated static lifecycle, not an unvalidated memory optimization.
- `prepare-rust-backend.sh` builds offline against already prepared local
  artifacts and writes a hash-checked deployment manifest. Neither preparation
  nor worker execution starts Python. Native dependencies are not downloaded or
  silently rebuilt by the launcher. Existing old baseline artifacts are retained.

## Validation evidence

The [evidence directory](rust-backend-server-evidence/README.md) retains test
results, typed source-oracle profiles, trained comparisons and input hashes.
The accepted trained controls cover the real HTTP/queue/worker path with
320×224 images and 256×256 Qwen text-to-image controls. Every compared output pixel is exact against the locked
original source. They do not establish arbitrary workload, model or pressure
coverage. In particular, the earlier [upgrade report](comfy-reference-upgrade-20261018.md)
retains the different automatic Qwen BF16/Turbo LoRA placement observations;
this change does not turn those failed automatic pairs into accepted parity.

The test controls are MOSS 4-step Euler/normal CFG 3.5, Qwen INT8 4-step
ER-SDE/beta CFG 2.5 with LoRA/reference, and Turbo 8-step Euler/manual sigmas
CFG 1 with reference. Additional Qwen text-to-image controls cover both a
nonempty and an empty negative prompt at four ER-SDE/beta steps and CFG 2.5. MOSS product defaults remain 50 steps/CFG 3. All use the
same seed and source profile operands as their individual original oracle.
These are functional comparisons, not performance or image-quality benchmarks.
Source-oracle pipeline times and HTTP job times have different boundaries.

Synthetic tests cover default selection and metadata, unsupported sampler
admission, family/component validation, raw-pixel framing/progress, process
reuse and cancellation/reaping, changed deployment hashes, MOSS model/group
switching and retained Qwen LoRA settings. The existing coordinator, API/UI,
proxy and session checks are retained. Initial sandbox socket denial was
resolved with test-only escalation. No automatic approval rejection occurred.

One repeated workspace run hit the unchanged clipboard timing test
`stale_failure_does_not_replace_a_newer_copy` under concurrent test/build load.
It was not weakened or edited. Separate rerun and reduced test concurrency are
recorded with the final results. Browser/native-window manual acceptance was
not performed; UI evidence is the headless Rust test suite.

The integration request itself did not include a commit or push. Unrelated dirty
native/UI work was retained; no ordinary server was restarted by the agent.

## User acceptance and baseline preservation, 2026-10-18

Tomas subsequently confirmed that the Rust backend works in Imagine and
explicitly requested a baseline commit before completing the remaining Stage 2
follow-up. This is user-reported manual product acceptance, separate from the
isolated automated evidence below. No ordinary runtime data was inspected to
obtain this confirmation. The baseline preserves the approved reference upgrade,
Rust integration, selected MOSS support, reference controls and regression tools.
Unrelated native optimization and UI layout changes remain outside this commit.
Historical evidence hashes and failed comparisons are retained unchanged.

The exact staged baseline was also exported into an isolated source checkout
and its complete offline workspace suite passed, with two test threads. This
separately checks that excluding unrelated native/layout work does not leave
the committed tree dependent on those unstaged changes. The first attempt was
blocked by the sandbox's socket restriction; the socket-enabled rerun passed.
Retained verbatim evidence logs have three trailing-blank-line whitespace
notices; owned source/documentation files pass the staged whitespace check.

The selected old-reference Stage 2 checkpoint remains accepted. Follow-up for
the upgraded/product path is ordinary BF16 model selection and the unresolved
automatic BF16/Turbo LoRA placement comparisons. Full sampler/scheduler/guider
coverage remains Stage 3 under master-design Section 15; it has not been removed
from the required full-parity scope. No push is authorized.

The final product review also checked the source node's empty-text behavior:
`prevent_empty_text=True` is unconditional for Qwen positive/negative encoding,
including when no images are supplied. The adapter preserves its blank-space
sentinel and `keep_vision=(len(reference_latents)==0)`. The dedicated empty-
negative text-to-image control shields the actual default workload. Dimension
validation retains the existing API's minimum of 32 for explicit components.

Final checks: 150 workspace tests pass (two explicitly ignored live tests),
three final private-pipe tests pass, two binary privacy tests pass, formatting
and workspace/inference Clippy pass. All five trained HTTP comparisons have
exact decoded pixels: MOSS, Qwen INT8 reference/LoRA, Turbo reference, Qwen
text-to-image with a nonempty negative, and the empty-negative default case.
The last of these records the final worker binary SHA256. Intermediate worker
versions are identified separately; no historical numerical result is relabelled.
