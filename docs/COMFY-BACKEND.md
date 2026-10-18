# Optional headless Comfy backend

Imagine defaults to `rust` (see [Rust backend](RUST-BACKEND.md)). To select the alternative Qwen engine at server
startup:

```sh
./start-server.sh --backend comfy
```

Imagine looks for Comfy source in `<project>/comfyui`, then `$HOME/git/ComfyUI`,
and uses `.venv/bin/python` inside the selected checkout. On Tomas's machine
this discovers `/home/thhel/git/ComfyUI` and its existing environment. Nothing
is installed or downloaded. Missing paths produce an actionable error.
Explicit flags override these defaults:

```sh
./start-server.sh --backend comfy \
  --comfy-root /absolute/path/to/ComfyUI \
  --comfy-python /absolute/path/to/ComfyUI/.venv/bin/python
```

The existing Imagine API, desktop UI and proxy GPU coordinator remain in use.
Do not start the ComfyUI server. Startup validates source/runtime paths but does
not launch Python or initialize CUDA. The first scheduled generation launches
the worker after acquiring the GPU reservation. Returning to native requires
restarting Imagine with `--backend native`. Returning to Rust uses
`--backend rust` or no backend flag. There
is no automatic engine fallback.

## Scope and defaults

- Qwen Image 2.1 INT8 ConvRot and official Turbo BF16 use Comfy's own loaders,
  text/image conditioning, sampler, VAE and memory management.
- Diffusion attention is explicitly `comfy_kitchen_int8`. Missing CUDA/Kitchen
  support fails the job rather than silently selecting another attention method.
  Kitchen's own precision/mask handling remains Comfy's implementation.
- Qwen prefix-cache device is `auto`, dtype `default`; Torch compilation is off.
- Ordered references use Imagine's shared Rust shrink/crop preprocessing. Python
  preserves that geometry and alpha; the Qwen encoder performs no second resize.
  Output dimensions always come from the request, independently of reference size.
- LoRA order and strengths come from the request. Each pass starts from pristine
  model/CLIP patches, including when a previous request used LoRAs. Only the base
  components remain resident between successful requests. No new prompt/encoding
  cache is added by this adapter.
  LoRA request paths are relative to the library's `loras/` directory, following
  the native contract. The Comfy adapter resolves them under `<library>/loras/`,
  including when the selected library differs from the default model root.
- Existing Rust color/brightness/contrast matching and character-sheet orchestration
  are reused. H3/video remains unavailable.

Turbo's shipped Quick draft profile (version 3) uses Euler, CFG 1 and these eight manual
steps (nine sigmas), passed directly to Comfy without an extra sigma shift:

```text
1, 0.978453, 0.954180, 0.926626, 0.895080, 0.845148, 0.704534, 0.414568, 0
```

Sampler and scheduler use the selected profile's defaults unless explicitly
overridden. The GUI's Settings dialog offers the supported library choices:
`er_sde`, `euler`, `res_multistep`, and schedulers `beta`, `simple`. Switching
models restores that model's sampling defaults, as it does for steps and CFG.
For Turbo, the scheduler control reads `Manual sigmas` and is disabled; the
sampler remains selectable and the fixed sigma list is preserved.

## Runtime requirements

Use a Comfy source/environment version providing `CLIPType.QWEN_IMAGE`,
`TextEncodeQwenImage21`, `QwenImage21Cache`, metadata-aware VAE/LoRA loading,
`set_model_optimized_attention`, the `comfy_kitchen_int8` attention registration
and `comfy_kitchen.int8_attention_is_available(device)`. Its Python environment
must already contain the matching CUDA PyTorch, Comfy dependencies, NumPy,
Pillow and a Kitchen build supporting the GPU. Imagine does not install packages,
download weights or modify that environment. Core APIs can change between Comfy
versions; incompatibility produces a sanitized startup/generation error.

Choose model files through Imagine's existing model library. The Comfy backend
does not need the native executable, does not initialize custom node packs,
and does not read Comfy workflows, saved jobs or output folders. The checked-in
worker imports the Qwen core node definitions directly; no workflow executor,
`main.py`, HTTP listener or `PromptServer` instance is started.

## Lifecycle and privacy

The Rust process supervises one persistent child. Versioned length-bounded
frames carry JSON controls and binary reference/PNG bytes over private pipes.
Prompts and pixels are never command-line arguments or worker temporary files.
The worker disables core dumps through the server's inherited limit, suppresses
third-party stdout/stderr and emits only allowlisted numeric/status events and
static error codes. Hugging Face/Transformers offline flags are set before imports.

Cancel, unload, lost GPU reservation, failure and shutdown use the existing Rust
scheduler. It kills and waits for the child before releasing the reservation.
The child is also killed if its parent dies. The next job starts fresh after a
failure or unload. Model file stamps are verified before inference; replacing
loaded component files forces a reload.

## Validation boundary

Synthetic checks only:

```sh
python3 -B -m unittest discover -s workers/comfy -v
cargo fmt --all --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace
```

These use stdlib/fake tensors and a fake Python child: framing limits, reference
order/geometry/alpha, manual Turbo sigmas, explicit Kitchen selection, component
reuse/replacement, LoRA removal, progress, persistent PID reuse, cancellation,
confirmed process reaping, malformed output and recovery. They do not import
the real Comfy runtime or load models. Model APIs are source-reviewed but still
need a later authorized GPU acceptance run. Performance, VRAM fit, image quality
and pixel parity with native are unmeasured for this adapter. The frozen native
1600×896 baseline remains the comparison control; its evidence is unchanged.

`/v1/runtime` identifies the selected backend. Model/catalog responses identify
Comfy/Kitchen and do not advertise native's validated resolutions as Comfy evidence.

Implementation validation (2026-10-16): Python's ten synthetic tests, the Rust
workspace tests, formatting, Clippy with warnings denied and the release server
build passed. The existing opt-in native GPU test remained ignored. Workspace
integration tests needed permission for temporary loopback listeners/sockets;
no real inference service was contacted and no model run was performed.

Subsequent isolated diagnostic (2026-10-16): one authorized INT8 generation with
a fixed synthetic prompt, no references/LoRAs, 1600×896, 40 ER-SDE/beta steps,
CFG 2.5 and seed 42 completed through PNG encoding under a GPU-coordinator ticket.
Image bytes were discarded without inspection or saving; this is not UI, image
quality or parity acceptance. The direct adapter call included an empty sigma
list, which concealed an IPC mismatch: Rust omits empty `custom_sigmas`, while
Python indexed it unconditionally. A model-free test through the actual Rust
request and Python adapter reproduced `comfy_generation_failed`; accepting the
omitted optional field fixes it. Explicit Turbo sigmas still pass unchanged.

LoRA diagnostic and fix (2026-10-16): the supplied
`qwen21_fullbody_pass2_fp16.safetensors` at strength 1 completed in a direct
fresh-worker check and in a persistent no-LoRA → LoRA → no-LoRA sequence
(model patch counts 0 → 192 → 0). The Rust bridge nevertheless resolved the
library's relative LoRA name against the model root instead of its `loras/`
directory. That path translation is fixed and covered by a synthetic regression
using a selected library root. A subsequent authorized generation through the
actual Rust backend, IPC and Comfy worker completed with the same LoRA and
1600×896 / 40 ER-SDE-beta / CFG 2.5 / seed 42 synthetic fixture. The process was
stopped before releasing its GPU ticket. Images were not saved or reviewed;
patch-count restoration is not an image-parity claim. Restart Imagine to load
this Rust-side fix. The live test is opt-in (`live_lora_through_rust_backend`)
and requires explicit test paths plus the coordinator socket.

Turbo conditioning fix (2026-10-16): Comfy conditioning may contain
`pooled_output=None`. The Turbo profile enables zero negative conditioning;
the adapter previously passed that None to `torch.zeros_like`, causing a
TypeError. It now follows Comfy's non-None guard, preserving absent/None pooled
outputs and other conditioning metadata. A synthetic regression reproduces the
old crash, and the shipped Turbo profile passes through Rust IPC and the actual
Python adapter with tensor doubles. The manual sigmas, sampler and scheduler
are unchanged. No Turbo model/GPU run was performed for this fix.

## Model-switch quality investigation

The synthetic GUI test `model_picker_roundtrip_keeps_cold_and_switched_requests_identical`
uses the actual picker twice in both directions. The complete Turbo request after
INT8 matches a direct Turbo selection; returning to INT8 restores its request.

An explicitly authorized three-generation check used the real Python adapter,
Kitchen attention, no LoRAs/references, a fixed synthetic prompt and seed 42 at
1600×896. One fresh process ran Turbo; another ran INT8 (40 steps, CFG 2.5,
ER-SDE/beta) followed by Turbo (8 steps, CFG 1, res_multistep, shipped manual
sigmas). All **358,400 pre-VAE latent values matched exactly**, with maximum
and mean absolute difference **0**. All values were finite. The comparison
stayed in CPU RAM; decoded images were discarded without inspection or saving.
This does not establish final-pixel equality or rule out other prompts, seeds,
dimensions or a VAE-stage difference. No production inference change was made.
