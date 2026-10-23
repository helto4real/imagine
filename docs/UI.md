# Imagine desktop UI

A small native Rust client for API v1, implemented with
[eframe/egui](https://docs.rs/eframe/0.36.2/eframe/) and
[rfd](https://docs.rs/rfd/0.17.2/rfd/) for native file/folder dialogs.
It uses HTTP exclusively; it does not load models or call the engine directly.
All application-owned labels, statuses, dialogs and errors are in English.

## Start

Run Imagine, the proxy stack, and the UI in separate terminals:

```sh
./start-server.sh
./start-proxy.sh
./start-ui.sh
```

`start-proxy.sh` starts all supported LLM backends by default, waits for readiness,
and exposes their combined model catalog for clients such as Pi. Ctrl-C stops
the stack; `--log` shows backend logs. Use `ST_PROXY_LLM_BACKEND` to select a
single backend. CLI options also configure the stack; use `--proxy-only` for
independently managed backends. ComfyUI is disabled by default and need not run.
Enable it explicitly with `--comfy-enabled true` when using Comfy routes.

`start-ui.sh` incrementally builds the release client. The API and desktop client
have separate lifetimes; closing the window does not stop the server or cancel
an accepted job. Use **Cancel** while a generation is active to request cancellation.

Default proxy API: `http://127.0.0.1:8189/imagine`. No token or credentials file
is needed. If the server is offline, the window remains usable for writing a prompt and selecting an
output folder; choose **Reconnect** after starting Imagine and the proxy.

```sh
./start-ui.sh --api http://127.0.0.1:8189/imagine
```

The proxy forwards `/imagine/*` on its image port to `--imagine-url` (default
`http://127.0.0.1:8080`), stripping the `/imagine` prefix. The UI uses that base
for metadata, uploads, job submissions/polling/cancellation and PNG downloads.
The proxy preserves session and idempotency headers and strips Authorization.
Imagine uses one local workspace for all clients, including runtime controls.
The HTTP forwarding does not reserve a second GPU slot: Imagine reserves its job ticket in the shared coordinator FIFO.
There is no automatic direct fallback when the proxy is unavailable.

The root launchers default to `.imagine/gpu.sock` on both sides. For custom
paths, set `IMAGINE_COORDINATOR_SOCKET` for the server and matching
`ST_PROXY_COORDINATOR_SOCKET` for the proxy (CLI options override environment
defaults). Use `--api` for a different API address. To run a standalone server
without coordination, invoke the server binary without `IMAGINE_COORDINATOR_SOCKET`.

Imagine binds only to loopback. When Imagine coordination is enabled, the proxy
also requires a loopback image listener. The UI needs only the API URL;
`--access` and `--client` have been removed. HTTP redirects remain refused.

Linux prerequisites include a working Wayland or X11 session, an OpenGL-capable
display driver and an XDG desktop portal file chooser. The UI uses the host's
system sans font when `fc-match` is available, with bundled font fallback.

In a Wayland session, install `wl-clipboard` to keep copied text available after
Imagine exits. The app hands Copy/Cut text to `wl-copy`, whose background owner
outlives the window. Text is passed through an anonymous memory-backed descriptor,
not command-line arguments or application temporary files. Delivery runs on a
worker with a timeout; shutdown briefly drains the latest pending copy. Paste
continues through the native clipboard. If the helper fails, the app falls back
to native copying and reports that persistence failed. Without `wl-copy` or
outside Wayland, the native clipboard remains in use and persistence depends
on the desktop's clipboard manager. See the [wl-copy manual](https://man.archlinux.org/man/wl-copy.1.en).

Glow VSync is disabled in both the application and the synthetic native preview.
This avoids requesting a presentation wait on the UI thread when the window is
on a hidden workspace. NVIDIA documents a potential indefinite `eglSwapBuffers`
wait for invisible windows under some Wayland compositor configurations
([egl-wayland2 compositor requirements](https://github.com/NVIDIA/egl-wayland2#compositor-requirements)).
This is a targeted mitigation; the reported Imagine hang has not been confirmed
with a native stack or reproduced in a live generation test.

## Workflow

Configure prompt enhancement in **Settings → Prompt enhancement**. Create or
duplicate a profile, edit its **System prompt**, choose the model ID, and save
with **Save profile**. Assign a saved profile to each model family, or select
**Off**. The default Chat Completions base URL is `http://127.0.0.1:5002/v1`,
matching `start-proxy.sh`. **Load models** queries that server's `/models` endpoint
using the entered key (or the saved key for the same server), without saving the
draft. Choose a returned model or enter a model ID manually. Model lookup never
forwards a saved key to a different server address.

Optional controls set temperature, Top P/K, Min P, Typical P, repetition,
frequency and presence penalties, and seed; unchecked controls use server defaults.
**Advanced → Extra parameters (JSON)** accepts additional provider fields, such as
`{"stop": ["<end>"], "repeat_last_n": 64}`. Dedicated fields cannot be overridden
in JSON. Advanced settings also select `max_tokens` or `max_completion_tokens`.
Unsupported parameters produce a provider error and preserve the prompt.
API keys are optional,
stored privately on the Imagine server, and never returned in profile responses.

Use **Enhance prompt** beside a text field to improve it without generating an
image. The replacement is one undo step. If you edit the draft while waiting,
the result offers **Use** or **Discard** instead of overwriting your changes.
Cancelling or a failed/truncated response preserves the draft. Text generation
through an inference text encoder is not part of this implementation.

Profiles are saved in `enhancement-profiles.json` in the server data directory;
family selections remain in `.imagine/ui-settings.json`. Enhancement inputs and
outputs are session-only and are not written to the profile file or logs.

1. Enter a **Prompt**. **Negative prompt** is optional and restored from your last edit.
2. Optionally choose the **+** icon to upload PNG/JPEG/WebP reference images.
   Thumbnails are numbered; the up/down controls change reference order.
3. Open **Settings** (gear) and choose the folder icon under **Save to**.
   The folder choice is saved immediately.
4. Choose **Add to queue** (sparkles). The result is shown after its PNG has been saved locally.

### Image queue and results

The collapsible **Queue & results** dock keeps the main viewer above a job list
and result thumbnails. **Add to queue** and Ctrl+Enter capture the current image
prompt, references, model/LoRA selection, sampling settings and output folder.
The composer stays usable while these jobs run. **Multiple prompts** opens a
list with one multiline field per prompt; **Add all to queue** captures the
shared settings and admits the entire series through `/v1/jobs/batch`. Blank
rows must be filled or removed; paragraphs are never split into separate jobs.
The server's shared FIFO owns execution. Pause and reordering are not included.

Each row shows its status and sampling progress. The main image footer mirrors
the running image job's progress, phase and ETA, even while an older result is
selected. Cancel in either location targets the corresponding job; the row
remains active until the server reports a terminal state. Errors remain
available for Retry. Uncertain admission reuses the original batch key; retrieval
and save retries reuse the accepted job and use the currently selected output
folder. A confirmed failed generation is retried
as a new job with its original captured settings. A failed row does not block
other jobs in the series.

Click a thumbnail to load its saved image in the main viewer. This leaves the
next prompt and controls unchanged and turns off **Follow latest**. Turning that
checkbox on selects the latest result and follows subsequent results while the
image workspace is idle. Editing and upscaling use the selected full image.
**Reuse settings** explicitly restores the result's image prompt, references and
generation/model settings while retaining the current output folder. Edited and
upscaled images also appear in the result strip.

The dock uses one privacy boundary: with **Hide previews** enabled, hovering
anywhere inside the focused dock reveals all its thumbnails and prompt excerpts.
Leaving the entire panel or losing focus hides them together. Thumbnail hover
does not independently reveal images. Main images and reference fields retain
their existing privacy behavior.

The UI job list, result index and captured prompts exist only for this UI session;
closing the client does not cancel server-accepted jobs. No history is restored
on reopening. Saved PNGs remain in the chosen output folder. The strip retains
small textures and loads a selected full image from its saved file on a worker.

The Reference desk layout places ordered references on the left, the image in
the center, and the prompt/size/seed inspector on the right. Every toolbar action
uses an outline icon with a tooltip and accessible name. The image, character-card
and film icons switch between Image, Character sheets and Video. Switching keeps
each workspace's drafts, references, results and pending job intact. Video opens
the Timeline. **Animate** in the image viewer creates a one-scene Timeline project
with the selected image as its first frame; there is no separate single-clip desk.
Disabled actions name themselves and give the reason on hover.

The gear opens a separate native **Settings** window with **Generation** (Steps,
CFG, Reference size, Upscale target), **Storage** (output folder), and **Privacy** (Hide previews). Changes save
automatically. The close icon or Escape closes only Settings. A failed preference
write shows **Not saved** and the error; it is never reported as saved. On backends
without native child windows the same settings content uses egui's embedded window.

No references means `text_to_image`; one or more means `image_edit`. The client
starts with 1600×896, 40 steps and CFG 2.5. **Steps**, **CFG**, **Seed**, **Ratio**
and **Max side (px)** are editable and saved automatically. The seed selector
offers **Random each time** (default) and **Fixed seed** (0–4,294,967,295). Random
mode sends a null seed so each new job resolves its own seed; retries reuse the
accepted job and seed. The result displays the resolved seed.

Ratios match AIO Generate: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16 and 16:9.
Max side accepts 256–4096 pixels in multiples of 32. The short edge uses AIO's
nearest-multiple-of-32 rounding, including ties-to-even. The resulting width and
height are shown below the controls. Source: AIO `services/dimensions.py`, preset
ratio mode with `multiple value=32`. For example, 16:9 at 1600 gives 1600×896.

**Reference size** sets the maximum megapixels per reference used during inference,
from 0.262144 to 4 MP, with the existing 1 MP default. It saves automatically and
can be edited during generation for the next job. The server only shrinks larger
references with Lanczos, then center-crops to multiples of 32, following AIO's
geometry. Small references are never enlarged; uploaded originals are unchanged.
MP here means 1,000,000 pixels (AIO uses 1,048,576). The cap applies independently
to each reference, not their combined area.

ER-SDE/beta and one PNG remain model defaults. Native code and the frozen
configuration are unchanged. Other output
sizes are accepted by the API but have no new live GPU or image-quality validation;
the frozen measurement scope is still 1600×896. Large images may exceed memory.
Restart both the API and UI after updating to enable the new dimensions contract.

Reference count is limited to ten. The existing server upload limits apply.
The prompt field has a bottom-right drag handle for changing its height; longer
text scrolls inside the field. Its height is kept for the current UI session.
The eye icon and **Hide previews** switch toggle visual privacy for the result,
reference images, prompt and negative prompt. It is on by default and the choice
is saved immediately. Each image or text field reveals only while the pointer is
over it and the window has focus. Leaving the area or losing window focus hides
it immediately, including selected text and the caret. Editing focus alone
(Ctrl+P, Tab or a click) does not reveal text. Once you type, paste or delete in
the focused field, that field alone stays revealed, also inside windows such as
Story or Cast, until it loses focus or the window loses focus. This is visual
concealment only: prompt content, clipboard operations and saved PNGs are
unchanged.
Uploading does not resize the source file: server preprocessing controls the
VAE and vision inputs, with further native resizing disabled. The client creates
small thumbnails only for display. The
file dialog's order is used initially and can be adjusted before submission.
Only one reference has local model-quality validation. H3 is not exposed here.

## Editing generated images

Choose **Edit image** (pencil) on the Image result to open an independent editing
draft. The source stays in the viewer; the right panel contains **References**,
**Prompt**, and **Create version**. The original generation prompt and references
are preserved separately and are not copied into the editing draft.

Use **Add references** to choose PNG/JPEG/WebP files. References have no content
categories or assigned semantic roles. Their numbers are stable within the edit
session: removing Reference 2 does not rename Reference 3, and a newly added image
gets a new number. Refer to them in the prompt as `Reference 1` or `referens 1`
(case insensitive). Explicit numbered references that are no longer present are
rejected before submission. The source occupies one of the model's ten image
slots, leaving up to nine optional extra references.

The client sends the source asset first and the extra assets in display order.
A short Qwen binding prefix identifies the source as `<image1>` and maps the
stable reference numbers to the remaining image positions. The user's prompt
is appended verbatim; references have only the meanings given by that prompt.
The output keeps the selected source dimensions and uses the current sampling,
seed and negative-prompt settings. Model configuration and reference preprocessing
are unchanged. Number binding does not guarantee model fidelity or unchanged pixels.

Each successful edit saves a new file using the normal dated output folder and
numbering. The source file is never overwritten. The selector below the viewer
switches between **Original** and **Version N**; the selected version becomes the
source for the next edit. **Finish editing** returns the selected result to the
normal Image view without replacing its generation draft. Reopening editing
restores the edit draft and versions. A new successful ordinary image generation
or upscale starts a new editing lineage; saved files remain untouched.

Hover over an edited version to compare it with the image that version was
created from. The source appears left of the pointer and the edited result right
of it, separated by a vertical line. Both sides share zoom and pan. Leaving the
image or unfocusing the window restores the edited result; Original has no
comparison. Visual privacy applies to both sides.

Uploads and job results return to the editing draft even after switching tabs.
Requests capture their source, prompt, reference bindings and settings at submission.
Retry reuses the same request, idempotency key and accepted job; changing a draft
does not alter that request. References, version selection and leaving the editing
mode are locked while uploading or while a job/retry is pending. Errors and
cancellation preserve the draft and earlier versions. If the server session has
lost an asset, select a current result or upload the missing reference again.

Editing drafts and version links are session-only. Existing visual privacy also
covers the edit source, all extra references and the prompt. Character-sheet
editing, masks, brushes and persistent version history are outside this feature.

## Character sheets

The **Character sheets** tab reuses the image desk's three-panel layout and controls.
**Character** and **Clothing & accessories** replace Prompt. Both fields resize,
scroll and follow the existing hover-to-reveal privacy setting. Add up to eight
identity references, or describe the character without references. Clothing is
text-only; this tab does not assign separate clothing-reference roles.

**Ratio** is fixed at 16:9. **Width** accepts 512–4096 in steps of 512 and starts at
1536 × 864; it is saved separately from the image tab's size. Portrait fraction
stays at 45%. Steps, CFG, negative prompt, seed mode/value, reference size and
output folder use the shared controls. Descriptions and reference selections
stay in memory. Generate requires server support for the operation and either
a character description or a reference.

**Sheet / Portrait / Views** select the three outputs of one job in the same
viewer. Changing the selection resets to Fit; 1:1 and Pan apply to the selected
output. The existing progress area names Portrait, Views and Composing; sampling
percentage and ETA refer only to the current pass. Cancel and Retry retain the
same job semantics as image generation. Async events always return to their
originating tab, even while another tab is visible.

The client downloads and decodes all three roles before saving them to the
selected folder's local-date subfolder as `image_0001_character_sheet.png`,
`image_0001_portrait.png` and `image_0001_turnaround.png`, sharing one sequence
number. **Saved** appears only after all files have
been synced. Handled write failures remove files created by that save attempt;
if cleanup itself fails the UI reports partial files. A process/system crash
can leave partial files. Retry retrieves the existing job and uses the currently
selected output folder, without rerunning inference.

## Upscale

**Upscale** replaces Generate again in both desks. It uses the displayed image
result's server asset, or the whole composed character sheet even when Portrait
or Views is selected. The original files and the sheet's separate Portrait/Views
results are retained; the new image replaces the main result after saving.

Choose **Upscale target** in Settings: **2K** (default), **3K** or **4K** means
2048, 3072 or 4096 pixels on the longest side. The image desk uses its selected
Ratio; character sheets use 16:9. The short edge rounds to the nearest multiple
of 32, matching generation sizing. For example, 3K at 3:4 gives 2304 × 3072;
a 4K character sheet is 4096 × 2304. The button tooltip shows the target size.
This setting saves independently of the normal generation size.

The request uses the server's existing `image_upscale` profile and one registered
output asset, with the selected seed mode/value and explicit target ratio/size.
Prompt, reference uploads and normal Steps/CFG controls do not override that
profile. Its stored private prompt requires the server's Helto session to be
unlocked. Upscale shares Cancel/Retry and save recovery with generation; changing
settings while it runs does not alter its captured request. It requires a saved
result in the current UI session and an asset still available on the server.

## Output folder and recovery

During sampling, the progress bar shows completed steps and percentage received
from the API. **Sampling ETA** and **ends** estimate remaining sampling time and
its local-clock finish time from measured step durations. They do not include
model loading, final VAE decoding or disk saving. These stages show status text;
the image is only reported as **Saved** after the local PNG write completes.
Neither the bar nor the percentage advances from a timer. Estimates disappear
when sampling completes or cancellation is requested.

After successful local saving, the footer shows 100% and the measured duration
of that client attempt (including submission, polling, download and saving).
This is not a GPU-only benchmark; a recovery attempt measures only that attempt.
Hovering the footer exposes the saved path.

Upscale beside Generate enhances the saved result as described above. Generate
uses the current editor values. Retry appears beside a recoverable error and
retains the unresolved
job/idempotency key. Cancel remains a separate icon. Fit and Actual pixels are separate view actions;
the hand tool enables/disables dragging at actual pixel scale. Selecting the hand
from Fit enters actual pixel view. Images always preserve their proportions.

The default folder is `<project>/.imagine/images`. Preferences live in
`<project>/.imagine/ui-settings.json` and contain the output folder, privacy choice
and negative prompt, plus steps, CFG, seed mode/value, ratio, max side and reference size. Changes
are saved automatically; existing folder-only settings
remain compatible. Both
are outside Git. A user-selected folder can be anywhere the process can write;
files there are normal unencrypted PNGs, subject to that folder's permissions.
The server retains its original asset according to the API's storage behavior.

New generations are saved under `<output folder>/YYYY-MM-DD/` using the local
calendar date when saving begins. Images and character-sheet sets share one
sequence per date, starting at `image_0001.png`; numbers have at least four digits
so ordinary name sorting puts 0009 before 0010. Existing filenames determine the
next number after restarting the UI, including partial sheet sets. Existing
files are neither renamed nor moved.

Upscale retains the original directory and base name, even on another day or
after selecting a different output folder: `image_0001_upscaled.png`, followed
by `image_0001_upscaled_2.png`, `_3.png`, etc. Upscaling an already upscaled image
continues that same variant sequence. Sheets use
`image_0002_character_sheet_upscaled.png`; their separate portrait/views files
keep their names. Number allocation is protected by a directory lock across UI
processes. The destination filesystem must support this advisory lock.

`create_new` prevents replacing an existing file; the client syncs the file before
reporting success.
Handled write failures remove that client's partial file. An abrupt process or
system crash during writing can leave an incomplete file; it is never announced
as successfully saved. No hard-link support is required from the destination.

Network operations, image decoding and generation run off the UI thread.
**Retry** preserves the original request, idempotency key and known job ID.
After an ambiguous submission it uses the same idempotency key; after acceptance
it polls/downloads the same job. It does not silently generate a new image.
If saving a new generation fails, select another output folder and retry to
download/save the completed server asset. For an upscale, restore write access
to the original image folder and retry; its destination stays beside the source.
Prompts and generation settings remain editable while
an image is being generated; changes prepare the next Generate action. The
running job keeps its captured request and output folder, and its completion
does not replace the draft. Retry retains the original request but uses the
currently selected output folder for saving new generations; an upscale retains
its captured source path. Reference order stays locked
while a legacy edit, sheet or upscale job is pending; queued image prompts allow
changing references for subsequent submissions. In-progress jobs are not restored by the UI after closing it; they
remain accessible through the API until Imagine stops. Server restart clears all
history and assets; already saved local images remain. The result strip is a
session-only browser, with no persistent history.

## Helto adaptation

`ui/src/helto.rs` maps the canonical Helto palette to native egui: Mocha surface
hierarchy, peach/gold primary/selection, blue keyboard focus, semantic errors/status
and 5–6 px control/panel radii. The selected Reference desk mockup uses 32 px icon
controls and 14 px body text. `ui/src/icons.rs` draws the outline glyphs and
`ui/src/workspace.rs` owns the workspace and settings layout.
Narrow windows place the image above references and prompt, with scrolling.
No web font, browser component or JavaScript frontend is involved.

## Checks

```sh
cargo fmt --all --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace
cargo build --release --locked -p imagine-ui
./start-ui.sh --help
# Optional synthetic software previews; no desktop/window/GPU:
IMAGINE_UI_PREVIEW_DIR="$PWD/target/ui-previews" cargo test --locked -p imagine-ui --lib headless_layout
# Reference desk and Settings side by side, plus the compact layout:
IMAGINE_UI_PREVIEW_DIR="$PWD/target/reference-desk-previews" cargo test --locked -p imagine-ui --lib headless_reference_desk
# Character-sheet layouts and all result roles, using only procedural pixels:
IMAGINE_UI_PREVIEW_DIR="$PWD/target/character-sheet-previews" cargo test --locked -p imagine-ui --lib headless_character_sheet
# Editing layout and concealment, using only procedural pixels:
IMAGINE_UI_PREVIEW_DIR="$PWD/target/edit-previews" cargo test --locked -p imagine-ui --lib headless_edit_layout_and_privacy
# Explicit opt-in native preview with a temporary profile and procedural images:
cargo run --locked -p imagine-ui --features ui-preview --example reference-desk
```

The native preview is excluded from normal builds. It never loads normal runtime
preferences, media or credentials. Its synthetic client targets unused loopback
port 1 and never connects at startup; it is for UI interaction, not inference.
Closing it removes its temporary profile. API behavior is tested separately with
the real HTTP client and a fake inference backend.

The client integration test uses the actual API router and SQLite with a fake
inference backend. It covers defaults, ordered uploads, generated PNG saving,
idempotent retries and recovery to another folder after a post-generation write
failure. Headless egui tests cover wide/narrow rendering and pending-job recovery.
Privacy tests inspect actual paint meshes for both viewport and thumbnail sizes:
initial concealment, pointer entry/exit, leaving the window, focus loss, clipped
areas and isolation from neighboring images. Hidden images submit no image mesh.
Synthetic software previews inspect geometry and tokens, not native compositor
rendering. Native window interaction and XDG file-dialog acceptance require a
separate manual check; no desktop-control tools are used by these tests.

### Local acceptance — 2026-10-14

Release build, launch-script `--help`, shell syntax, formatting, workspace Clippy
with warnings denied, and all **11 workspace tests** passed (7 server, 4 UI).
The canonical Helto asset validator also passed. Wide (1180×760) and compact
(640×900) synthetic previews were rendered and inspected under
`target/ui-previews/`. Keyboard activation, focus and disabled controls were
checked headlessly. Native window/portal interaction was not run, and no new GPU
inference was needed for this client-only change. Dependency fetching and the
loopback HTTP test required elevated tool access; no desktop control was used.
The backend's earlier real GPU acceptance remains in `API-VALIDATION.md`.

The subsequent hover-reveal change passed all four UI unit tests, workspace
Clippy with warnings denied, formatting and a release UI build. Hidden wide and
compact synthetic previews were inspected. Native window behavior was not
manually tested; this change required no elevated access or GPU inference.

The generation-controls and English UI update passed 18 unit/integration tests
(8 server, 10 UI), workspace Clippy with warnings denied, formatting and release
builds for both applications. The real HTTP API test used a fake inference backend
and required elevated loopback access. It verified custom size, steps, CFG and
fixed seed reaching the backend, plus retry behavior. Sizing fixtures were checked
against the local AIO source. No new GPU generation or native window test was run.

Progress and ETA checks additionally cover split native stdout records before a
newline, byte-counter rejection, sequential steps and measured ETA, job ownership
and stale progress isolation in GET/SSE, UI client delivery and clearing on finish.
Synthetic progress previews cover wide and compact layouts. The parser is based
on the frozen source's output format; a new live GPU run has not been performed.

### Proxy route — 2026-10-15

The UI now defaults to the image proxy's `/imagine` namespace. The root launchers
share a coordinator socket by default. The real UI HTTP client was exercised
through the real Rust proxy and Imagine API with a fake inference engine and
synthetic backend controls. Uploads, model defaults, custom generation settings,
progress, PNG downloads, retry without duplicate generation, output-folder
recovery and cancellation errors passed. Additional assertions cover anonymous
401 responses, client/admin separation, cross-client asset/job 404s and SSE.
A proxy regression test verifies explicit 503/502 errors when coordination or
Imagine is unavailable, encoded-prefix rejection, and request-counter cleanup.

All 12 UI unit tests, the UI/proxy/API integration, the proxy failure regression,
workspace Clippy with warnings denied, formatting, release workspace build,
launch-script `--help` and shell syntax checks passed. Local socket tests used
approved escalation. No new GPU inference, native window interaction or
Computer Use was performed. Existing services were not restarted; changes are
uncommitted.

### Reference desk — 2026-10-15

The selected prototype is implemented as the image/reference/prompt workspace,
with icon actions, Image/Video tabs and a separate Settings dialog. Video remains
an unavailable workspace as scoped in the design request.

All 16 UI unit tests and the complete workspace suite passed, including the real
UI/proxy/API flow with fake inference. New interaction tests exercise tab state,
reference ordering/removal, pending-job control locking, Settings open/close and
Escape, numeric preference editing/persistence, preference-write failure/recovery,
privacy and Fit/Actual/Pan. Existing clipboard, dimensions and privacy-mesh tests
remain passing. Formatting, workspace/all-features Clippy with warnings denied,
the release UI build, launcher help and diff checks passed.

Synthetic CPU renderings cover wide, compact, progress and Settings views. Native
Hyprland interaction also checked the floating dialog's size, close control,
privacy switch and synchronized toolbar state, concealment of every preview,
and tab switching without losing the image workspace. The native example used
only a fresh temporary profile and procedural test pixels. Captures are under
`target/reference-desk-previews/`; they are validation artifacts, not product media.
Native file-chooser acceptance and live inference were not exercised. Desktop
access and local socket tests required approved sandbox escalation. No running
Imagine service was restarted.

Clipboard persistence checks use synthetic text and an isolated helper process:
large Unicode input, exact stdin delivery, rapid copies followed by shutdown,
helper failure/timeout fallback and stale-failure suppression. They do not read
or replace the desktop clipboard. A real Wayland copy/close/paste check remains
manual acceptance.

### Character sheets — 2026-10-16

The new tab reuses the existing Reference desk and Helto components. The complete
workspace suite passed: 90 tests, including 32 UI unit tests and two HTTP client
integration tests. The real proxy/API with fake inference covers text-only and
reference sheets, both sampling stages, all three saved roles, cancellation and
save recovery without rerunning inference. A separate isolated HTTP fixture
verifies that an interrupted third download saves nothing and retries the same job.
Headless interaction checks cover per-tab drafts/results, async upload/job routing,
reference limits, width persistence, result selection/zoom reset, unavailable
server support and pending-job controls.

Formatting, workspace/all-targets/all-features Clippy with warnings denied, the
release UI build and launcher help passed. Synthetic CPU renderings of the actual
UI were inspected for desktop, compact, progress and privacy states. These are
procedural fixtures, not runtime images or native compositor captures. Local
socket tests required sandbox escalation; native window/file-dialog acceptance
and live GPU generation were not run. Existing services were not restarted.

### Upscale — 2026-10-16

All 94 workspace tests passed, including 36 UI unit tests. The real proxy/API
test uses synthetic prompts and fake inference with the shipped upscale profile
settings: image 3:4 at 3K and sheet 16:9 at 2K, registered output references,
profile sampling defaults, new files and save retries without repeated inference.
Headless tests cover persisted targets, every supported ratio, captured retry
settings and replacing only the composed sheet while retaining Portrait/Views.

Formatting, all-targets/all-features Clippy with warnings denied, the release UI
build and launcher help passed. Synthetic CPU-rendered settings and image/sheet
layouts were inspected. Local socket tests used sandbox escalation; no live GPU
generation, native window interaction or service restart was performed.

### Dated output folders — 2026-10-16

All 96 workspace tests passed, including 38 UI unit tests. Synthetic filesystem
checks cover dated directories, shared image/sheet numbering, sorting beyond nine,
existing partial sets, concurrent saves and repeated upscales beside an older
source. The real proxy/API fixture verifies image/sheet file layout, save recovery
and upscale destination retention without repeating inference. Formatting,
all-targets/all-features Clippy, release UI build and launcher help passed.
Only synthetic data was used; socket tests required sandbox escalation. No live
generation or native window check was run.

### Image editing with generic references — 2026-10-16

All 105 workspace tests passed, including 47 UI unit tests. New checks cover
stable sparse reference numbers, source-first model bindings, exact user prompt
preservation, source dimensions, missing references, nine extra inputs, isolated
drafts, async routing across tabs, version selection, cancellation, upload errors
and frozen retry requests. The real proxy/API fixture with fake inference edits
a registered result with extra references, preserves the original file and
recovers saving without repeating inference. Missing source assets are rejected.

Headless interaction and CPU-rendering checks cover desktop, compact and privacy
states using procedural images only; those renderings were visually inspected.
Formatting, workspace/all-targets/all-features Clippy with warnings denied,
release UI build, launcher help and diff checks passed. The initial socket test
was blocked by the sandbox; the passing workspace run used approved escalation
for isolated local test ports. No runtime data, live model generation, Computer
Use, native window/file-dialog interaction or service restart was involved.
Multi-reference model fidelity remains unvalidated. Changes are uncommitted.

### Hover comparison in image editing — 2026-10-16

All 49 UI unit tests passed. New synthetic checks cover version source links,
branches from Original, pointer-driven clipping, shared zoom/pan geometry,
HiDPI, leaving the image, window focus and privacy. A CPU-rendered comparison
using procedural pixels was visually inspected. UI all-targets/all-features
Clippy with warnings denied, workspace formatting, release UI build and diff
checks passed. No runtime data, model generation or Computer Use was involved;
native-window interaction remains untested. No escalation was needed for these
checks. Changes are uncommitted.

### Token-free local API — 2026-10-16

Imagine credentials have been removed from API, UI and proxy integration. This
supersedes the earlier authentication/owner-separation acceptance notes above.
The full workspace run passed 106 Rust tests. A final UI-only run against the
latest UI work passed 49 unit tests and two synthetic HTTP integration tests. Checks cover shared local assets/jobs,
idempotent retries, cancellation, unload/resume, GPU coordination, stale sessions,
MCP recovery, refusal of a non-loopback Imagine proxy listener, and stripping
Authorization while preserving session/idempotency headers. Two isolated actual
server-binary tests passed without a credentials file or model loading.

Formatting, all-targets/all-features Clippy with warnings denied, release builds
of all workspace binaries, CLI help checks and syntax checks of six updated
Python test scripts passed. Socket/bin tests required sandbox escalation. Live
GPU and native UI acceptance were not run; running services were not restarted.
Restart API, proxy and UI together to use the new connection contract. Existing
`access.json` files remain untouched and unused. Helto profile decryption and
separate proxy MCP authentication retain their existing behavior.

### Model selection and LoRA desk

Model selection and the LoRA button are beside the prompt in generation and
editing. Settings contains the model root, refresh and component selectors.
LoRA choices persist per model group; missing active choices block submission.
See [Models and LoRAs](MODELS.md) for supported files, discovery, API and persistence.

Validation: all **125 workspace tests** passed, including the actual UI HTTP
client through the proxy/API with synthetic model headers and fake inference.
New catalog and UI checks cover discovery, persistence, group/variant isolation,
missing/replaced files, component ambiguity, queued snapshots and widget-driven
LoRA add/remove/close. Regression checks also cover LoRAs four directories deep,
unfiltered LoRA selection regardless of headers or metadata, and visible Settings controls with long
paths and 40 collapsed scan warnings. Widget tests verify immediate multi-word
search across path segments, reversed word order, case, whitespace, no matches
and clearing the query. CPU-rendered desktop, compact, LoRA and
Settings views were inspected. Local test ports required sandbox escalation. No runtime data,
live model generation, Computer Use or service restart was involved. Native
window/file-dialog acceptance and real LoRA compatibility remain untested.

Formatting, workspace/all-targets/all-features Clippy with warnings denied,
release workspace builds, UI launcher help and diff checks also passed. The
implementation builds on baseline commit `c81e0af`.

### Image queue and bottom dock — 2026-10-19

The workspace suite passed, including 83 UI unit tests. New synthetic checks
cover frozen series settings, multiline prompt rows, continued composition,
result selection without draft replacement, selected-image edit/upscale sources,
cancel acknowledgement versus completion, and stale asynchronous results.
An isolated HTTP fixture verifies uncertain batch admission and result retrieval
retries without duplicate jobs, plus thumbnail-driven loading of saved images.
The real UI client through the proxy/API verifies atomic batch rejection,
idempotent admission, FIFO execution and cancellation of a queued job with fake
inference. Socket tests used approved sandbox escalation.

Workspace formatting, all-targets/all-features Clippy with warnings denied,
release UI build and diff checks passed. Synthetic CPU renderings were inspected
at desktop and compact sizes and with privacy enabled. Mesh assertions verify
that hovering the dock header, a thumbnail or empty panel space reveals every
thumbnail together; leaving or losing focus submits none of their textures.
Live model/GPU generation and native-window interaction were not run. No runtime
data was read, no running service was restarted, and changes remain uncommitted.
