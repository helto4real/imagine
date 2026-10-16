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

Defaults: proxy API `http://127.0.0.1:8189/imagine`, the `local` client token in
`.imagine/access.json`. Start the server once to create that file. If the server
is offline, the window remains usable for writing a prompt and selecting an
output folder; choose **Reconnect** after starting Imagine and the proxy.

```sh
./start-ui.sh --api http://127.0.0.1:8189/imagine --access /private/api-data/access.json --client local
```

The proxy forwards `/imagine/*` on its image port to `--imagine-url` (default
`http://127.0.0.1:8080`), stripping the `/imagine` prefix. The UI uses that base
for metadata, uploads, job submissions/polling/cancellation and PNG downloads.
The proxy preserves bearer credentials and idempotency keys; Imagine enforces
client ownership and admin permissions. The HTTP forwarding does not reserve a
second GPU slot: Imagine reserves its job ticket in the shared coordinator FIFO.
There is no automatic direct fallback when the proxy is unavailable.

The root launchers default to `.imagine/gpu.sock` on both sides and
`.imagine/access.json` for the proxy's coordinator credentials. For custom paths,
set `IMAGINE_COORDINATOR_SOCKET` for the server and matching
`ST_PROXY_COORDINATOR_SOCKET` / `ST_PROXY_IMAGINE_ACCESS_FILE` for the proxy (CLI
options still override proxy environment defaults). Use `--api` for a different
proxy address or an explicitly chosen direct API. To run a standalone server
without coordination, invoke the server binary without `IMAGINE_COORDINATOR_SOCKET`.

The application reads only the selected client token and does not use the admin
token. Tokens are not accepted on the command line or stored in UI preferences.
The optional API URL supports HTTPS proxies; HTTP redirects are refused.

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

1. Enter a **Prompt**. **Negative prompt** is optional and restored from your last edit.
2. Optionally choose the **+** icon to upload PNG/JPEG/WebP reference images.
   Thumbnails are numbered; the up/down controls change reference order.
3. Open **Settings** (gear) and choose the folder icon under **Save to**.
   The folder choice is saved immediately.
4. Choose **Generate** (sparkles). The result is shown after its PNG has been saved locally.

The Reference desk layout places ordered references on the left, the image in
the center, and the prompt/size/seed inspector on the right. Every toolbar action
uses an outline icon with a tooltip and accessible name. The image, character-card
and film icons switch between Image, Character sheets and Video. Switching keeps
each workspace's drafts, references, results and pending job intact. Video currently has an explicit unavailable state, not a video engine
or editing tools.

The gear opens a separate native **Settings** window with **Generation** (Steps,
CFG, Reference size), **Storage** (output folder), and **Privacy** (Hide previews). Changes save
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
it immediately, including selected text and the caret. Editing focus alone does
not reveal text. This is visual concealment only: prompt content, clipboard
operations and saved PNGs are unchanged.
Uploading does not resize the source file: server preprocessing controls the
VAE and vision inputs, with further native resizing disabled. The client creates
small thumbnails only for display. The
file dialog's order is used initially and can be adjusted before submission.
Only one reference has local model-quality validation. H3 is not exposed here.

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
selected folder as `imagine-<UUID>-character_sheet.png`, `-portrait.png` and
`-turnaround.png`, sharing one UUID. **Saved** appears only after all files have
been synced. Handled write failures remove files created by that save attempt;
if cleanup itself fails the UI reports partial files. A process/system crash
can leave partial files. Retry retrieves the existing job and uses the currently
selected output folder, without rerunning inference.

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

The circular arrow beside Generate repeats the last successful generation with
its original prompt, ordered references and settings, using a new job and
idempotency key. Random seed requests stay random; fixed seeds remain fixed.
It saves to the currently selected output folder. The last successful request
is kept only in memory for this UI session. Generate uses the current editor
values. Retry appears beside a recoverable error and retains the unresolved
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

Each downloaded image is saved as `imagine-<UUID>.png`. `create_new` prevents
replacing an existing file; the client syncs the file before reporting success.
Handled write failures remove that client's partial file. An abrupt process or
system crash during writing can leave an incomplete file; it is never announced
as successfully saved. No hard-link support is required from the destination.

Network operations, image decoding and generation run off the UI thread.
**Retry** preserves the original request, idempotency key and known job ID.
After an ambiguous submission it uses the same idempotency key; after acceptance
it polls/downloads the same job. It does not silently generate a new image.
If saving fails, select another output folder and retry to download/save the
completed server asset. Prompts and generation settings remain editable while
an image is being generated; changes prepare the next Generate action. The
running job keeps its captured request and output folder, and its completion
does not replace the draft. Retry retains the original request but uses the
currently selected output folder for saving. Reference order stays locked
while a job is pending. In-progress jobs are not restored by the UI after closing it; they
remain accessible through the API until Imagine stops. Server restart clears all
history and assets; already saved local images remain. The current UI has no history browser.

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
