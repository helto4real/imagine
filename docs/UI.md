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

## Workflow

1. Enter a **Prompt**. **Negative prompt** is optional and restored from your last edit.
2. Optionally choose **Add** to upload PNG/JPEG/WebP reference images.
   Thumbnails are numbered; the up/down controls change reference order.
3. Choose **Save to → Choose folder**. The folder choice is saved immediately.
4. Choose **Generate**. The result is shown after its PNG has been saved locally.

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

ER-SDE/beta, one PNG and the 1 MP per-reference resize target remain model
defaults. Native code and the frozen configuration are unchanged. Other output
sizes are accepted by the API but have no new live GPU or image-quality validation;
the frozen measurement scope is still 1600×896. Large images may exceed memory.
Restart both the API and UI after updating to enable the new dimensions contract.

Reference count is limited to ten. The existing server upload limits apply.
The **Privacy: on / off** button toggles privacy for the result and reference images.
It is on by default and the choice is saved immediately. When on, images use hover-reveal privacy:
images stay opaque and hidden until the pointer enters that image area. Leaving
the area or losing window focus hides them immediately. This affects the preview
only; saved PNGs are unchanged.
Uploading does not resize the source file: the native per-image resize setting
controls inference. The client creates small thumbnails only for display. The
file dialog's order is used initially and can be adjusted before submission.
Only one reference has local model-quality validation. H3 is not exposed here.

## Output folder and recovery

During sampling, the progress bar shows completed steps and percentage received
from the API. **Sampling ETA** and **ends** estimate remaining sampling time and
its local-clock finish time from measured step durations. They do not include
model loading, final VAE decoding or disk saving. These stages show status text;
the image is only reported as **Saved** after the local PNG write completes.
Neither the bar nor the percentage advances from a timer. Estimates disappear
when sampling completes or cancellation is requested.

The default folder is `<project>/.imagine/images`. Preferences live in
`<project>/.imagine/ui-settings.json` and contain the output folder, privacy choice
and negative prompt, plus steps, CFG, seed mode/value, ratio and max side. Changes
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
completed server asset. The request and reference order are fixed during a
pending job. In-progress jobs are not restored by the UI after closing it; they
remain accessible through the API until Imagine stops. Server restart clears all
history and assets; already saved local images remain. The current UI has no history browser.

## Helto adaptation

`ui/src/helto.rs` maps the canonical Helto tokens and component recipes to native
egui: Mocha surface hierarchy, peach/gold primary/selection, blue keyboard focus,
semantic errors/status, 5–6 px control/panel radii and 24 px gradient buttons.
The main surface has a compact prompt/reference column and an aspect-preserving
image viewport. Narrow windows stack the two areas with scrolling.
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
```

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
