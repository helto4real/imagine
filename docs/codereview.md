# Imagine code review

- Date: 2026-10-27.
- Source revision at final inspection: c11c39648c4753b896d1584a72eb21344bbdc602.
- Method: static source and documentation review.

## Assessment

The main risks are incomplete transaction boundaries, recovery behavior and excessive coupling between application state and filesystem operations. The highest-priority finding is a launcher that terminates unrelated processes when their ports conflict. Other actionable findings affect interrupted uploads, request retries, settings undo and unsaved UI drafts.

For privacy, the normal product startup routes persistent project content, generated media and private temporary files onto the verified VeraCrypt volume. I found no confirmed ordinary product path that automatically persists submitted private prompt text or generated private media outside that volume. This is a source-level conclusion about the paths inspected, not a filesystem audit or a guarantee about external programs.

The negative prompt is treated as a non-sensitive template, as explicitly clarified by Tomas. Its presence in the public model settings is **not a finding**. Application-level encryption of files already on the encrypted volume is also **not requested or recommended**.

## Scope and evidence limits

The review covers correctness, data integrity, privacy, security boundaries, architecture, code smells, UI behavior, maintainability and test design.

| Area | Inspection performed |
| --- | --- |
| Product startup and builds | Entry scripts, product builder, private-storage startup, native build lock and worker allowlist. |
| Server and job lifecycle | Request admission, idempotency, scheduling, cancellation, output publication, session storage and media delivery. |
| Private storage | Layout, mount verification, directory descriptors, atomic writes, project bundles, external references and workspace files. |
| Proxy and MCP | Imagine forwarding, workspace authentication boundary, session handling, logging and stack supervision. |
| Desktop UI | Application state, queue/results, settings, model selection, reuse/undo, Timeline drafts, close behavior, shared controls and selected authoring controls. |
| Inference integration | Product worker entry points, pipe protocols, cache/temp placement, optional Comfy adapter and selected native filesystem call sites. |
| Tests | Relevant source-owned test helpers and regression cases were read to evaluate coverage. No tests were executed. |
| Documentation/tooling | README, privacy/build guidance and the synthetic benchmark's output destinations. |

No runtime logs, submitted prompts, databases, job history, session history, generated media or runtime reference files were read. Nothing on the encrypted volume was accessed. No service, browser, GUI, model or GPU workload was started. No subagents were used.

This is a broad repository review with detailed inspection of the product's high-risk boundaries. It is not a line-by-line audit of every vendored file, archived experiment, generated inference table or CUDA operation. Numerical model parity, runtime performance, dependency vulnerability status and rendered UI appearance remain unverified.

Concurrent uncommitted inference work appeared during the review in numeric.rs, its Qwen test support and a new reference-fixture directory. That in-progress work is not assessed as a completed change here. The cited product files were unchanged by those edits, and this review did not modify or validate that work.

The published native product reconstructs its core from pinned revisions and patches; it does not simply execute every current file under the inference checkout. Findings here distinguish current product adapters and startup from experimental tools. See [the product lock](/home/thhel/ChatGPT/imagine/native/product/build-lock.json:3) and [worker preparation](/home/thhel/ChatGPT/imagine/tools/product_build.py:294).

## Prioritized findings

Status 2026-10-31: ten findings are fixed in commits that name them. CR-09
(save under the service-wide mutex, `src/video/api.rs` `save`) and CR-11
(dismissed results keep their completed queue jobs, `ui/src/queue_ui.rs`) are
still open. The architecture observations A-01 to A-06 have not been taken up.
The status was checked against commit messages, the files they change and the
cited code; no tests were run for this check.

P1 means address before relying on the affected operation; P2 means a substantive correctness, integrity or resilience issue; P3 means a lower-priority resource or usability improvement. “Conditional” identifies a required environment or failure condition, not a reproduced runtime failure.

| ID | Priority | Area | Finding | Evidence | Status 2026-10-31 |
| --- | --- | --- | --- | --- | --- |
| CR-01 | P1 | Startup | A port conflict can terminate an unrelated process or process group. | Source-confirmed | Fixed in `662ae33` |
| CR-02 | P2 | Uploads | An interrupted workspace upload can still publish its partial file. | Source-confirmed | Fixed in `9cf8cd2` |
| CR-03 | P2 | API | Single-job idempotency hashes differ before and after default destination injection. | Source-confirmed | Fixed in `1ef2895` |
| CR-04 | P2 | Storage | The fallback for unsupported no-replace rename can overwrite a concurrent upload. | Conditional: filesystem rejects the flag | Fixed in `20303ff` |
| CR-05 | P2 | UI | Closing the main window has no protection for pending Timeline drafts. | Source-confirmed | Fixed in `f0e5382` |
| CR-06 | P2 | UI/settings | Undo Reuse settings overwrites later changes to unrelated model groups. | Source-confirmed | Fixed in `360a3e5` |
| CR-07 | P2 | Workspaces | Key rotation can commit the hash without committing the recoverable key. | Conditional: second write fails | Fixed in `b72907b` |
| CR-08 | P2 | HTTP security | Token-free API/bridge ingress lacks the Host/Origin checks already present on MCP. | Missing checks confirmed; browser exploit untested | Fixed in `db93463` |
| CR-09 | P2 | Architecture | Synchronous project persistence runs while holding a service-wide async mutex. | Design exposure confirmed; latency unmeasured | Open |
| CR-10 | P2 | Documentation/privacy | Documentation promises legacy cleanup that normal startup does not perform. | Source-confirmed | Fixed in `ae3f890` |
| CR-11 | P3 | UI/resources | Removing a result leaves its completed queue job and input snapshot retained. | Source-confirmed | Open |
| CR-12 | P3 | UI copy | An authoring selector displays Rust variant names instead of its own option labels. | Source-confirmed | Fixed in `90cdc54` |

### CR-01 — Do not kill arbitrary owners of conflicting proxy ports

**Sources:** [port owner signaling](/home/thhel/ChatGPT/imagine/proxy/st-stack.zsh:622), [port release](/home/thhel/ChatGPT/imagine/proxy/st-stack.zsh:636), [normal startup call](/home/thhel/ChatGPT/imagine/proxy/st-stack.zsh:1289).

When the supervisor does not recognize an existing proxy, startup calls **ensure_proxy_ports_available**. **free_proxy_port** obtains the port owners and sends SIGTERM; after the timeout it sends SIGKILL. **signal_port_owners** can signal the entire process group when the owner is its group leader.

The code does not establish that these processes belong to this Imagine stack before terminating them. A different development server on a configured port, or an older instance outside the expected directory, can therefore be killed by an ordinary start command. Other members of its process group can be affected too.

**Recommended change:** reject the conflicting port with an actionable error. Automatic termination should require a verified, recorded stack-owned process identity. Keep any deliberate takeover operation explicit.

**Regression check to add:** occupy a configured port with an unrelated synthetic listener; startup must fail while preserving both that listener and its process group.

### CR-02 — Publish workspace uploads only after explicit successful completion

**Sources:** [upload writer and interruption handling](/home/thhel/ChatGPT/imagine/src/video/workspace_api.rs:178), [workspace upload allocation](/home/thhel/ChatGPT/imagine/src/workspaces.rs:483).

The blocking writer drains an MPSC channel and calls **file.commit()** whenever that channel closes after receiving nonzero bytes. The HTTP handler also closes the same channel on a body-stream error. It then awaits the writer and returns **upload_interrupted**.

Channel closure therefore means both “complete upload” and “upload failed.” The comment claiming that dropping the receiver side prevents commit is incorrect: the code drops the sender, and the writer takes its normal commit path. Dropping the HTTP handler can similarly leave the detached blocking writer treating sender disappearance as success.

A connection interrupted after a valid media header can leave a truncated file under a final, visible workspace filename despite the error response. A later listing identifies media from its header; it does not establish that this upload completed.

**Recommended change:** require an explicit success/commit message after normal EOF. Stream errors, sender disappearance and cancellation should drop the uncommitted file. Reuse the completion/cancellation principles already present in [streaming media ingress](/home/thhel/ChatGPT/imagine/src/video/streaming_import.rs:70).

**Regression checks to add:** body-stream error after several chunks, dropped request future and successful EOF. Only successful EOF may create a final file.

### CR-03 — Use one canonical fingerprint for single-job idempotency

**Sources:** [retry comparison and insertion](/home/thhel/ChatGPT/imagine/src/scheduler.rs:448), [destination injection and second fingerprint](/home/thhel/ChatGPT/imagine/src/scheduler.rs:588).

The single-job submit path hashes the incoming resolved request before looking up its idempotency key. For a new request, **prepare_request** subsequently inserts a server-selected **client_metadata.video_destination** when the caller omitted it, then computes another fingerprint. The database stores this second fingerprint.

Repeating the identical caller request without a destination proof compares the original request hash against the enriched request hash and returns **409 idempotency_conflict** instead of the original job. This affects the supported direct-API/default-destination path; clients that supply the destination themselves can avoid it. The separate batch fingerprint has different handling and should not be described as having this same defect.

The test helper [automatically inserts a destination proof](/home/thhel/ChatGPT/imagine/tests/api_lifecycle.rs:355), which hides this distinction from existing helper-based idempotency scenarios.

**Recommended change:** define and use one canonical idempotency identity for both lookup and insertion. Keep server-generated execution data separate from caller request identity, and preserve reuse after a lost response.

**Regression check to add:** send the exact same single-job JSON and key twice through the raw route, without helper enrichment or client metadata. Expect the same job ID and one admitted job; changed input with that key must still conflict.

### CR-04 — Preserve atomic no-overwrite semantics in the rename fallback

**Sources:** [NewFile commit fallback](/home/thhel/ChatGPT/imagine/shared/private_storage.rs:473), [upload filename selection](/home/thhel/ChatGPT/imagine/src/workspaces.rs:500).

When **renameat2(RENAME_NOREPLACE)** returns EINVAL, **NewFile::commit** checks the destination with **fstatat** and then falls back to ordinary **renameat**. There is no lock covering that check and rename.

Two uploads can choose the same currently unused filename. On a filesystem using this fallback, both can observe an absent destination and the second ordinary rename can replace the first completed file. The method's promise that an existing file is never replaced no longer holds.

This is a conditional race in the fallback, not evidence that the mounted production filesystem takes that branch. Its capabilities were not inspected.

**Recommended change:** use an actually atomic no-replace publication mechanism supported by the filesystem, or fail closed when that guarantee is unavailable. Do not replace it with another existence-check-then-rename sequence.

**Regression check to add:** force the unsupported-flag branch and concurrently publish two distinct payloads under one name. Exactly one may succeed, and the winner's bytes must remain unchanged.

### CR-05 — Protect pending Timeline drafts when closing the application

**Sources:** [unsaved-work detection](/home/thhel/ChatGPT/imagine/ui/src/video/mod.rs:1259), [replacement confirmation](/home/thhel/ChatGPT/imagine/ui/src/video/mod.rs:1270), [application exit](/home/thhel/ChatGPT/imagine/ui/src/app.rs:982).

The UI already detects pending Timeline edits and protects project replacement through New, Open and Animate. The main application close path does not use that protection. **on_exit** pauses playback and saves preferences, but does not commit or confirm the disposition of Timeline drafts. No main-window close cancellation path was found.

Closing with locally edited scene/section/global prompt or other pending draft fields can discard those drafts. Server-side autosaved edits remain saved; this finding concerns local, not-yet-submitted edits, not loss of the whole saved project.

**Recommended change:** intercept a normal main-window close while local drafts remain, using the existing unsaved-work detection. Offer a concrete save/discard/cancel decision and wait for an acknowledged save before exiting when save is chosen.

**Regression checks to add:** close with a pending text draft, cancel close, save and close, and explicit discard. Confirm that already autosaved projects do not acquire unnecessary close prompts.

### CR-06 — Undo only the model-group settings replaced by Reuse settings

**Sources:** [reuse action](/home/thhel/ChatGPT/imagine/ui/src/queue.rs:551), [undo action](/home/thhel/ChatGPT/imagine/ui/src/queue.rs:590), [existing undo test](/home/thhel/ChatGPT/imagine/ui/src/queue_tests.rs:736).

Reuse settings updates the active generation configuration, model-group selection and one group's settings. Its undo stores a complete **Preferences** snapshot and restores the entire **model_groups** map.

For example: reuse a result for group A, change LoRAs or a template for group B, then choose Undo Reuse settings before generating again. Group B's later changes are replaced with the old snapshot even though the original reuse operation did not modify group B. The undo path persists the overwritten settings.

The existing test checks that a later privacy toggle survives. That does not cover later changes within the model-group map.

**Recommended change:** record the specific group entry and fields that reuse actually changes. Undo that patch while preserving unrelated entries; explicitly define how later edits to the same affected fields should behave.

**Regression check to add:** modify a different group after reuse, invoke undo and verify that the later group's settings survive both in memory and after reload.

### CR-07 — Make workspace key rotation recoverable across partial writes

**Sources:** [key rotation](/home/thhel/ChatGPT/imagine/src/workspaces.rs:350), [current-key retrieval](/home/thhel/ChatGPT/imagine/src/workspaces.rs:332).

Rotation first publishes the new key hash in **workspace.json**, then writes the plaintext recovery key to **AGENT-KEY.txt**, and only returns the new key after both writes succeed.

If the second write fails, the old key is already invalid, the stored key file still contains the old key, and the caller never receives the new key. Current-key retrieval then reports **workspace_key_missing** because the two files disagree. Existing workspace content is not deleted, but normal agent access and key recovery are disrupted until an operator repairs or repeats the operation successfully.

Each file write is atomic individually; the two-file operation is not.

**Recommended change:** establish one authoritative commit record and make the companion representation recoverable, or provide an explicit transaction/recovery protocol. Simply reversing the two writes creates the opposite inconsistency and is insufficient.

**Regression checks to add:** inject failure before and after each publication step and simulate interruption between them. The old key or a recoverable new key must remain available.

### CR-08 — Apply browser-facing Host/Origin checks at the API and bridge ingress

**Sources:** [API session middleware](/home/thhel/ChatGPT/imagine/src/api.rs:84), [Imagine bridge](/home/thhel/ChatGPT/imagine/proxy/src/service.rs:867), [proxy header rewriting](/home/thhel/ChatGPT/imagine/proxy/src/http.rs:20), [existing MCP host protection](/home/thhel/ChatGPT/imagine/proxy/src/mcp_workspace.rs:456).

The server enforces loopback binding and checks supplied session headers, but its general request middleware does not validate Host or Origin. The Imagine forwarding route also lacks those ingress checks. In contrast, MCP explicitly validates the local host names against DNS rebinding.

The token-free main API exposes local administrative operations, workspace listings and key retrieval. A session ID is publicly discoverable and is a lifecycle guard, not a caller authentication secret. Loopback binding alone does not validate the browser origin addressing the service.

This leaves a source-level DNS-rebinding/cross-origin protection gap. A browser exploit was not attempted, and actual exploitability depends on browser and network restrictions. The finding is not a request to restore the removed bearer-token or password workflow.

**Recommended change:** preserve token-free native clients while checking allowed local Host values and rejecting unexpected browser Origins at both ingress points. Validate the original incoming headers before proxy rewriting; an upstream-only Origin check would see the rewritten value.

**Regression checks to add:** reject foreign Host and Origin values while accepting configured loopback clients and supported proxy paths.

**Trust-boundary clarification:** workspace keys isolate access on the MCP surface. A client allowed to reach the unrestricted local administration API can list workspaces and obtain keys without presenting a workspace key. That may be the intended trusted-local-client model, but filesystem sandbox denial by itself does not extend workspace-key isolation to HTTP.

### CR-09 — Move slow persistence outside the service-wide async critical section

**Sources:** [save route](/home/thhel/ChatGPT/imagine/src/video/api.rs:388), [bundle resource copying and verification](/home/thhel/ChatGPT/imagine/src/video/bundle.rs:484), [scheduler publication lock](/home/thhel/ChatGPT/imagine/src/scheduler.rs:908).

The save route holds the shared video-service mutex while executing synchronous project persistence. Bundle saving can inspect directories, read and hash existing owned resources, copy missing resources and perform multiple file and directory syncs.

These operations run in an async request task and serialize access to the service for all projects. A large project or slow/erroring filesystem can delay unrelated video operations. Scheduler result publication also needs this mutex, so a slow save can delay publication even after generation finishes.

The exposure follows directly from the lock scope; no latency or throughput numbers were measured. Moving work to a blocking thread while continuing to hold the same global lock would avoid blocking a Tokio worker but would not remove service-wide contention.

**Recommended change:** first isolate filesystem work from async workers. Then narrow ownership to a per-project transaction or snapshot/validate/commit boundary, retaining revision checks and atomic publication. Avoid a broad scheduler rewrite.

**Validation to add:** introduce a deliberately slow synthetic store, save one project and verify that another project's independent operations and cancellation path remain responsive.

### CR-10 — Correct the documented storage and legacy-cleanup guarantees

**Sources:** [privacy cleanup claim](/home/thhel/ChatGPT/imagine/docs/PRIVACY.md:23), [actual startup](/home/thhel/ChatGPT/imagine/src/main.rs:153), [unused legacy purge helper](/home/thhel/ChatGPT/imagine/src/session.rs:66), [README storage instructions](/home/thhel/ChatGPT/imagine/README.md:87).

The privacy document says startup deletes the legacy SQLite files and internal media directory. Normal startup instead opens the private layout and does not call **session::purge**. The document itself later describes a test that preserves legacy project data, contradicting the earlier cleanup promise.

README also describes the old **.imagine/images**, **.imagine/ui-settings.json**, **.imagine/gpu.sock** and **mcp-images** defaults. Current code uses the private project/workspace layout, split home preferences and an XDG runtime socket. Its suggested arbitrary **--data** override is rejected unless it matches the computed private work directory.

The privacy consequence is false confidence that old unencrypted data has been removed. The source establishes that these locations are left untouched; it does not establish that any such old data currently exists.

**Recommended change:** document the current storage classes, persistent output lifetime and absence of automatic legacy migration/deletion. Retain explicit user control over any historical cleanup; do not add automatic deletion as a documentation fix.

**Validation:** compare each documented path and lifecycle claim against the actual entry points. Any later cleanup tool should have its own separately authorized scope.

### CR-11 — Release completed job snapshots when their result is dismissed

**Sources:** [queue job ownership](/home/thhel/ChatGPT/imagine/ui/src/queue.rs:11), [completion handling](/home/thhel/ChatGPT/imagine/ui/src/queue.rs:401), [result removal](/home/thhel/ChatGPT/imagine/ui/src/queue_ui.rs:397).

Successful queue jobs remain in **queue.jobs**, holding their submission, prompt/preferences snapshot and reference texture handles. They are hidden from the queue rows once successful. Remove from results removes only the gallery item; it does not remove the corresponding successful job.

Over a long session, even dismissed results therefore leave otherwise inaccessible job/input state retained. This is unnecessary session-memory retention, not a confirmed disk leak. Reference previews are thumbnails, so this should not be described as retaining every original full-resolution image.

**Recommended change:** remove terminal jobs when their presentation state is no longer needed, or retain a bounded, explicit history record. Keep pending batch/retry semantics intact and preserve saved output files.

**Regression check to add:** finish and dismiss a batch, then verify that its obsolete jobs and snapshots are released while the saved images remain.

### CR-12 — Give authoring image-use modes stable user-facing labels

**Source:** [story image-use selector](/home/thhel/ChatGPT/imagine/ui/src/video/authoring.rs:1323).

The menu offers “Setting reference,” “Required moment,” “Directed start frame” and “Directed end frame,” but the selected value uses the Rust Debug representation. After selection, labels such as **SettingReference** or **StartFrame** appear instead of the wording the user selected.

This exposes internal enum names and changes terminology between the closed selector and its menu. It is a source-confirmed copy inconsistency; no rendered layout defect is claimed.

**Recommended change:** use one label mapping for the selected value and menu entries. Keep Debug formatting for diagnostics and synthetic QA identifiers only.

## Privacy and storage map

This map describes source-defined behavior. The paths were read as literals in source; none were opened on the encrypted volume.

| Data class | Source-defined location/lifetime | Assessment |
| --- | --- | --- |
| Image job requests, submitted prompts, queue/idempotency records | SQLite in memory, with SQLite temporary storage set to MEMORY. | No persistent database path in normal startup. |
| Ordinary uploaded image bytes | Session-owned in-memory upload map. | No automatic disk copy from this path. |
| Generated product images/videos/audio | Verified project output directories under /run/media/veracrypt1/imagine, or approved workspace/project folders on that volume. | Meets the stated at-rest requirement in the inspected normal path. |
| Video project prompts, cast, resources and autosaved documents | Private project bundle under the verified volume. | Intended persistent private data, not disposable job history. |
| MCP workspace keys, metadata, uploads and outputs | /run/media/veracrypt1/mcp-root, in dated workspace folders. | Intended private persistence; see CR-02, CR-04 and CR-07 for integrity defects. |
| Staging media and worker temporary files | Private project tmp directories; child TMPDIR/TMP/TEMP refer to a pinned directory descriptor. | Normal product workers do not intentionally fall back to /tmp for private payloads. |
| Private UI preferences and enhancement profiles | Private library files, including output-folder/catalog/profile selections. | Kept separate from public home preferences. |
| Public UI settings | ~/.config/imagine/ui.json. | Layout, display/privacy preference and generation settings; no authored project prompt field. |
| Model settings | ~/.config/imagine/models.json. | Model paths/selection, LoRAs and the explicitly accepted non-sensitive negative template. |
| Compiler/CUDA/Triton cache | ~/.cache/imagine, separated by worker family. | Inspected writes concern compiler artifacts and specialization data, not prompt strings or image tensors. |
| Runtime coordination/locks | $XDG_RUNTIME_DIR/imagine. | Socket/lock/control metadata, not intended media persistence. |
| Diagnostic output | Process stderr; proxy logging emits selected technical fields. | No ordinary prompt-text file sink found. The declared public logs path alone is not evidence of a file writer. |
| Synthetic Rust benchmark output | Repository-local comfyui/bench, including copied benchmark images and its workspace key. | An explicit tool exception outside the encrypted volume; its workloads are fixed synthetic inputs. Not evidence of copying ordinary private user jobs. |
| Explicitly linked reference originals | Remain at the user-selected source path. | The application does not encrypt, relocate or erase an original selected from another disk. |
| Historical legacy files | Older project/.imagine and other existing exports are not automatically migrated or purged. | Possible residual private data outside the volume cannot be confirmed without a separate filesystem review. |

Relevant sources:

- [Memory database and private output ownership](/home/thhel/ChatGPT/imagine/src/storage.rs:67).
- [Database memory configuration](/home/thhel/ChatGPT/imagine/src/storage.rs:179).
- [Image upload storage](/home/thhel/ChatGPT/imagine/src/storage.rs:823).
- [Storage classes](/home/thhel/ChatGPT/imagine/shared/storage_layout.rs:4).
- [Worker environment](/home/thhel/ChatGPT/imagine/src/config.rs:147).
- [Public/private preference serialization](/home/thhel/ChatGPT/imagine/ui/src/settings.rs:431).
- [Sanitized proxy diagnostics](/home/thhel/ChatGPT/imagine/proxy/src/logging.rs:157).
- [Synthetic benchmark output and key receipt](/home/thhel/ChatGPT/imagine/tools/rust_bench.py:151).
- [Synthetic image copies](/home/thhel/ChatGPT/imagine/tools/rust_bench.py:181).
- [Linked-original ownership](/home/thhel/ChatGPT/imagine/src/storage.rs:413).

OS swap, hibernation, external clipboard managers, terminal capture, filesystem snapshots/backups and external LLM services were not inspected. Their actual configuration cannot be inferred from repository source. No additional encryption requirement is inferred from these unverified external conditions.

## Architecture and code smells

These observations are separate from confirmed functional failures. They identify focused improvements, not authorization for a general refactor.

### A-01 — File splitting has outpaced responsibility separation in the UI

[ImagineApp](/home/thhel/ChatGPT/imagine/ui/src/app.rs:132) owns image generation, editing, sheets, model selection, preferences, enhancement, queue presentation and Timeline orchestration. Numerous child modules implement methods on the same parent type using broad parent imports. [Timeline](/home/thhel/ChatGPT/imagine/ui/src/video/mod.rs:105) similarly owns many feature states, async identities, drafts, playback and history.

The problem is not file size alone: unrelated workflows can access and replace each other's state. CR-06 and CR-11 are concrete examples of ownership/lifetime rules that are not expressed by a narrow boundary.

Prefer gradual extraction of cohesive state owners for queue/result lifecycle, settings changes and Timeline session/drafts. Keep orchestration in the app, but expose operations that enforce invariants rather than letting every module mutate the whole object. Preserve existing shared contracts.

### A-02 — Behavior depends on display strings and independent flags

[Queue jobs](/home/thhel/ChatGPT/imagine/ui/src/queue.rs:11) combine status strings with **working**, **cancelling**, **retryable**, optional job IDs and optional errors. [Cancellation recognition](/home/thhel/ChatGPT/imagine/ui/src/queue.rs:409) compares the user-facing error text against “Generation cancelled.” [Queue counts](/home/thhel/ChatGPT/imagine/ui/src/queue_ui.rs:22) interpret English status labels.

This makes copy changes capable of changing behavior and allows contradictory combinations of flags. Use typed internal lifecycle states and stable error codes, converting to display text at the UI boundary. Start with the queue and its cancellation/retry transitions rather than redesigning every API DTO.

### A-03 — Atomic file helpers do not imply atomic business operations

Private storage, output publication, project bundles and preferences each have their own filesystem mechanics and commit/error conventions. Their different ownership requirements are legitimate, but guarantees such as “no overwrite,” “committed but durability uncertain” and “operation failed without changing state” need explicit, consistent contracts.

CR-04 and CR-07 show where combining individually safe steps breaks the larger guarantee. Share narrowly compatible mechanics and document transaction outcomes; do not merge all storage owners just to reduce similar-looking code.

### A-04 — Preferences persistence can block the UI thread

[save_preferences](/home/thhel/ChatGPT/imagine/ui/src/app.rs:775) calls synchronous private and public persistence directly. Those operations include mount verification and file/directory syncs. Deferring an edit until typing settles reduces write frequency, but each actual save can still block a frame.

Use a small serialized settings-write owner or worker, with versioned acknowledgments and visible failure state. Preserve ordering between private/public portions and prevent a stale asynchronous save from replacing a newer one. No frame-time measurement was performed.

### A-05 — Test helpers should not silently repair the API contract under test

The existing suite has substantial synthetic coverage, including transaction failures, session changes and external-resource identity. That is valuable. However, the helper discussed in CR-03 enriches inputs before dispatch, so an apparently ordinary API test does not exercise an ordinary caller request.

Separate explicit fixture preparation from raw boundary calls. Add failure/cancellation cases at the adapter boundary where they occur. Avoid adding implementation-order assertions when a small observable contract regression can establish the required behavior.

### A-06 — Workspace lookups repeatedly traverse the entire workspace

[Workspaces::file](/home/thhel/ChatGPT/imagine/src/workspaces.rs:472) implements a single-file lookup by listing and walking all workspace files. Listing also opens file headers and may hash uncached files. Both [download](/home/thhel/ChatGPT/imagine/src/video/workspace_api.rs:117) and info operations can invoke this synchronously from an async route.

This creates work proportional to workspace size for a lookup by file ID and compounds the synchronous-I/O concerns in CR-09. Prefer a validated, invalidation-aware file index or a bounded lookup owner if workspace scale justifies it. Preserve path/symlink checks and content identity; do not trade integrity for caching. No benchmark was run.

## UI assessment

The source contains useful product patterns: shared command availability, retained drafts, project revision checks, canvas-based concealment, semantic colors and bounded preview handling. Keep those patterns.

The most actionable UI problems are **CR-05** and **CR-06**, because they can lose user edits. **CR-11** concerns the lifetime of state the UI no longer exposes. **CR-12** is a small terminology fix. **A-01**, **A-02** and **A-04** explain why future UI changes are harder than the file layout suggests.

A future visual acceptance pass should cover narrow layouts, independent settings/LoRA windows, keyboard focus, hidden/revealed canvases, popups and error recovery. This review does not claim that those states were rendered, clicked or manually verified. Existing source tests are evidence of intended checks, not successful results from this review.

Do not address these findings by adding persistent explanatory banners. Prefer correct transitions, consistent labels and concise, actionable errors. An unsaved-work confirmation at the consequential close action provides information the user actually needs.

## Strengths worth preserving

- **Private writes fail closed.** The normal entry requires a verified mounted volume, and private path handling checks filesystem identity rather than relying only on a path prefix.
- **Descriptor-based writes reduce fallback and symlink exposure.** Core private writes use directory descriptors, restricted permissions and atomic publication.
- **Job history is session-local.** SQLite history and ordinary upload bytes remain in memory; core dumps are disabled in the product entry points.
- **Original references have explicit ownership.** Linked originals are not silently copied, and identity/content checks exist before later use.
- **Product/native ownership is explicit.** Main product adapters and the frozen inference core have a documented build boundary. Do not bypass it to make an unrelated fix easier.
- **The video domain already has reusable typed contracts.** The contract crate, project revision checks, frozen generation inputs and prepared publication objects are useful foundations.
- **Privacy-aware diagnostics already exist.** The proxy's sink removes free-form paths/error text rather than trusting every log caller.

These controls substantially improve the baseline. They do not remove the narrower recovery and transaction defects above.

## Recommended implementation order

1. Fix CR-01's destructive startup behavior.
2. Fix CR-02 and CR-03 with focused boundary regressions.
3. Fix CR-04 and CR-07's publication/rotation guarantees, keeping private volume checks intact.
4. Fix CR-05 and CR-06's user-edit loss paths.
5. Add CR-08's ingress validation without restoring the removed login/token workflow.
6. Address CR-09 and A-04 at their actual blocking boundaries.
7. Correct CR-10, then handle CR-11, CR-12 and targeted architecture work.

Each item should remain independently reviewable. No fixes, service changes, dependency upgrades or inference optimizations were made as part of this review.

## Validation status and suggested follow-up

**Performed:** source/documentation inspection, call-path tracing, inspection of relevant existing test source, and document-only source-link and whitespace checks. The only requested repository change is this report.

**Not performed:** compilation, unit/integration tests, synthetic reproduction runs, patch restoration, live API requests, model generation, GPU measurements, browser/desktop automation, rendered UI acceptance or inspection of stored user data. No passing test results or runtime reproductions are claimed. No permission escalation occurred.

Suggested checks for an authorized implementation task:

| Change | Smallest useful validation |
| --- | --- |
| CR-01 | Isolated listener/process-group preservation test for the supervisor. |
| CR-02 | Failed-body and dropped-handler upload tests, plus complete-upload control. |
| CR-03 | Raw single-job idempotency test without destination enrichment. |
| CR-04 | Injected unsupported-rename branch with concurrent competing publishers. |
| CR-05 | UI close/cancel/save/discard acceptance using synthetic pending drafts. |
| CR-06 | Reuse/modify another group/undo/reload settings test. |
| CR-07 | Two-file rotation failure injection and restart recovery checks. |
| CR-08 | Host/Origin request matrix for direct API and proxy ingress. |
| CR-09/A-04 | Delayed synthetic filesystem operations with independent responsiveness checks. |
| CR-11 | Completed-job retention check after result dismissal. |
| CR-12 | Selected/menu label consistency check. |

Any future validation must continue using source-owned synthetic fixtures. Running an existing test command is not automatically safe: some repository checks require the real private mount or live services. Inspect the chosen test's setup first and keep source/synthetic, UI acceptance and GPU evidence separate.
