# Parity and evidence

Reference: st-proxy `10201aa1bc29f31977009d2c29a436da04eae497`. `reference/source-lock.json` records 76 source/test/workflow hashes, the original three untracked files (not copied or executed), installed MCP SDK sources/versions, and AIO color/tone sources. Ten workflow/profile files are byte-identical copies. Test traffic and generated images are synthetic.

The initial checkpoint is Imagine commit `491e07d`. Port/integration changes after it are uncommitted. No push or service migration was performed.

## Approved profile migration

The user subsequently replaced the workflow-based MCP contract with Imagine-owned
YAML profiles and client overrides. See [profile API](../profiles/README.md).
MCP now exposes `list_profiles`/`describe_profile` and accepts `profile` rather
than `workflow_id`; Turbo is selected by `quick-draft`. Graph translation is no
longer on the runtime path. The old graphs, manifests and their differential
checks remain historical fixtures, not configuration. Arbitrary graph translation
is outside the requested scope and is not a parity defect. General Comfy proxy
forwarding remains independent.

The original counts/GPU evidence below predate this migration. They do not claim
byte-identical MCP metadata for the new contract. Current profile tests cover
recursive overrides, list replacement, invalid requests, atomic rejection,
reference sizing/ownership, stored profile version/hash, seed retries, and MCP
through the real Imagine HTTP API with a fake engine. Native settings/binaries
are unchanged. See the current run evidence in `profiles/VALIDATION.md`.

## Implementation map

| Contract | Reference owner | Rust owner | Evidence |
|---|---|---|---|
| CLI, env, defaults, validation, aliases, check-backend | cli.py, config.py | config.rs, main.rs | differential config cases; actual CLI/start-script tests |
| Kobold legacy and Router lifecycle, default-model alias, cache/bootstrap | llm/koboldcpp.py, kobold_router.py, scripts/kobold_bootstrap.py | backend.rs, requests.rs, st-kobold-bootstrap.rs | HTTP and multi adapter comparisons; stack fixtures |
| TabbyAPI model snapshot, SSE load, unload, context, VRAM classification | llm/tabbyapi.py, tabby_thinking.py | backend.rs, requests.rs | HTTP lifecycle; 486 thinking/rewrite cases; guarded multi-model switches |
| Ollama load/unload and metadata lease behavior | llm/ollama.py | backend.rs, service.rs | HTTP lifecycle and logs |
| Prism and four Strata adapters, catalogs, passive metadata, model selection | llm/prism.py, llm/strata.py, llm/multi.py | backend.rs, service.rs | eight single-backend choices plus seven-adapter multi tests |
| Model profiles, duplicate-key rejection, aliases, presets | model_profiles.py | profiles.rs, json_wire.rs | 102 profile comparisons; raw Tabby/Prism wire bodies |
| FIFO, 202 workflow lifetime, cleanup, retry state, ownership | coordinator.py, comfy.py | coordinator.rs, comfy.rs | HTTP/multi/failure comparisons and guarded Imagine integration |
| Route policy and aliases | comfy_routes.py | routes.rs + reference/comfy-routes.json | route Cartesian-product differential; encoded route regressions |
| HTTP streaming, headers, request body, disconnect drain | http.py, service.py | http.rs, service.rs | all-adapter HTTP tests; 10 gateway edge cases |
| WebSocket text/binary, subprotocol, counters, heartbeat, close | http.py | http.rs | echo/binary/subprotocol/close/counter comparisons; handoff test |
| Logging, Catppuccin, TTY, NO_COLOR, TERM and Strata filter | logging.py, strata_logging.py | logging.rs, request_log.rs, st-strata-logging.rs | exact formatter bytes, causal HTTP log streams, 12 PTY/redirect cases |
| Prism diagnostic summaries | llm/prism_diagnostics.py | diagnostics.rs | 24 synthetic comparisons; no prompt content logged |
| Six MCP profile tools, process-owned jobs/retry | approved profile migration | mcp.rs, mcp_protocol.rs, root profiles.rs | current profile integration; earlier protocol evidence below |
| YAML profile defaults/overrides, seeds, reference registration | approved profile migration | root profiles.rs, references.rs | API/MCP profile tests; legacy workflow normalization is fixture-only |
| General Comfy proxy retained | service.py | service.rs | original forwarding remains independent of MCP Imagine execution |
| Shared GPU scheduling for direct Imagine and MCP | approved integration | coordination.rs, scheduler.rs, api.rs | direct jobs, queue cancellation, consecutive batch, LLM restoration, coordinator loss/physical stop |
| Qwen Turbo and workflow sampling/LoRAs | approved Imagine extension | contract.rs, engine.rs, workflow_image.rs, native/workflow | real standard/Turbo/LoRA/upscale runs; frozen v2 pixel regression |
| AIO color/tone matching | services/color_match.py, tone_match.py | color_match.rs | 15 CPU fixtures, maximum 1 level difference in 8-bit output |
| Stack start/stop, menus, colors, child process groups | st-stack.zsh | proxy/st-stack.zsh | unchanged reference expectations with paths redirected to fixture-local Rust helpers |

Local consolidated logs and counts: `.imagine/test-runs/rust-proxy-validation-20261015/summary.json`.

## Synthetic validation

- `cargo test --workspace --offline`: 29 tests passed, including 7 API lifecycle tests and the shared coordinator integration. Unit/UI tests are not a browser acceptance run.
- Differential oracle: 3,398 exact cases (routes, formatting, workflow/config/profile normalization and request rewrites).
- Single-adapter HTTP/log comparisons: KoboldCpp, TabbyAPI, Ollama, Prism, SC117, Orca, SC117 128k and Orca 128k. Tests cover passive metadata, generic forwarding, WS, 202 lifetime, streaming disconnect drain and lifecycle order.
- Multi: 31 results/operation traces, all seven adapters with fixtures asserting exclusive GPU ownership; a disconnected waiting client does not switch models.
- Failure comparisons: failed load HTTP, false confirmation, load timeout, unload failure, image timeout. Both implementations retain blocked ownership after an unconfirmed Strata transition; they do not infer safety from an HTTP success alone.
- Recovery: Tabby insufficient-VRAM failures and active-request retry (passive metadata does not retry); Kobold Router generation failure, failed unload, recovery and confirmed handoff. Responses, status and lifecycle traces match.
- MCP: initialize-era versions plus 2026-07-28 per-request envelopes, `server/discover`, tools, HTTP security/error behavior. Both transport generations retain SSE keepalives; process-owned jobs survive a transport disconnect.
- Gateway (10 cases): raw percent-encoded URL/query, GET request body, duplicate headers, Origin/Referer/Location, encoded lifecycle/GPU routes, WS binary/subprotocol/counter/close, 64 HTTP and 32 WS connection limits.
- Stack: all 37 original behavior tests passed after redirecting executable paths and adapting only the filter process-name assertion (`st_proxy.strata_logging` → `st-strata-logging`). An actual stale process-name check in the launcher was found and fixed.
- Source/patch restoration and hashes, fmt, clippy and shell syntax are checked separately from model behavior.

## Actual model/GPU evidence

Only Imagine used real model/GPU execution. LLM and Comfy controllers in these runs were disposable fakes. No real KoboldCpp, TabbyAPI, Ollama, Prism, Strata or Comfy model run is claimed.

| Run | Result | Local evidence (ignored) |
|---|---|---|
| Frozen binary + Turbo, before compatibility fix | Standard succeeded; Turbo failed F32 assertion | `.imagine/test-runs/rust-proxy-live-hdyf4bcc/evidence.json` |
| Separate workflow build, MCP standard | 256px, 11.090 s | `.imagine/test-runs/rust-proxy-live-u4rfl2z6/evidence.json` |
| Same run, Turbo quick draft | 256px, 12.681 s, seed `9007199254740991` | same |
| Same run, explicit NSFW LoRA switch | Synthetic teapot, 9.031 s | same |
| Same run, reference upscale pipeline | 256→256, 22.514 s; pipeline only, not enlargement | same |
| Actual 2× upscale | 256→512, 27.764 s | `.imagine/test-runs/rust-proxy-live-np_z91_a/evidence.json` |
| Frozen default v2, unload/resume, reference editing, cancellation | Passed; default pixel hash unchanged before/after resume | `.imagine/test-runs/native-api-h1eeh__5/results.json` |

Frozen 1600×896 RGBA hash: `473bfa4ab0c2230258a168afbe07523bd6bc7d81a4a2daf458500b6df036132d`. Baseline generations took 41.546 s and 41.316 s; reference edit 47.077 s. These are smoke timings, not performance comparisons across Comfy and Imagine.

Frozen v2 binary SHA256: `9fd49b0e5cb3426043ebd85271ad167555551ac65252590cde5ac25bed44bbb0`.
Separate workflow binary SHA256: `078c02af85d8b71cac35db3d6af6d722d30f00e2ec22cd88f1ff40545db847d5`.
Compatibility patch SHA256: `d8fbe74f8a57b27874268a88b732c6299919b7a0431cd6deb20456a0df38445f`.

The failed Turbo run is retained as evidence. Only the separate compatibility build fixes it. The default executable, baseline artifacts, base native patches and accepted inference profile were preserved.

## Remaining differences and validation gaps

This is not a claim of exhaustive, byte-for-byte parity across every input and failure schedule.

- Native inference is the approved Imagine adaptation. Its pixels need not match ComfyUI. CPU color/tone matching is within one 8-bit level in tested cases, not bit-identical. PNG compression differs; reference resize/JPEG decoder edge cases have not been exhaustively compared.
- MCP intentionally uses Imagine profiles instead of graphs. Supported overrides are constrained by Imagine capabilities; there is no graph translator or Comfy fallback in the generation path.
- MCP queue memory accounting uses Imagine API request sizes, rather than serialized Comfy graphs plus embedded reference bytes. Limits and atomic batch admission exist, but the exact byte threshold differs.
- MCP JSON-RPC/typed validation covers tested contracts; all malformed nested envelopes, installed-SDK extension methods and exact Pydantic/argparse diagnostic strings are not exhaustively reproduced. CLI supports unambiguous abbreviations, but diagnostic wording/order and usage banners can differ.
- Connection budgets now match the reference (control 16, chat 64, images 64, WS 32). Every truncated upstream stream timing and heartbeat deadline timing has not been exhaustively compared.
- Real LLM/Comfy adapter acceptance remains unrun. `tests/live_stack.py` is prepared for the explicitly selected stack. No browser/manual acceptance or desktop control was used.
- Escalated commands were used for disposable local sockets, native builds and GPU runs. One automatic approval review timed out before a local test command ran. The explicitly permitted single retry was approved and the test passed; no action remains blocked.

These gaps must remain visible before calling the full original parity assignment complete.

## Launcher and Imagine integration corrections

The root launcher now applies CLI options through the shared Rust option resolver
and starts the selected stack by default. `--proxy-only` is explicit. Original MCP
input/output roots are retained for Pi. Its `qwen-scene` instructions use profiles.
The root launcher defaults `ST_PROXY_COMFY_ENABLED=false`; disabled Comfy routes
fail closed while Imagine/LLM coordination remains active. Direct binary defaults
retain Comfy support for original-proxy compatibility.
MCP retries ambiguous submissions using their original idempotency key and retains
jobs during status/result outages instead of marking an unknown outcome failed.

Validation for these corrections: 39 workspace Rust tests passed, including real
API/proxy coordination with a fake inference engine and no Comfy server. The MCP
regression covers ambiguous submission, status outages, result download recovery,
identical idempotency keys, and a concurrent retry of the same job. Also passed:
3,398 differential cases, four root-launcher checks, 17 Pi launcher tests,
workspace Clippy, formatting, shell syntax, source hashes and release build.
An actual release-launcher dispatch was checked with the supervisor replaced by
a capturing fixture. No user services, real Pi session or GPU inference were run.
