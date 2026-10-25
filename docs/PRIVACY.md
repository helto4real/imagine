# Session privacy

Start Imagine with `./start-server.sh`. There is no password or YubiKey login,
Python authentication helper, or application-level media encryption. API, UI and the proxy
connect without Imagine tokens; all local clients share jobs/assets and runtime
controls. Loopback binding remains required. `access.json` is no longer created
or read; an existing file is left unused. Separate MCP endpoint authentication
and Helto profile decryption are unchanged.
`--no-encryption` is accepted as a compatibility no-op; plain storage is always used.
The removed `--unlock` and `--enroll-yubikey` options are no longer supported.

Existing encrypted profile templates still use their separate Helto unlock
mechanism. The old Imagine `.imagine/privacy-keystore.json` and `.venv-privacy`,
if present, are unused and are not read or removed by the server.

## Data lifetime

- Jobs, submitted prompts, effective parameters, idempotency records, queue state,
  operation history and asset metadata exist only in a SQLite **memory** database.
  SQLite temporary storage is also memory-only. Nothing resumes after restart.
- Internal uploaded/generated images are ordinary files, created atomically with
  mode 0600 inside a session directory with mode 0700.
- Startup deletes legacy `imagine.sqlite` and its journal/WAL files and the
  internal `media` directory, without opening their contents. This happens
  before HTTP serving starts.
- Ctrl-C/SIGTERM stops the worker and HTTP service, closes the memory database
  and removes the session's internal media. Core dumps are disabled for Imagine,
  its child processes and the proxy.
- Submitted prompt text is not written
  to ordinary logs or persistent job history. Prompt-based troubleshooting can
  be handled explicitly by the client.

Agent files live in workspaces under `/run/media/veracrypt1/mcp-root`, one
folder per workspace in its creation date folder, on the user's VeraCrypt volume
(see [MCP-WORKSPACES-DESIGN.md](MCP-WORKSPACES-DESIGN.md)). Agents never see
these paths: files move through MCP tools and the proxy's HTTP file routes, and a
workspace is reached only with its key, stored in plain text in the folder's
`AGENT-KEY.txt` and as a SHA-256 in `workspace.json`. Workspace files are not
disposable. Local agents' sandboxes deny reads of the whole encrypted volume.

The proxy monitors the Imagine session once per second. Losing
that session, or seeing a new session ID, stops old MCP tasks and removes their
history, reference previews and unconfirmed exports. Requests fail closed while
Imagine is unavailable. MCP API requests carry `X-Imagine-Session`; the server
rejects requests from an old session before they can mutate the new session. Network probes time out after two seconds; an in-flight
admission operation can delay cleanup by at most five additional seconds, plus
local filesystem work. Proxy shutdown also clears the entire staging area.

Agent workspaces under `mcp-root`, older agent folders under `mcp-inputs`, images copied to client project directories,
UI-saved images (including `.imagine/images`), UI preferences, profiles, unused legacy API
credential files, native builds and baseline evidence are not disposable history.
They are not removed or encrypted by this lifecycle.

Deletion unlinks files; it is not a secure overwrite of SSDs, snapshots or
backups. SIGKILL/power loss cannot run shutdown cleanup: memory is lost, internal
media files remain readable until deleted on the next Imagine start, and MCP
exports are deleted by the live proxy's monitor or on its next start. OS swap,
user-directed terminal log capture and already copied files are outside this
application's session cleanup.

## Verification

Automated checks use disposable stores and synthetic prompts/images. They cover
plain media, token-free local access, restart isolation, cleanup and MCP
session loss.

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo build --bin imagine-server
python3 scripts/test_server_privacy.py
```

The binary test launches the actual server in temporary projects with closed
stdin, verifies startup without terminal login, plain media, token-free API access, startup purge, SIGTERM cleanup and restart isolation. It also
checks the legacy `--no-encryption` flag. It never starts a model or accesses
real credentials/history.
