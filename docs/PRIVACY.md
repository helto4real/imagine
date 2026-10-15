# Session privacy

Start Imagine in a terminal with `./start-server.sh`. First start asks you to
create a password; subsequent starts ask you to unlock. No HTTP listener or
native engine starts before a successful unlock. API bearer credentials remain
required after unlock, so the UI and MCP continue using their existing tokens.

On a new installation, run `./setup-privacy.sh` once. This creates
`.venv-privacy` and installs the pinned Helto FIDO2 provider plus its dependencies.
`IMAGINE_PRIVACY_PYTHON` can select another Python environment with those packages.

```sh
./start-server.sh                     # password setup/unlock, then serve
./start-server.sh --enroll-yubikey     # unlock, register key, exit
./start-server.sh --unlock yubikey    # PIN + touch, then serve
./start-server.sh --no-encryption     # wipe old history, skip unlock, then serve
```

YubiKey enrollment preserves the password slot. Either password alone or the
enrolled YubiKey with its PIN and touch can unlock. No password is required for
subsequent YubiKey unlock. Enrollment needs two touches and a FIDO2 key supporting
hmac-secret, credential protection and a configured PIN. There are no automatic
PIN retries. Enrollment or replacing a key requires unlocking the existing store.
Passwords and PINs are entered through the terminal with echo disabled; they are
never command arguments, environment variables, log entries or session files.

The dedicated `.imagine/privacy-keystore.json` contains a random authentication
verifier wrapped independently with AES-256-GCM by scrypt(password) and the
verified FIDO2 hmac-secret. Password scrypt uses N=131072, r=8, p=1 and a random
16-byte salt. The store is written atomically with mode 0600. The shared Helto
keystore is not changed. Existing encrypted profile templates still use their
separate Helto unlock mechanism.

## Data lifetime

- Jobs, submitted prompts, effective parameters, idempotency records, queue state,
  operation history and asset metadata exist only in a SQLite **memory** database.
  SQLite temporary storage is also memory-only. Nothing resumes after restart.
- Internal uploaded/generated images use AES-256-GCM, with a random nonce for
  each file and the asset ID as authenticated data. A fresh random key exists
  only in the server process for each session; it is never saved or exported.
  Encrypted files from a crashed session cannot be recovered on restart.
- Startup deletes legacy `imagine.sqlite` and its journal/WAL files and the
  internal `media` directory, without opening their contents. This happens
  before unlock and before either encrypted or unencrypted serving starts.
- Ctrl-C/SIGTERM stops the worker and HTTP service, closes the memory database
  and removes the session's internal media. Core dumps are disabled for Imagine,
  its child processes and the proxy.
- `--no-encryption` starts an empty session without password/YubiKey and stores
  internal media as ordinary files. The same shutdown/startup deletion applies.
  The credential store and API tokens are retained in both modes.

MCP export files must remain readable by the client for byte-size/SHA-256 checks
and copying. They stay at `/run/media/veracrypt1/mcp-images`, on the user's
VeraCrypt volume. **Imagine does not encrypt MCP exports itself.** Use an encrypted
volume for that path (and for any configured replacement). The output directory
is exclusively disposable: startup removes all previous contents, including
registered reference previews. A lock prevents two proxies from sharing it.

The proxy monitors the authenticated Imagine session once per second. Losing
that session, or seeing a new session ID, stops old MCP tasks and removes their
history, reference previews and unconfirmed exports. Requests fail closed while
Imagine is unavailable. MCP API requests carry `X-Imagine-Session`; the server
rejects requests from an old session before they can mutate the new session. Network probes time out after two seconds; an in-flight
admission operation can delay cleanup by at most five additional seconds, plus
local filesystem work. `confirm_saved` still removes confirmed staging files
immediately. Proxy shutdown also clears the entire staging area.

Original files under `mcp-inputs`, images copied to client project directories,
UI-saved images (including `.imagine/images`), UI preferences, profiles, API
credentials, native builds and baseline evidence are not disposable history.
They are not removed or encrypted by this lifecycle.

Deletion unlinks files; it is not a secure overwrite of SSDs, snapshots or
backups. SIGKILL/power loss cannot run shutdown cleanup: memory and its encryption
key are lost, internal remnants are deleted on the next Imagine start, and MCP
exports are deleted by the live proxy's monitor or on its next start. OS swap,
user-directed terminal log capture and already copied files are outside this
application's encryption. Use encrypted swap/storage for those layers.

## Verification

Automated checks use disposable stores, synthetic prompts/images/passwords and
a fake FIDO provider. They cover independent password/key unlock, wrong credentials,
tampering, encryption, ownership, restart isolation, cleanup and MCP session loss.
Real YubiKey enrollment/touch requires the user's own interactive acceptance test.

Reproduce the synthetic checks:

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo build --bin imagine-server
.venv-privacy/bin/python -m unittest discover -s scripts -p test_privacy_unlock.py
.venv-privacy/bin/python scripts/test_server_privacy.py
```

The terminal test launches the actual binary in temporary projects, enters only
synthetic passwords through a test PTY, and checks the pre-unlock API gate,
wrong-password rejection, encryption, `--no-encryption` startup purge and SIGTERM
cleanup. It never starts a model or accesses the real keystore/history.
