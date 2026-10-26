# YAML profile validation — 2026-10-15

- `cargo test --workspace --offline`: 35 tests passed. Includes YAML parsing and
  strict defaults, deep overrides, list replacement, invalid/atomic batches,
  profile/version/hash persistence, reference sizing/ownership, the UI API flow,
  MCP profile discovery/description/validation, consecutive shared-FIFO jobs,
  53-bit explicit seeds and random seed retries. A completed MCP retry also
  succeeds without a reachable Imagine server or unlocked private session.
- `cargo clippy --workspace --all-targets --offline -- -D warnings`: passed.
- `cargo fmt --all -- --check`, `git diff --check`: passed.
- `cargo build --release --workspace --offline`: passed.
- Frozen baseline check and separate native workflow source/binary hash check:
  passed; no native files, precision or frozen settings changed in this migration.
- Reference integrity: 76 files / 10 byte-identical copies passed. Migrated
  encrypted prompt strings and prompt guidance match their source values exactly,
  checked without decryption.

Real Imagine GPU smoke through MCP and YAML profiles:

| Case | Size | Elapsed seconds |
|---|---|---|
| qwen standard | 256×256 | 12.211 |
| quick-draft, maximum 53-bit seed | 256×256 | 13.421 |
| quick-draft, explicit NSFW LoRA, synthetic teapot | 256×256 | 8.951 |
| image_upscale, consistency LoRA / texture VAE | 256→512 | 24.865 |

All four succeeded, expose profile version/hash and effective settings, omit
private prompts from MCP results, and attach repeated calls to the same job.
Evidence: `.imagine/test-runs/profile-mcp-live-r4s5cc0y/evidence.json` (ignored).
The disposable profile copies use synthetic prompts; real private defaults were
not decrypted. LLM and Comfy controllers are fakes. No real external LLM/Comfy
model acceptance or interactive UI testing is claimed. The subsequent random
seed range, schema cleanup and offline-retry changes were verified by the final
Rust checks; they do not affect these explicit-seed model runs.

The sandbox initially denied local test sockets. The same test suite was rerun
with approved escalation and passed. GPU/socket tests used escalation; all own
processes were stopped. No existing service was restarted, no Computer Use was
performed, and no changes from this task were committed or pushed.
