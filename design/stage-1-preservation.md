# Stage 1 preservation before Stage 2

Date: 2026-10-18. Tomas requested a commit of the inference implementation and
reusable integration evidence before starting Qwen Image 2.1 Stage 2.

The commit includes `inference/`, the parity design, Stage 0 reference closure,
all Stage 1 reports/checkpoints, the approved-decision ledger and the compressed
synthetic oracle bundle. Unrelated existing API/UI/native/profile work and UI
prototypes are not part of this commit. No model weights, compiler distributions,
build caches or production/runtime data are added to Git.

The saved suite contains 64,413 physical records and 20 source/compiler oracle
families. The manifest preserves 19,204 files with member and archive SHA-256
checks. Three compressed parts preserve the original expected bytes without
requiring the original `/tmp` directory. Direct AOT/compiled cubin assets are
synthetic regression fixtures, not a production inference fallback.

The native kernels and original 286 implementation/tool inputs remain unchanged
from the completed 155-test /146 Python-free checkpoint. That successful full
validation is reused for unchanged inference inputs. New preservation-tool checks
cover integrity, repeatable restore, refusing different existing files, path and
symlink escapes, missing members, corrupt member content and split gzip streams.
Archive streaming verification is recorded separately before the commit.

Future upgrades follow [the bundle guide](../inference/fixtures/README.md): retain
old evidence, inventory the actual source delta and add source-backed new cases.
No numerical tolerance, model/backend substitution, autotuning change or new
parity exception is adopted by preserving these artifacts.

Preservation checks passed: all 19,204 archived members verified, seven archive
unit tests passed, both new shell launchers passed `bash -n`, and all original
286 source/tool hashes still match the executed checkpoint. The exact staged
scope was reviewed before committing; unrelated dirty files are excluded.

`git diff --cached --check` reports four pre-existing formatting observations:
two Markdown hard-line-break spaces in the parity design, a literal context
line in the historical integration patch, and one blank line in the frozen
NVFP4 reference tool. These bytes are retained to preserve the recorded source
identity and patch content. This check is not reported as passing; no inference,
archive integrity, numerical or typing check is waived or weakened.
