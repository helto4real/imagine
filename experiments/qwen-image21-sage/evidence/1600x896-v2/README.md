# Accepted 1600×896 v2 evidence

Accepted 2026-10-15. The exact dynamic-block executable and environment from the
completed eight-image experiment were copied into `baselines/1600x896-v2/`.
No GPU rerun was performed solely to freeze unchanged validated inputs.

`accepted-plan.json` pins the original 1600 candidate command, environment,
prompt hashes and request; only binary/model path strings are normalized.
`acceptance-results.json` retains the original comparison, including the failed
4096 auto-tiling identity test. It is not a claim of 4096 acceptance.
`native-restoration.json` verifies the complete native patch restores identical
source files. The v1 manifest, source evidence files and executable are intact.

The current launcher and Rust service select v2. Default engine settings,
precision, API revision and request validation remain separate and unchanged.
The algorithm is dynamic; the profile name records its accepted validation scope.
Frozen binaries are local ignored artifacts and are never added to Git.
