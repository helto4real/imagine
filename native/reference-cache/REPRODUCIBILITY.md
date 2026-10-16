# Retained reference optimization material

The accepted reference baseline is the separately pinned hybrid/prefetch
executable. Keep frozen v1 and v2 unchanged, including their manifests and
source/evidence files. Text-only Qwen uses v2; reference passes use the accepted
reference binary, with v2 as explicit or verification fallback. Future models
require their own validation.

## Source and binaries

Tracked source is the published base plus the five patches and chained source
locks in this directory. Restore them in the order documented in [README.md](README.md).
The active local executable is `native/reference-cache/bin/sd-server`, pinned by
`reuse-budget-source-lock.json`. The ignored development checkout, its working
build, model files and the separate Turbo/workflow executable are retained.

The following earlier executables remain as local comparison controls under
`test-generations`:

- `qwen-reference-ram-cache-implementation/sd-server`
- `qwen-reference-hybrid-cache-implementation/sd-server`
- `qwen-reference-prefetch-implementation/sd-server`
- `native-upstream-d637cfb-1a9564/build/bin/sd-server`

These are comparison tools, not selectable production defaults. Frozen v1/v2
stay under `experiments/qwen-image21-sage/baselines/`. Binaries and runtime test
evidence are deliberately excluded from Git.

## Local reproduction kit

`test-generations/optimization-kit/` contains the retained material on this
machine. Its `retained-controls.json` pins 18 reference/control PNGs and the
retained executables. It preserves:

- The accepted identity sheet and the earlier portrait reference.
- One complete text-only sheet result.
- ComfyUI comparison inputs and results.
- CFG 1, CFG 2.5, text-only and ordinary multi-reference v2 control results.
- Scoped result metadata, numeric diagnostics and historical trial summaries,
  including rejected candidates; duplicate result images are removed.
- The final native synthetic test sources and the helper/link information used
  to compile them, plus archived implementation/acceptance helper scripts.

The maintained local entry points are:

```sh
python3 test-generations/optimization-kit/verify-source.py
# Only with authorization for live GPU tests and their scoped data:
python3 test-generations/optimization-kit/acceptance.py
python3 test-generations/optimization-kit/selection-smoke.py
```

The verifier restores source into a new temporary directory and checks the
patch chain and final hashes. The live harnesses use isolated test data and
require the existing release API, accepted native executables and model files.
They do not depend on old `/tmp` helper files. Their fixed settings and hashes
represent the accepted workload, not a general benchmark configuration API.
The acceptance harness stabilizes free GPU memory before native startup; this
benchmark-only wait contributes to its total job timings. Compare sampling
separately and retain the startup budget when interpreting results.

No new model run is authorized merely by these instructions. Synthetic and live
results remain separate, and data access must stay within the approved test scope.

## Cleanup boundary

After committing the source and documentation, remove redundant candidate
executables, copied API binaries, test credentials/session directories, object
files, synthetic test executables, Triton caches, temporary upstream source/build
copies and non-control generated images. Keep compact historical evidence and
the controls above. Original metadata may name deleted duplicate output paths;
use the retained control images and their hashes for future comparisons.

`optimization-kit/cleanup-plan.json` records the exact local file list;
`cleanup-report.json` records the applied cleanup and integrity checks. Temporary
helpers belonging to this cache implementation are archived before removal.
Other project work, UI prototypes, production/session data and model files are
outside this cleanup.
