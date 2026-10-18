# Rust server integration evidence

These files contain explicitly requested isolated development checks and
synthetic inputs only. They contain no ordinary Imagine/Dreamchat runtime data.
Read [the integration report](../rust-backend-server-integration.md) for scope.

- `checks/`: full 150-pass workspace run with two test threads, two deliberately
  ignored live tests, workspace/inference Clippy, release/worker builds and two
  binary privacy tests, plus the final three pipe tests. The failed parallel clipboard run and separate passing
  rerun are retained; the unrelated test/source was not changed.
- `moss/`: trained HTTP/model-library/Rust-worker comparison against the retained
  original 320×224 MOSS control. Original results remain in
  `../reference-20261018/validation/moss-runs/small-python/`.
- `qwen-int8/` and `qwen-turbo/`: trained HTTP/model-selection results, generated
  PNGs and matching full original-source oracle results/import provenance.
  All decoded output pixels are exact. These controls do not establish other
  dimensions, memory-pressure cases, model variants or timings as benchmarks.
- `qwen-t2i/` and `qwen-empty-negative/`: 256×256 text-to-image controls with
  nonempty and empty negative prompts. The latter validates the unconditional
  source blank-space sentinel and no-reference keep-vision policy.
- `deployment-manifest.json` retains the earlier trained worker deployment;
  `deployment-current.json` identifies the final worker, whose only subsequent
  changes aligned minimum dimensions and no-reference tokenizer options with
  the existing API/original node. The final empty-negative test records that
  worker's binary SHA256 explicitly.
- `source-inputs.json`: hashes of the affected implementation, profiles, tools
  and required reference/runtime metadata at this checkpoint.
- `artifacts.json`: hashes of retained evidence. Large tensors, model weights,
  executables, caches and temporary projects are retained only in the ignored
  `comfyui/moss-20261018` local build root.

New source controls `qwen-server-int8.json` and `qwen-server-turbo.json` use
224×320 synthetic RGBA references with no resize (resolution zero), independent
320×224 output, seed 3061383787 and source-matched policies. INT8 uses four
ER-SDE/beta steps, CFG 2.5 and one LoRA; Turbo keeps eight explicit Euler sigmas
and CFG 1. The MOSS HTTP test uses four Euler/normal steps and CFG 3.5; it does
not change the shipped MOSS 50-step/CFG 3 defaults.

`inference/tools/check_server_backend.py --help` documents the explicit runner.
It requires a new own-root path and runs the actual default server with an
explicit prepared Rust runtime manifest. It does not restart the normal server.
Its final MOSS/INT8/text-to-image runs also assert that the worker dies and session media is
removed after stopping the isolated server. Synthetic pipe tests separately
cover cancellation and failure/reaping. Browser/native-window manual checks
were not run.
