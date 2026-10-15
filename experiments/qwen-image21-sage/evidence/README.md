# Preserved measurement evidence

`1600x896-v1/` preserves the accepted coordinator results, four three-image
series, request plans, pixel hashes and complete-VAE allocation validation.
These are historical measurements, not new results from repository cleanup.
Personal filesystem prefixes were removed; no numerical results were changed.

Referenced raw images, logs, traces and redundant executables were deleted after
preserving the independent frozen binary locally. Historical run paths identify
their origin; files at those paths are no longer required by the baseline
launcher or offline checker. The full experiment chronology and reasons for
rejecting alternatives remain in the reports and `TEST-DECISIONS.md`.

`vae-stage2-strips/` preserves the subsequent 800×448 convolution experiment:
its isolated patch and source hashes, numerical/memory gates, four session
plans, twelve image results and validation. It is a tested optional candidate,
not a modification of frozen v1. No raw images, prompts or device logs are
included in this evidence directory.

`vae-stage2-dynamic/` preserves the subsequent dynamic dispatch patch, binary/source hashes, request plans and eight-image results at 1600 and 4096. It is not a new frozen baseline.

`comfy4096-reference/` contains the subsequent two-image headless AIO/Kitchen reference at 4096×2304, with sanitized settings, coarse timings, dispatch validation and source hashes.

- `vae-dynamic-blocks/`: experimental broader dynamic whole-block and
  upscale/convolution strips, portable delta patch and restoration hashes,
  full-graph memory, 13 synthetic identity checks and eight image results.
  Exact at 1600; 4096 differs from v1 with larger outer tiles. Not promoted.

- `1600x896-v2/`: accepted dynamic-strip freeze, exact command/environment,
  original acceptance results and complete native patch restoration checks.
- `vae-fixed-tiles/`: 4096 v1/v2 control with identical 49-tile geometry;
  all four images exact, workspace 3321.56→2476.69 MiB, warm VAE 33.05→32.00 s.
