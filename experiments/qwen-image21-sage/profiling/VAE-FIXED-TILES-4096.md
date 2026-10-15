# 4096 VAE comparison with identical outer tiles

Requested 2026-10-15 after accepting the dynamic-strip candidate as the new
1600×896 baseline v2. This is a diagnostic, not a new default tile configuration.

## Protocol

Frozen v1 executable versus the exact v2 executable, with every accepted switch,
precision and sampling value retained. Both requests explicitly set:

```json
{"vae_tiling_params":{"enabled":true,"tile_size_w":1024,"tile_size_h":576,"target_overlap":0.5,"rel_size_w":0.0,"rel_size_h":0.0}}
```

4096×2304, frozen synthetic positive/negative prompts, seed 42, 40 ER-SDE/beta
steps, CFG 2.5, CPU RNG, 11 threads, Sage2++, no LoRA/reference/second pass.
Two images per process (initial and warm). Each saved image is compared by RGBA
pixel values, ignoring PNG metadata. Logs must show exactly 49 tiles of
1024×576 and no automatic spatial fallback. Request values and actual executed
geometry are both asserted in the result collector.

The internal v2 strip rows remain dynamic. No fixed outer tile size is added
to the baseline launcher or Rust defaults. The only ordinary profile difference
is `SD_VAE_DYNAMIC_BLOCK_STRIPS=1`. Complete native graph workspace is measured
separately from whole-device occupancy; the actual per-tile VAE arena is the
relevant number in this control. Start-to-PNG includes native HTTP/polling and
closed PNG write, excludes process startup and fsync. Sampling time is reported
separately and must not be attributed to VAE changes.

Only approved synthetic inputs and benchmark logs are read. GPU/loopback access
was escalated for the explicitly authorized experiment. No private ComfyUI data
or user-generated assets are accessed. Native source remains unchanged during
these runs. Native v2 freezes the existing tested binary; no new native build or
kernel/precision change is introduced by this diagnostic.

Coordinator: `runs/20261015T043811Z-fixed-tiles-uswfuyvq`.
Drivers: `run-vae-fixed-tiles.py`, `compare-resident-fixed-tiles.py`,
`collect-vae-fixed-tiles.py`. Raw output remains ignored under `runs/`.

## Results

The initial images use exactly the same 49×1024×576 tile grid, without fallback,
and have identical RGBA pixels, matching the original v1 4096 reference:
`a639cc6b533fbe1db0379892f529f61ac046c99b9a46bdeec3fceff2f0b4f75d`.

| Initial request | v1 | v2 |
| --- | ---: | ---: |
| VAE arena, MiB | 3321.56 | 2476.69 |
| VAE, s | 33.52 | 32.95 |
| Sampling, s | 466.01 | 470.98 |
| Start to saved PNG, s | 503.053 | 507.359 |

The internal strip change preserves this complete image at fixed outer geometry.
Memory is 844.87 MiB lower; timing is close. The initial total is higher due to
sampling variation, and is not a VAE slowdown. The warm repeat also passes; all four images have the same RGBA hash.
The previously observed 49→9 auto-tiling image change is not reproduced when the
outer grid is held constant. This supports the outer-grid explanation for this
prompt/seed/model and does not prove universal bit identity.

## Completed warm comparison

| Warm request | v1 | v2 |
| --- | ---: | ---: |
| Actual VAE arena, MiB | 3321.56 | 2476.69 |
| VAE, s | 33.05 | 32.00 |
| Sampling, s | 463.59 | 474.20 |
| Start to saved PNG, s | 498.072 | 507.611 |
| Prepared GPU weights, MiB | 4787.91 | 6036.94 |
| Weight loading, s | 3.75 | 4.34 |
| Whole-graph diffusion forwards, of 80 | 77 | 77 |
| Final outer tiles | 49 | 49 |

All four saved images are bitidentical, including alpha. Request and executed
grid checks pass for every image; neither engine retries into another grid.
V2 saves **844.87 MiB / 25.4%** of VAE workspace at this geometry. Warm VAE is
1.05 seconds / 3.2% faster in this sample; initial VAE is 0.57 seconds faster.
There is one warm sample per mode in sequential order, so this is not a
statistically established speed improvement. Overall time is higher for v2 in
this series because sampling/weight staging differ; do not call the full-image
time difference a VAE regression or claim improved 4096 weight retention.
The staging difference was not isolated further in this scoped VAE test.

The earlier auto-grid pixel difference disappears when the outer grid is fixed.
For this reference workload, that supports the changed outer tiling as the
explanation. It does not establish identity for every resolution, prompt or
model. The general v2 algorithm stays dynamic; 49 tiles is a diagnostic control,
not a permanent setting or an API restriction. In ordinary auto mode the known
4096 behavior remains 9 larger tiles, with different pixels relative to v1.

1600×896 v2 remains frozen and selected by the launcher/Rust service. V1 is
preserved as control. No user app was restarted, no live API/edit test was rerun,
and no commit/push was made. Rust unit/integration tests (11), Rust formatting,
launcher offline checks, frozen hashes, full native patch restoration, Python
syntax and shell syntax pass. Compact results are in `../evidence/vae-fixed-tiles/`.

Reproduce, after both frozen binaries and the selected model files are available:

```sh
python3 experiments/qwen-image21-sage/profiling/run-vae-fixed-tiles.py --model-root "$QWEN_MODEL_ROOT"
# Pass the resulting coordinator directory explicitly:
python3 experiments/qwen-image21-sage/profiling/collect-vae-fixed-tiles.py "$FIXED_TILE_RUN"
```
