# Frozen-baseline portrait-to-environment editing smoke test

Date: 2026-10-14. User requested a generated portrait of a woman followed by an
environmental image using that portrait as the image reference, with a visual
likeness comparison. This is a new workload, not a change to the frozen profile.

## Configuration and execution

- Exact frozen `qwen-image21-1600x896-v1` executable, SHA-256
  `4e33830f7eba36573cba9785249641ae8c727be5e6a506e9c7ef0ee74beea2e3`.
- All 28 frozen environment switches unchanged; same selected model files.
- Both requests: 1600×896, 40 ER-SDE/beta steps, CFG 2.5, seed 42, CPU RNG,
  batch 1, Sage2++, 11 CPU threads, auto-fit, no LoRA or second pass.
- Same resident native server, loopback only. New prompts; original frozen
  negative prompt retained. No ComfyUI or external image service involved.
- First request: photorealistic head-and-shoulders portrait of an adult woman
  with copper-auburn hair, green eyes, freckles, gold hoops, ivory turtleneck
  and a forest-green coat, against a plain studio background.
- Second request: preserve the woman and clothing, place her waist-up in a
  Scandinavian autumn harbor with red timber houses and fishing boats.
  The exact first PNG was sent unmodified in `ref_images[0]`. No `init_image`.
- Native reference-encoding defaults. The VAE reference remained 1600×896;
  the reference image was not externally resized or cropped.
- PNG metadata disabled. Only synthetic prompts and generated images used.

The reproducible request code and complete prompts are in
[`test-reference-edit.py`](test-reference-edit.py). Run from the project root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 experiments/qwen-image21-sage/profiling/test-reference-edit.py
```

The script pauses after the portrait for inspection; press Enter to submit the
reference edit. It verifies frozen artifacts through the baseline launcher and
stops only its own server and GPU monitor on completion or failure.

## Measured results

| Measurement | Portrait, text-to-image | Environment, image reference |
| --- | ---: | ---: |
| Request to saved PNG | 40.569 s | 138.934 s |
| Sampling | 36.380 s | 132.460 s |
| Conditioning | 1.900 s | 3.480 s |
| Reference VAE encode | Not applicable | 0.810 s |
| Final VAE decode | 1.960 s | 1.610 s |

These are one request each, not benchmark averages or a matched edit/non-edit
performance study. Timers exclude initial process startup and the manual pause;
startup to API readiness was 0.407 s and does not mean all model weights were
already GPU-resident. Request time includes required loading, encoding,
sampling, HTTP polling/base64 decoding and closed PNG write, without fsync.

Editing was confirmed by the native `EDIT mode` message, reference VAE encoding,
29 Qwen3-VL vision-segment calls and matching SHA-256 of the supplied PNG.
The two BF16 prefix caches contained 5,801 and 5,663 tokens. All 80 diffusion
forwards ran across 34 segments (2,720 segment executions). This differs from
the single-segment text-only workload. No further performance attribution or
memory-policy optimization was attempted.

Sampled whole-GPU peak over the test was **15,443 MiB**, including desktop use.
No logged warnings/errors, VAE spatial-tiling retry or observed prefix-cache
fallback. After shutdown, GPU use returned to the initial 1,351 MiB. GPU access
required sandbox escalation; no unrelated process was stopped.

## Visual comparison

The two outputs show very strong visual consistency for the synthetic woman:
face outline, eyebrows, nose, mouth/smile, freckle distribution and auburn
hairstyle are closely preserved. Gold hoop earrings and clothing also transfer.
The requested harbor, wider framing and outdoor lighting are present.

The head angle and expression remain close to the portrait, so this is an
easier likeness test than a large viewpoint or expression change. Fine facial
detail has less output area in the environment image. This is a qualitative
inspection, not an embedding-based identity score or proof of universal editing
quality. It does not establish bit identity against ComfyUI or a control engine.

## Local artifacts and hashes

Ignored output directory:
`experiments/qwen-image21-sage/runs/20261014T182206Z-reference-edit-onplx_2o/`.
Contains `portrait.png`, `environment.png`, exact prompt files, plan, timings,
validation, local server log and sampled GPU usage. Images are not published.

- Portrait decoded RGBA SHA-256:
  `028657236347b30632225e17ad647a3209a1f62207912fcec4279be3321e3495`.
- Environment decoded RGBA SHA-256:
  `e7ee0c997f0cf962c9987c7b434fe3f920995b636277d421ddadaacef965ca95`.
- Supplied reference PNG file SHA-256:
  `bbae7af37d3cc0920bd3ab149ac50c124b514352fb12c9a82fb21b671b0c7fa1`.
- Alpha extrema: portrait 254–255; environment 251–255. Images were inspected
  and passed to the model as generated, without postprocessing.

Both generations completed; the frozen executable hash was checked again
afterward. No native source, frozen settings or baseline prompts were changed.

## Follow-up: 1-megapixel reference budget

User requested the same test with a smaller reference. The only request-plan
change was `ref_image_args="vae_input_max_pixels=1000000"` on the environment
request. The portrait was regenerated first in the same two-request sequence;
its decoded RGBA and PNG file hashes were identical to the original. All prompts,
28 runtime flags, models, seed, sampling settings and output dimensions matched.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 experiments/qwen-image21-sage/profiling/test-reference-edit.py --reference-max-pixels 1000000
```

The original PNG was supplied unchanged, then resized by the native pipeline
before VAE encoding. Qwen 2.1 passes that resized reference onward to its
conditioner too. This avoids an external downscale being undone by default
reference auto-resizing. Native independent rounding to multiples of 32 gave
**1344×736 = 989,184 pixels**. Its aspect ratio differs slightly from the original
(1.826 versus 1.786); there was no external crop or resampling-filter change.

| Measurement | Original 1600×896 reference | ~1 MP reference, 1344×736 |
| --- | ---: | ---: |
| Portrait request to PNG | 40.569 s | 40.457 s |
| Environment request to PNG | 138.934 s | **47.401 s** |
| Environment sampling | 132.460 s | **42.330 s** |
| Conditioning | 3.480 s | 2.240 s |
| Reference VAE encode | 0.810 s | 0.740 s |
| Final VAE decode | 1.610 s | 1.640 s |
| BF16 prefix tokens, positive / negative | 5801 / 5663 | 4065 / 3927 |
| Diffusion forwards | All 80 at 34 segments | One at 34 segments, 79 at 1 segment |
| Sampled whole-GPU peak over both requests | 15,443 MiB | 15,269 MiB |

The environment request was **91.533 s shorter (65.88%, a 2.93× time ratio)**.
This is a single follow-up versus the preceding run, not repeated paired timing.
The large reduction coincides with the smaller prefix and mostly single-segment
execution; this test does not separately attribute kernel versus transfer time.
No warnings/errors or observed VAE-tiling/prefix fallback were logged.

Visual inspection still shows very strong likeness to the portrait: hair,
eyebrows, eyes, nose, mouth, freckles, earrings and clothes remain recognizable.
There are small changes to facial rendering/framing and harbor details relative
to the larger-reference result. No obvious likeness degradation was visible in
this pair, but the output pixels differ and no quantitative identity metric was
used. This does not establish equivalent quality for other faces or viewpoints.

Local follow-up artifacts:
`experiments/qwen-image21-sage/runs/20261014T182855Z-reference-edit-ht_ass9o/`.
`comparison.json` records the matching portrait/reference hashes, timing ratios,
segment counts, prefix sizes and sampled peak. Environment RGBA SHA-256:
`0b722072c8e4a21ac1691289d30481cf73eea39a98ab29cc46bcaf7f9b87fd6f`.
The server and GPU monitor stopped normally. The frozen baseline was not changed.
