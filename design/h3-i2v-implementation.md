# Minimal H3 I2V integration — 2026-10-20

Base commit: `a1a935817459704505405dfcdac797ca98be397a` (previous signed geometry change).
This increment is uncommitted. No GPU/model run, server/UI start or stop, or push was performed.

## Profiles and verified identity

- `h3-av-noref-v2`: existing text-only Ref2VA behavior; the old fixed `h3-short-av-noref-v1` remains accepted at its original shape.
- `h3-fl2va-i2va-v1`: exactly one owned start image; image + prompt vision tokens and video-VAE latent anchored at frame 0; generated stereo audio.
- Ref2VA DiT SHA-256: `9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779`.
- I2V file: `/mnt/media/gemensam/AI/ComfyUI/models/diffusion_models/minimax_h3/minimaxH3.int8.safetensors`.
- I2V SHA-256: `e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a` (full file read; no tensor loading).
- Combined lock: `design/h3-i2v-model-lock.json`, SHA-256 `4ee404910332151d529509d3ca6cbfa7c829d0df34109fc0e93166f3dccdcc18`.

The user identified the neutral-named I2V file. Its 932 tensor entries, dense 50-block architecture and 200 INT8 descriptors (`int8_tensorwise`, ConvRot, group size 256) match the existing loader/schema. Identical headers do not prove output quality or GPU compatibility. No embedded provenance established training/distillation or recommended sampling settings. Fixed 4-step Euler/simple, CFG 1 and seed 42 are retained as an explicitly experimental profile, not validated I2V recommendations. No LoRA is loaded.

Both profiles share the existing Qwen3-VL text/vision encoder and video/audio VAEs. The manifest explicitly pins each profile's DiT hash. Protocol v3 carries bounded raw RGB with dimensions and SHA-256; worker rejects incompatible profiles, bodies and hashes before generator construction. Model hashes remain checked before GPU initialization. Pillow's C-only resize library is part of the verified dependency map.

## User flow and geometry

Choose **Animate** on a generated image in the Image workspace. The Video workspace retains that asset and selects **Animate image · I2V**. Enter a motion/audio prompt and select size and duration. **Source image ratio** is the I2V default; the five existing ratios remain available. The text profile remains the initial application default.

Director source: `/home/thhel/git/comfyui-helto-director`, commit `ab8a643af92b3fe63adc3b78e626e4656b9d800e`. Reuses 0.1/0.2/0.4/0.5/1.0 MP target areas, nearest-32 half-up alignment, portrait handling, 24 fps and upward `17*k+5` frame alignment. Actual dimensions and duration are shown. Source ratio extends that same area formula using actual source dimensions; portrait override is disabled for this mode.

Conditioning source: `/home/thhel/git/ComfyUI`, commit `446ef4e005fc7134d53ce6869ebd77d2134cd39c`, `comfy_extras/nodes_minimax_h3.py::MiniMaxH3ImageToVideo`. First image uses Pillow Lanczos stretch without crop to the chosen canvas. The same resized RGB feeds vision tokens and video VAE. Frame 0 is a geometry condition, not an identity reference and not an output-frame replacement. Non-source ratios intentionally stretch, matching ComfyUI. No last-frame input, timelines or reference bank.

The start asset stays in the existing owner-scoped session storage; a UI clone retains its preview while later image results change. Admission checks ownership, type and bounds; execution checks encoded-file hash and decodes bounded RGB. A missing/closed-session asset is rejected rather than recovered from a local private path. Input image limits: 8192 per edge and 32 MiB encoded/RGB. Output retains the existing 2 GiB decoded AV bound. These are host/transport bounds, not VRAM-fit promises.

Native work remains sequential: text/vision → unload → video-VAE image encode → unload → selected DiT → unload → video-VAE decode → unload → audio-VAE decode → unload. The same supervised worker slot and GPU coordinator ticket are used. Errors/cancellation reap the worker; there is no second resident model profile.

## Verification and remaining acceptance

CPU coverage includes strict profiles/inputs, source geometry, bounded RGB/hash, wrong-owner/stale-session rejection, source lifetime, image→I2V→text H3→image queue flow, pipe errors/cancellation/reaping, manifest drift, headless UI and privacy, AV timing, existing image contracts and frozen keyframe/layout fixtures. Native Pillow matches installed Pillow byte-for-byte on six synthetic resizes.

The broad historical `primitives` target could not compile because unrelated core diagnostic fixture files were absent from the frozen checkout. Those files were not read, restored or changed. `inference/tests/h3_product_layout_cpu.rs` runs only the three existing H3 layout reference tests and passes; it does not initialize the native Runtime, model loader or CUDA.

GPU acceptance remains pending for **all I2V configurations**, including the default, the chosen sampling settings, vision encoder and frame0 VAE/DiT coupling, generated audio, quality, 16 GB peak/release and real model switches/error/cancellation. Previous Ref2VA GPU evidence applies to the previous fixed package, not this new binary. Native GUI and perceived AV sync still need manual verification. The pre-existing MKV duration estimate discrepancy remains open.

No D-05/core implementation changes. The worker uses frozen base `cd0ef53be69dd539ef0ffdf0cb03f4c1c5835fc7` plus the enumerated H3 adapter overlay. Runtime package `comfyui/h3-i2v-20261020` records source hashes, patches, dependency hashes and test evidence. Follow its `START.txt` for manual startup and rollback. Existing loaded binaries are not overwritten.
