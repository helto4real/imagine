# Selected upgrade validation, 2026-10-18

Reference: ComfyUI `446ef4e005fc7134d53ce6869ebd77d2134cd39c`, Kitchen 0.2.38.
Read [the scope and decisions](../../comfy-reference-upgrade-20261018.md) before
interpreting the results. These are explicitly authorized isolated development
artifacts, not ordinary product logs/history/media.

- `checks/`: successful final selected regression log, all 60 individual
  Python-free isolation reports, final Clippy/build output and fresh kernel
  fixture provenance. No trained weights were mounted for these regressions.
- `preservation/`: old environment/source-wrapper check, requirements conflict
  comparison and reference-path edit verification. The same 16 pip conflicts
  predate this upgrade.
- `moss-1024-final-buffered/`: accepted 1024×1024/50-step strict comparison,
  native binary SHA256, RAM guard and isolation proof. The explicitly reused
  original run is identified in `results.json`.
- `moss-runs/{large-rust,large-python}/`: original result metadata, memory policy,
  PNG and SHA256 manifests of every exported Float32 boundary. Tensor dumps
  remain at the `source_own_run` paths in those manifests, not in Git.
- `moss-first-trained-fused/` and `moss-runs/small-*`: preceding 320×224/4-step
  numerical control. Its older comparator did not check memory-policy equality;
  do not relabel it as the final strict memory result.
- `moss-1024-source-comparison/`: preceding large control with exact numerics
  and a failed metadata-only comparison. Reporter corrections and unbuffered
  export timing are documented in the report; this failed result is preserved.
- `qwen/automatic-int8/`: exact automatic trained control.
- `qwen/automatic-bf16-turbo/`: completed automatic comparisons with unequal
  placements and unequal outputs. These are not accepted numeric parity.
- `qwen/recorded-placement-diagnostic/`: exact BF16/Turbo comparisons after
  explicitly replaying recorded original DiT placements, including those plans.
  This is diagnostic evidence, not a replacement automatic memory policy.
- `implementation-inputs.json`: source/resource/fixture/profile/build-file
  hashes for the current implementation. Reference archives and native package
  hashes have their separate parent lock. `artifacts.json` hashes this evidence.

To repeat the bounded regression set after preparing the documented local
native environment, set `IMAGINE_PARITY_ROOT` and `IMAGINE_REFERENCE_SITE`, source
that root's `stage2-env.sh`, then run:

```sh
bash inference/tools/check_reference_upgrade.sh
```

The trained comparison is separate and opt-in. It uses the three local model
paths in `inference/profiles/moss-1024x1024-int8.json`; it checks available host
RAM before loading weights. Use a fresh prefix and run both engines serially:

```sh
python3 -S -B inference/tools/check_moss_reference_upgrade.py \
  --root "$IMAGINE_PARITY_ROOT" --site "$IMAGINE_REFERENCE_SITE" \
  --comfy /home/thhel/git/ComfyUI \
  --model-root /mnt/media/gemensam/AI/ComfyUI/models \
  --binary "$CARGO_TARGET_DIR/debug/moss-first-image" \
  --prefix moss-recheck-UNIQUE --profile inference/profiles/moss-1024x1024-int8.json \
  --mode run
```

A different reference requires new captures and an explicit lock; do not relabel
these expected results. This archive contains no model weights or executable
binaries. The local retained root is
`/home/thhel/ChatGPT/imagine/comfyui/moss-20261018`.
