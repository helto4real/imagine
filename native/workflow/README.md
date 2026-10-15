# Separate Qwen Turbo compatibility engine

`compat.patch` is applied after both patches in `../upstream-lock.json` to a separate source restoration. It casts the BF16 zero-centered RMSNorm weight to F32 before `ggml_scale_bias`, whose CUDA implementation requires F32. Only the explicit Turbo workflow model selects this binary; frozen v2 remains unchanged.

From the Imagine root:

```sh
python3 native/workflow/build.py
python3 native/workflow/build.py --check
```

The first command restores pinned Git archives, applies all three patches, verifies the complete source-tree digest in `source-lock.json`, builds CUDA architecture 89, and writes a manifest with binary/input hashes under `.imagine/workflow-native`. Reuse rejects changed sources or inputs. To restore after local tampering or a future patch revision, preserve any needed evidence and use a fresh `.imagine/workflow-native` directory. Do not replace the accepted baseline build.

The check command verifies the base restoration, separate workflow source tree, input hashes and binary hash without compiling. The API also checks the base lock, compatibility patch and actual executable against the workflow manifest before starting Turbo. Full real-GPU evidence is listed in `proxy/PARITY.md`.
