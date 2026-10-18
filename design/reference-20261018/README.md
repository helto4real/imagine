# Approved development reference, 2026-10-18

ComfyUI `446ef4e005fc7134d53ce6869ebd77d2134cd39c`, Kitchen 0.2.38,
Torch distribution 2.13.0 (module 2.13.0+cu130). This is an explicitly approved
upgrade of the main local reference. It is not a rolling/latest policy.

The preceding accepted code baseline `c0d291c472b9cd0c20226f5dcf0e870be0e98cc2`
and its old locks in `../reference-local` are preserved. See
[the upgrade report](../comfy-reference-upgrade-20261018.md) for source/environment
preservation, installation results, implementation decisions and validation.

The static inventory contains 45,345 source/asset fingerprints, 750 parsed
modules and 17,971 definitions. Capability contracts are source inventory;
their presence does not expand inference scope to graphs, UI or hosted services.
The native linkage lock fingerprints the actual installed distribution objects;
the source acquisition lock records exact native source revisions and headers.
The two source archives contain source/config/tokenizer assets, not weights or
ordinary runtime data.

`../reference-current.json` selects this development reference for tooling.
Set `IMAGINE_REFERENCE_DIR` explicitly to select another lock. Old source/JIT
paths are restored through `inference/tools/with_frozen_reference.sh` together
with the preserved code/binaries; old expected results are not relabelled.

No claim of full upgraded Kitchen registry parity or full ComfyUI model parity
is made. New packed/mesh/autoregressive paths beyond selected Qwen and MOSS
inference remain part of the outstanding capability matrix.

Selected validation: 60 individual Python-free checks pass, and MOSS SFT INT8
ConvRot 1024×1024/50-step outputs and observed phase-memory policy are exact.
See [the retained evidence](validation/README.md). Qwen automatic BF16/Turbo
placement differences and untested MOSS branches remain explicit in the report.
