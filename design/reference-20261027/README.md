# Approved development reference, 2026-10-27

ComfyUI `426f5eea52a7ece1dd599fff4cb6cd1f352ddd24` (v0.43.0), Kitchen 0.2.40,
comfy-aimdo 0.6.0, Torch distribution 2.13.0. Tomas approved this upgrade on
2026-10-27; see [the upgrade report](../rust-upgrade-20261027.md). It is not a
rolling/latest policy.

The previous reference, ComfyUI `446ef4e` with Kitchen 0.2.38, stays locked in
`../reference-20261018` together with its expected results, and its source is
preserved at `comfyui/moss-20261018/ComfyUI`. Its Kitchen 0.2.38 Python
environment was not preserved separately: the main ComfyUI `.venv` already had
Kitchen 0.2.40 before this capture.

The static inventory contains 45,357 source/asset fingerprints, 704 parsed modules,
1,014 node contracts and 108 model classes. Only the Kitchen and Aimdo native
objects differ from the previous reference; the other 775 package ELF files,
including Torch and CUDA, are byte-identical. The capture reads Git-listed source
and installed package code without importing or running it.

The source acquisition lock captures Kitchen `v0.2.40` (`f8b7e97`) and Aimdo
`v0.6.0` (`68088de`) from `comfyui/rust-upgrade-20261027/sources`; their Python
sources match the installed packages byte for byte (Aimdo's generated
`_version.py` aside). Kitchen keeps the same cutlass and flash-attention pins, so
those and the other unchanged repositories carry over from the previous lock.

`../reference-current.json` selects this reference for development tools.
Product builds do not read it: the product build lock still pins the Rust core and
the Kitchen 0.2.38 bridge built for the previous reference.

Verify with `python3 -I tools/capture.py verify` and
`python3 -I tools/native_linkage.py verify` from this directory.
