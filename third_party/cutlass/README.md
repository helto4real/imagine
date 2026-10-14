# Exact CUTLASS header snapshot

These 822 headers are the archived FlashInfer-bundled CUTLASS headers used by
the accepted native build. The version header says 4.5.0, but 52 files differ
from NVIDIA CUTLASS tag v4.5.0 (commit
`e406c186f510a15091cce01f782020ceb7ba8eb5`). Substituting that tag is therefore
not an exact dependency restoration.

`manifest.json` records the SHA-256 of every header. `native/prepare.py --check`
validates them. `LICENSE.txt` and the original per-file notices are retained.
No Python/FlashInfer installation is needed to compile these C++ headers.
