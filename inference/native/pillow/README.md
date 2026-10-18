# Original Pillow pixel-buffer dependency

The Rust Qwen21 reference path uses the frozen Pillow 12.3.0 resampler and
RGB/RGBA conversion C code. `resources/pillow-source-lock.json` pins the
original archive, compiled C sources, headers and license. Upstream files are
compiled unchanged from the preserved 29-file local source subset in
`resources/pillow-12.3.0` (195,652 bytes including its license).
No Python interpreter, Python extension module or image codec is linked.

`bridge.c` supplies ownership/error hooks and a bounded RGB/RGBA buffer ABI.
The mutex replaces the original GIL gate around shared allocator state;
original `ImagingSectionEnter/Leave` releases/reacquires it during compute.
RGBA resizing retains Pillow's RGBa premultiplication, original LANCZOS and
conversion back. A same-size copy bypasses that conversion as `Image.resize`
does. Floating-point/byte conversions around this ABI remain in Rust.

Prepare offline from the preserved sources; `--source` can select a complete
matching extracted archive instead:

```sh
python3 -B inference/tools/prepare_pillow.py --output /absolute/isolated/pillow
```

Preparation verifies source hashes, builds the library, checks that no Python
symbol/dependency remains and records source/adapter/library hashes. Isolation
verifies that record before mounting only the resulting ELF file. The shim is
build-time declarations, never a Python runtime. Upstream Pillow's HPND license
remains required when distributing the compiled dependency.

Differential fixtures cover CPU/CUDA conversions, RGB/RGBA, alpha,
same/up/down dimensions and threaded independent callers. This selected ABI
does not claim general Pillow modes, codecs or environment allocator override
coverage. Those remain explicit broader-parity work.
