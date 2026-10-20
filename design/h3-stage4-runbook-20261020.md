# Standalone H3 freeze: restore and run

This runbook belongs to the local H3 freeze on `h3-stage4-isolated`, based on
`901c847929cc9a68d046910fefdda1dba30e8214`. The commit containing this file is
the freeze commit. `h3-stage4-evidence/freeze-package.json` locks its files,
the accepted code snapshot, dependency inputs and retained executable hashes.
The accepted results remain **415 passed / 0 failed / 7 separately passed CUDA
tests**, 25 selected gates and 55 exact trained artifact comparisons. Packaging
does not rerun that acceptance or start GPU work.

## Fixed profile and prerequisites

Use the committed `inference/profiles/h3-short-av.json` unchanged: 384×256,
39 frames at 24 fps, stereo 32 kHz / 52,000 samples, seed 42, four Euler/simple
steps, CFG 1 and shifts 12/3. Its prompt is the existing synthetic blue-cube
prompt. Ref2VA uses the existing deterministic synthetic image fixture created
by the runner; this interface does not accept arbitrary prompts or user media.

The source reference is ComfyUI
`446ef4e005fc7134d53ce6869ebd77d2134cd39c`, selected by
`design/reference-current.json`. The retained host inputs are Python 3.12.13,
Torch 2.13.0+cu130 and Kitchen 0.2.38. CUDA builds use the existing pinned
manylinux image and toolchain identified by `inference/tools/build_native.sh`.
No dependency or model download is part of these commands.

This is a host-bound replay with these existing read-only dependencies:

- `/home/thhel/ChatGPT/imagine/comfyui/moss-20261018`: pinned source/toolchain,
  LLVM/Triton compiler and generated/AOT/runtime dependencies. Its existing
  immutable artifacts are prerequisites, not contents restored by Git alone.
- `/home/thhel/git/ComfyUI/.venv/lib/python3.12/site-packages`: locked packages.
- `/home/thhel/.pyenv/versions/3.12.13`: original-oracle interpreter runtime.
- `/mnt/media/gemensam/AI/ComfyUI/models`: exactly the four files named by the
  profile and `design/h3-stage4-evidence/selected-model-sha256.json`.
- `/home/thhel/ChatGPT/imagine/.imagine/gpu.sock`: the existing canonical Ticket
  coordinator. This procedure never starts, replaces or discovers a coordinator.

The model lock has SHA-256
`9cb461e3b118be5775602a81e4ddbd7e727b178f45aba9ce8fab0a3917c296b4`;
the profile has SHA-256
`53f5f4ac008fdb828da50c612415dc0fd043384d4a65233831b84ef601bcd5c4`.
The runner checks the exact selected filenames, sizes and mtimes. Each worker
also verifies its component's full SHA-256 before loading trained tensors.
Do not regenerate or replace these accepted locks to admit different weights.

## Environment

Run these blocks in Bash from this checkout. All new writable files remain
inside the ignored H3 root. The retained immutable dependency root is separate.

```bash
cd /home/thhel/.codex/worktrees/be07/imagine
export IMAGINE_PARITY_ROOT=/home/thhel/ChatGPT/imagine/comfyui/moss-20261018
export IMAGINE_REFERENCE_SITE=/home/thhel/git/ComfyUI/.venv/lib/python3.12/site-packages
export IMAGINE_REFERENCE_PYTHON=/home/thhel/git/ComfyUI/.venv/bin/python
export IMAGINE_H3_ROOT="$PWD/comfyui/h3-stage4-20261020"
export IMAGINE_NATIVE_BUILD_ROOT="$IMAGINE_H3_ROOT/native-build"
export IMAGINE_H3_KITCHEN_LIBRARY="$IMAGINE_NATIVE_BUILD_ROOT/kitchen-manylinux/libimagine_kitchen.so"
export IMAGINE_H3_KITCHEN_PROVENANCE="$IMAGINE_NATIVE_BUILD_ROOT/kitchen-manylinux/provenance.json"
export CARGO_HOME="$IMAGINE_H3_ROOT/cargo-core"
export CARGO_TARGET_DIR="$IMAGINE_H3_ROOT/target-core"
export CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=2
export LIBTORCH="$IMAGINE_REFERENCE_SITE/torch" LIBTORCH_CXX11_ABI=1 CXX=clang++
export IMAGINE_CUDA_INCLUDE="$IMAGINE_PARITY_ROOT/cuda13/nvidia/cu13/include"
export LD_LIBRARY_PATH="$IMAGINE_REFERENCE_SITE/torch/lib"
export XDG_CACHE_HOME="$IMAGINE_H3_ROOT/cache" TORCH_HOME="$IMAGINE_H3_ROOT/torch"
export TRITON_CACHE_DIR="$IMAGINE_H3_ROOT/triton" CUDA_CACHE_PATH="$IMAGINE_H3_ROOT/cuda"
export HF_HOME="$IMAGINE_H3_ROOT/hf" TMPDIR="$IMAGINE_H3_ROOT/tmp"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset LIBTORCH_USE_PYTORCH LIBTORCH_BYPASS_VERSION_CHECK
mkdir -p "$TMPDIR"
```

## Read-only preflight; no GPU execution

Verify the accepted source snapshot and existing executable identities. The
snapshot is the source used for acceptance; the package lock additionally
covers this runbook and the retained failed/successful evidence.

```bash
python3 -S -B inference/tools/stage2_snapshot.py \
  --verify design/h3-stage4-evidence/final-source-inputs-v3.json
python3 -S -B - <<'PY'
import hashlib, json
from pathlib import Path
lock = json.loads(Path('design/h3-stage4-evidence/freeze-package.json').read_text())
def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
for name, digest in lock['tracked_files'].items():
    assert sha(Path(name)) == digest, name
for name, digest in lock['base_dependency_inputs'].items():
    assert sha(Path(name)) == digest, name
for item in lock['retained_native_artifacts']:
    assert sha(Path(item['path'])) == item['sha256'], item['path']
assert lock['acceptance']['result'] == 'passed'
print('Frozen files, model/profile locks and retained executables verified; no GPU work')
PY
```

The accepted native worker is `target-core/debug/h3-stage4` beneath the H3 root,
SHA-256 `b0c73a729febdd413773a9eb60532925efb3d302aa4bf020570a494c401c7705`.
The final bridge is **not** the old dependency-root control bridge: it has
SHA-256 `ab7c807cd5f9aac244f58a302d9500a6ab04b0314b7223404881ce407ed0d4a3`
and source provenance SHA-256
`aa6d068f30fabc2b161415a96e270a55094d6aca2033cba964a64e78d757ed5d`.
The runner verifies both the selected bridge and its current source hashes.

## Restore missing executables; CPU compilation only

Skip this section when the retained files pass preflight. These are build
recipes, not commands run during freeze packaging. They require the existing
track-local Cargo caches and pinned external inputs above. Cargo remains
locked/offline. No tests, inference or GPU devices are used by these builds.

```bash
cargo build --manifest-path inference/Cargo.toml --bin h3-stage4 --locked --offline
env CARGO_HOME="$IMAGINE_H3_ROOT/cargo-product" \
    CARGO_TARGET_DIR="$IMAGINE_H3_ROOT/target-reservation" \
    cargo build --manifest-path inference/tools/gpu-reservation/Cargo.toml \
      --bins --locked --offline
bash inference/tools/build_native.sh
```

The optional ignored primitive-test executable can be rebuilt without running
tests using `cargo test --manifest-path inference/Cargo.toml --test primitives
--no-run --locked --offline`. Its accepted hash is in
`full-primitives-torch-registration-01/selection.json`. It is not needed to run
the trained worker above.

`IMAGINE_NATIVE_BUILD_ROOT` keeps generated native files, objects, the bridge
and `provenance.json` in the H3 tree. Do not rebuild into the immutable
dependency-root `kitchen-manylinux` directory. Keep Kitchen's original static
CUDA linkage: only host register/unregister resolve through LibTorch's already
loaded shared runtime. The rejected shared-linkage candidate is not a fallback.

These recipes restore from tracked sources and locked external dependencies;
they do not promise identical ELF hashes under a changed Rust/host compiler.
Compare a rebuilt artifact against the frozen identity before reusing the old
acceptance claim. A different executable/provenance remains a new build, without
changing the accepted locks or starting validation automatically.

No ignored stable-diffusion.cpp source was changed for H3. Its existing
`native/upstream-lock.json` and two `native/patches/` files remain unchanged base
inputs. The H3 Rust code and LibTorch/Kitchen bridge changes are tracked directly.

## GPU replay: no-reference

This block starts new GPU work. It was not executed during packaging. Choose a
fresh run name; existing accepted evidence and outputs are never overwritten.
The CPU Python process supervises the reservation; the actual native worker is
a Python-free ELF inside the admitted PID namespace. Do not invoke the worker
directly or use a separate lock instead of the canonical coordinator.

```bash
h3_run=noref-freeze-replay-01
for h3_phase in text dit video audio; do
  "$IMAGINE_REFERENCE_PYTHON" -S -B inference/tools/run_h3_trained_phase.py \
    --engine native --phase "$h3_phase" --run-name "$h3_run" \
    --root "$IMAGINE_PARITY_ROOT" --site "$IMAGINE_REFERENCE_SITE" \
    --binary "$CARGO_TARGET_DIR/debug/h3-stage4" --own-root "$IMAGINE_H3_ROOT" \
    --reservation "$IMAGINE_H3_ROOT/target-reservation/debug/imagine-native-gpu-reservation" \
    --entry "$IMAGINE_H3_ROOT/target-reservation/debug/namespace_entry" \
    --output "$IMAGINE_H3_ROOT/replays/$h3_run/$h3_phase" || break
done
```

Phases load one component at a time. They exchange only this run's explicit
tensor artifacts under `$IMAGINE_H3_ROOT/trained/$h3_run/native/`. Each reservation
finishes only after the owned namespace exits and the launcher is reaped.
Check all four phase selections report `passed` before exporting the result.

## GPU replay: one synthetic-reference Ref2VA

This uses the same profile and a separate fresh run. The text phase receives
the own image and the reference phase runs the actual trained video VAE encoder.

```bash
h3_run=reference-freeze-replay-01
for h3_phase in text reference dit video audio; do
  "$IMAGINE_REFERENCE_PYTHON" -S -B inference/tools/run_h3_trained_phase.py \
    --engine native --reference --phase "$h3_phase" --run-name "$h3_run" \
    --root "$IMAGINE_PARITY_ROOT" --site "$IMAGINE_REFERENCE_SITE" \
    --binary "$CARGO_TARGET_DIR/debug/h3-stage4" --own-root "$IMAGINE_H3_ROOT" \
    --reservation "$IMAGINE_H3_ROOT/target-reservation/debug/imagine-native-gpu-reservation" \
    --entry "$IMAGINE_H3_ROOT/target-reservation/debug/namespace_entry" \
    --output "$IMAGINE_H3_ROOT/replays/$h3_run/$h3_phase" || break
done
```

## Viewing export and retained artifacts

After either complete successful replay, the existing CPU-only export creates a
new MP4 and codec/source manifest. `$h3_run` selects that replay; do not reuse an
existing output filename.

```bash
"$IMAGINE_REFERENCE_PYTHON" -S -B inference/tools/export_h3_probe_clip.py \
  --site "$IMAGINE_REFERENCE_SITE" --own-root "$IMAGINE_H3_ROOT" \
  --run "$IMAGINE_H3_ROOT/trained/$h3_run/native" \
  --output "$IMAGINE_H3_ROOT/trained/$h3_run/native/replay.mp4"
```

Git retains sources, Cargo locks, the fixed profile/model lock, original-source
locks, accepted/failed evidence and native build provenance. It does not contain
trained models, executables, shared libraries, native objects, caches, raw
trained output tensors or MP4 files. Keep the existing ignored H3 root to retain
the exact accepted artifacts. The native artifacts and clips are identified by
hash in `freeze-package.json` and `acceptance.json`.

The accepted clips remain:

- `comfyui/h3-stage4-20261020/trained/noref-final-repeat-01/native/no-reference.mp4`
- `comfyui/h3-stage4-20261020/trained/reference-final-01/native/synthetic-reference.mp4`

If lost, rebuild missing executables above and regenerate outputs using the
existing profile and model locks through the GPU replay blocks when GPU use is
authorized again. Viewing exports are outside exact tensor acceptance. Codec
versions may change MP4 bytes; they do not replace the retained tensor comparisons.
Historical rejected-binary sources and its restoration patch are identified in
the package lock; failed evidence remains unchanged.
`h3-stage4-evidence/rejected-shared-runtime.patch` reconstructs the rejected
candidate's three native source files from this freeze. It was checked in an
own temporary copy, with resulting source hashes compared to the retained
candidate. Apply it with `git apply --unidiff-zero` only to a disposable source copy and
build into a separate
own native build root; it is not an alternative accepted backend.

## Product boundary

This freeze is the standalone core and development runner. Product worker
selection, engine/scheduler/server wiring, request/result types, UI and product
video/audio export remain with the separate video integration track. No merge,
push, product restart or expansion to other models/profiles is part of this freeze.
