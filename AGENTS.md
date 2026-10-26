# Imagine project guidance

## Runtime data must not be read

Prompt templates and prompt guidance in project/source files may be read.
Never read runtime data: logs, actual submitted prompts, job databases, job
history, chat/session history, generated images, runtime reference images or
metadata from these sources. This prohibition also applies to read-only access,
filtered queries and diagnostic or debugging work. Do not infer an exception
from access availability, a debugging request or permission from prior tasks.

Before changing Qwen inference settings or proposing another optimization, read:

- `experiments/qwen-image21-sage/BASELINE.md` — accepted frozen 1600×896 profile.
- `experiments/qwen-image21-sage/TEST-DECISIONS.md` — completed experiments, rejected options and evidence.

Use the frozen baseline as the comparison control. Do not repeat an already
completed experiment or propose a rejected setting as new without identifying
concrete changed evidence, a relevant regression or a newly requested workload.
A historical report's proposed next step is not authorization to execute it.
Keep native engine defaults separate from this resolution-specific profile;
preserve the frozen executable and its evidence when cleaning artifacts.
For native changes also follow `stable-diffusion.cpp/AGENTS.md`.

The local native checkout is ignored by this outer repository. Its published
form is `native/upstream-lock.json` plus the two complete patches under
`native/patches/`. Include new CUDA files when updating those patches. Verify
patch restoration and source hashes before publishing native changes; do not
assume an outer Git commit includes edits inside the ignored checkout.

## Product startup and native lock

Read `docs/PRODUCT-BUILDS.md` before changing launch/build behavior.
Normal start scripts build the local `main` checkout (including dirty product
changes) through `tools/product_build.py`; they never select `active-product.sh`.
`native/product/build-lock.json` explicitly separates main's product adapters
from the pinned Rust inference core, compiler, headers and native libraries.
Do not expand its product allowlist to bypass a core lock, rebuild CUDA during
normal startup, or silently accept changed dependency hashes. Only the system
inputs in its `build.system` section (the system `clang++` and `/usr/`
libraries, including the GPU driver) are recorded and reported instead of
enforced; do not widen that section to project-owned inputs. Keep pinned inputs
under the project, never in `/tmp`; `tools/restore_native_inputs.py` restores them. Use build-only
preparation for validation; service restarts and GPU runs are separate actions.
