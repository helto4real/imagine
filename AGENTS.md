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
