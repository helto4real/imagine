# Upgraded Qwen Stage 2 follow-up evidence

These are explicitly authorized, isolated development artifacts with synthetic
inputs. They contain no ordinary Imagine/Dreamchat runtime data or model weights.
See [the report](../stage-2-upgraded-followup.md) for scope and acceptance.

- `accepted-bf16/` and `accepted-turbo/`: exact automatic original/Rust pairs,
  source-import and profile provenance, current tensor boundaries, PNGs, LoRA
  operands, component ownership and complete DiT/sampler lifecycle states.
  `acceptance.json` requires actual automatic offload and rejects placement
  replay. Turbo used original-first ordering. Both engines' algorithms and
  numerical/autotuning/memory policies are unchanged.
- `initial-bf16-failure/`, `turbo-first-failure/` and
  `turbo-post-stop-failure/`: preserved failed comparisons with different live
  free-memory/placement inputs. They are not accepted or relabelled. One failed
  pair retains bounded Float32 tensors for negative regression controls.
- `bf16-http/`: new ordinary BF16 product model selection/reference control
  through the real HTTP queue and private Rust worker, with exact original PNG
  pixels. The original control has no LoRA; automatic LoRA controls are above.
- `checks/`: complete workspace/Clippy/build results, exact initial-baseline
  source-checkout tests and six negative acceptance-gate controls. Socket-denied
  attempts are preserved separately from passing escalated runs. Verbatim logs
  may have trailing blank lines; they are not reformatted.
- `focused-source.patch`: reconstructs the independently tested focused source
  export on baseline `f6fc1cd`, with 150 passing tests and one ignored live test.
  It excludes unrelated native optimization/UI layout changes. The main working
  checkout's 152-pass/two-ignored run is recorded separately. Source-execution
  paths and compiled code are preserved; final report/evidence packaging was
  written afterward. This patch is not a subsequent Git commit.
- `aggregate-idle-gpu.json`: 20 seconds of aggregate GPU memory only, collected
  after owned test processes exited. It is not a process list. The 65 MiB range
  does not identify or attribute a particular external allocation.
- `deployment.json`: separately built candidate worker/native dependencies,
  not a modification of the normal baseline deployment.
- `source-inputs.json`: current working-source inputs for the trained controls.
  `focused-source-inputs.json` separately fingerprints the reviewable source
  without unrelated native/UI changes. `artifacts.json` fingerprints this
  evidence, excluding itself.

The small retained Float32 fixtures let the acceptance validator and its
negative controls run using only Python's standard library. They do not load
Torch, model weights, CUDA or ComfyUI. From the repository root:

```sh
evidence=design/stage-2-upgraded-followup-evidence
python3 -B inference/tools/test_qwen_automatic_acceptance.py \
  --rust "$evidence/accepted-bf16/rust" \
  --original "$evidence/accepted-bf16/python" \
  --comparison "$evidence/accepted-bf16/comparison.json" \
  --failed-rust "$evidence/initial-bf16-failure/rust" \
  --failed-original "$evidence/initial-bf16-failure/python" \
  --failed-comparison "$evidence/initial-bf16-failure/comparison.json" \
  --own-root /tmp/imagine-gate-regression-UNIQUE
```

`--own-root` must not already exist. Controls reject diagnostic placement
replay, unequal latents, a missing boundary, stale physical-memory accounting,
a stale tensor report and an actual failed trained comparison. Existing evidence
is never modified. The same strict positive gate runs for both trained controls
through `inference/tools/check_qwen_reference_upgrade.py`; `--original-first`
selects the separate order diagnostic. Fresh live runs remain explicit and
require the three selected local checkpoints, a new prefix, host-RAM headroom
and the prepared isolated runtime. No ordinary service is restarted.

These are bounded functional/numerical controls, not performance benchmarks,
physical peak-VRAM measurements or full model/pressure-matrix acceptance. Old
1600-pixel results and the baseline commit `f6fc1cd` retain their original scope.
