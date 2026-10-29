# H3 Combo/Influence on GPU: memory findings, 2026-10-29

Combo and Influence (Dreamchat's I5) were run for real for the first time:
through the proxy against the running Imagine, on the RTX 4070 Ti SUPER (16 GB,
about 2.4 GB used outside Imagine), with a synthetic project. Three 2 s scenes at
640×352 used the "Default ref2v" preset (20 steps `res_multistep`/`simple`,
shifts 12/3). Scene 2 continues scene 1 with Combo, scene 3 with Influence; the
boundary tail is the whole 56-frame source Take. No runtime data or logs were
read; the job API's own error codes and integer diagnostics were the evidence.

## Imagine runs

| Run | Worker input | Result |
| --- | --- | --- |
| Source scene | text only | succeeded, 85 s |
| Combo (twice), PyTorch SDPA | boundary video + frame-0 guide + reference window | `h3_out_of_memory` in sampling |
| Influence, PyTorch SDPA | boundary video + Picture 1 + reference window | `h3_out_of_memory` in sampling |
| Control: Motion Context Exp | continuation, no reference video | succeeded, 90 s |
| Control: plain reference video | 1.6 s of the source Take, no boundary window | succeeded, 115 s |
| Diagnosis: Combo without the window | temporary switch, not evidence | succeeded, 110 s |
| Combo, Kitchen INT8, dense query slices | as above | `h3_out_of_memory`, later (40 s) |

The failed Combo reported (MiB): free 110 of 15,992; active 12,250, reserved
13,420; allocated peak 13,273, reserved peak 13,472. The device was full. The
reference window's real split (16 full + 16 fading frames, then no boundary
attention) only ever ran on CPU before; Motion Context Exp's window is not split
at this length. The window adds transient key/value copies and a second
attention over the fade rows, which a placement without headroom cannot absorb.

## Changes kept

- **Kitchen INT8 attention is the H3 worker's default** (`h3_product/native.rs`).
  The worker had PyTorch SDPA although the checkpoints' `ComfyAttention`
  preference names Kitchen INT8, which ComfyUI follows. This was a defect; it
  also explains part of the speed difference to ComfyUI. Earlier H3 numerical
  evidence was taken with PyTorch SDPA and is not exact for Kitchen output.
- **Dense query slices** in `motion::Window::run`: each split query range is
  passed to attention contiguous, so a fused kernel can take it. Values are
  unchanged.
- **Memory figures on a sampling OOM**: the worker fills the existing `sizes`
  diagnostic with free/total device memory, active/reserved, and allocated and
  reserved current and peak, in MiB. Integers only.

## Tested and not adopted: counting references in the DiT budget

`memory_budget::dit` follows ComfyUI's `estimate_memory`, which counts only the
target video and audio latents. A test build added the keyframe and reference
latent elements (for Combo: 17 reference steps and the guide) to that estimate,
so static placement left more headroom. With it Combo (115 s) and Influence
(110 s) succeeded with the window on and an applied boundary receipt.

It was removed, by Tomas's decision, because:

- It is not ComfyUI's behavior. ComfyUI's `MiniMaxH3` has no
  `extra_conds_shapes`, so its estimate ignores references too, and Imagine
  already ports ComfyUI's reserve (`minimum_inference_memory`, 0.8 GB + 400 MB).
- The Python reference below shows ComfyUI succeeds with the same estimate. The
  real difference is the placement strategy, so a larger estimate would only
  hide it.
- It offloads more weights for every reference scene, so all of them get
  slower.

## Python reference: ComfyUI's own code for the same cases

`tools/h3-python-boundary-reference.py` runs ComfyUI's H3 Python code (checkout
`426f5eea`, Imagine's current reference target) for one case in one process. It
starts like `main.py` (allocator selection, comfy-aimdo dynamic VRAM, default
attention), loads the same four model files, builds the target with
`MiniMaxH3ReferenceToVideo`, adds `MiniMaxH3AddGuide` (Combo) or Picture 1
(Influence), installs Director's own reference-tail window (taken from
`attention_window.py` by AST) and samples with the preset's settings and
Imagine's compiled prompt. It ran under the canonical coordinator's ticket with
`gpu-reservation --process-form bwrap-pid-namespace`, without network, with
read-only mounts and only its own output directory writable.

| Case | Reference blocks | Conditioning | Sampling | Lowest free |
| --- | --- | --- | --- | --- |
| Source | none | 7.0 s | 25.6 s | 263 MiB |
| Combo | video (17 steps) + frame-0 guide | 11.6 s | 61.5 s | 237 MiB |
| Influence | image + video (17 steps) | 12.2 s | 66.1 s | 219 MiB |

ComfyUI also runs with the device nearly full, but its dynamic VRAM evicts
weights under NVML pressure, where Imagine's static placement fixes them once.
PyTorch allocator peaks are not comparable there: comfy-aimdo manages most memory
outside that allocator. All three reference runs faulted afterwards, in VAE
decoding, inside comfy-aimdo's native log callback; the tool now writes its
report after every phase and exits without interpreter teardown. Decode times
are therefore missing; the source scene took 33 s through sampling in ComfyUI's
code against 85 s for the whole Imagine job.

## Image diffusion models: Kitchen INT8

The image worker had the same defect: Qwen and MOSS diffusion models used
PyTorch SDPA (`server::attention`). Their diffusion executions now use
`dit_attention`, Kitchen INT8 as the checkpoints prefer; text encoders and VAEs
keep PyTorch SDPA, as in ComfyUI. The files are product files, so the core lock
is unchanged. One synthetic prompt and seed through the proxy, each run loading
its models after a server start:

| Profile | PyTorch SDPA | Kitchen INT8 |
| --- | --- | --- |
| Qwen 1600×896, 40 steps `er_sde`/`beta` | 72.2 s | 52.1 s |
| MOSS 1024×1024, 50 steps `euler`/`normal` | 48.1 s | 36.1 s |

The outputs differ in bytes, as INT8 attention is expected to; the Kitchen
images were inspected and are intact. A side-by-side quality comparison needs
both modes in one server session and belongs to the planned test matrix.

## Status and next steps

The Combo/Influence path is coupled end to end in code and synthetic tests, and
both ran on GPU with the tested budget, but without it they run out of memory at
this size on a 16 GB card. Next, in order: an attention mode runtime setting (Auto default, Kitchen
INT8, PyTorch SDPA) for video and images, dynamic VRAM as a runtime setting (on
by default, as in ComfyUI) for the H3 DiT and the image models, and a repeatable
performance comparison against this Python reference.
