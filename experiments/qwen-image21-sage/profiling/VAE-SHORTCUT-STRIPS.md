# DupUp3D shortcut strips with direct addition

Date: 2026-10-14. Experimental, default off. The image-test memory gate did not pass.

## Results

The targeted shortcut and addition are bit-identical in all synthetic cases,
including the production shape. Its isolated workspace decreases, but the
complete VAE graph arena increases slightly.

| Measurement | Existing shortcut | 32 input rows per strip |
| --- | ---: | ---: |
| Isolated shortcut + addition workspace, retained live inputs | 5512.500 MiB | 2531.250 MiB |
| Complete VAE CUDA graph arena | 8663.867432 MiB | 8669.360596 MiB |
| Complete graph nodes | 1355 | 1472 |
| Additional CUDA pool estimate in empty probe | 394 MiB | 394 MiB |
| Scheduler fallback | No | No |
| Differing production output floats | — | 0 |
| Production output bytes compared | 825753600 | 825753600 |

The local saving is **2981.25 MiB (54.08%)**. The complete graph increases by
**5.493164 MiB**, so no extra capacity for diffusion weights is demonstrated.
Isolated workspace includes deliberately retained test inputs and must not be
substituted for the full-graph allocation. No full-image VAE time, image identity,
end-to-end latency or weight-reload measurement was run.

Two additional metadata-only variants, using 16 and 64 input rows, both produce
the same **8669.360596 MiB** arena. Their graph node counts are 1605 and 1415,
respectively. The 32-row variant is the one implemented and numerically tested;
the other row sizes are isolated header overrides used only by the probes.

## Where the arena limit moved

Exact allocation tracing explains why the local reduction does not reduce total
memory:

| Arena-ending allocation | Control | Shortcut strips |
| --- | --- | --- |
| Node / operation | 728 / CONT | 867 / IM2COL |
| Size | 1575 MiB | 3543.75 MiB |
| Offset | 7088.867432 MiB | 5125.610596 MiB |
| End | 8663.867432 MiB | 8669.360596 MiB |
| Owner | `decoder.upsamples.2` DupUp3D shortcut | `decoder.upsamples.3.upsamples.0.residual.2` |

The former full-size shortcut repacking tensor `[2,400,224,2304]` is absent from
the candidate graph. The next 800×448 convolution, with 576 input and 288 output
channels, creates an F16 im2col tensor `[5184,800,448,1]`. Its allocation now sets
the arena end. This is allocator placement evidence, not a sum of simultaneously
live tensor sizes or a promised saving from removing one tensor.

The normal and allocation-instrumented graph TSVs match byte for byte. Their
621 / 676 allocation entries reproduce the measured arena sizes exactly. The
existing diagnostic base archive was reused after verifying that only its
`ggml-alloc.c.o` member differs from the current normal archive. No ggml source
or allocator policy was changed.

## Implementation

`SD_VAE_STRIP_SHORTCUT=1` enables a bounded path in
`first_stage_model.decoder.upsamples.2.avg_shortcut`: 1152→576 channels,
spatial and temporal factor 2, first chunk, one input frame, CUDA, F32 input and
main result, contiguous correctly sized main result, and no temporal-tile graph.
Unsupported configurations use the existing full shortcut and addition.

The existing DupUp3D mapping runs on up to 32 input rows at a time, producing
64 output rows. No halo is needed because this shortcut only duplicates and
reorders elements. The existing `ggml_acc_inplace` operation adds each strip
directly into the main result, with unchanged main-plus-shortcut operand order.
The dependent ACC chain allows strip temporaries to be reused. No new CUDA
kernel, precision change or public API is introduced.

The current CUDA ACC kernel traverses the full destination for each strip,
passing through values outside that strip. This bounds temporary storage but
does not establish a latency benefit; its bandwidth cost is not benchmarked here.

All accepted diffusion flags and the existing two convolution strips remain
enabled. `SD_QWEN21_RETAIN_IDLE=0` and `SD_VAE_STRIP_RESIDUAL=0` throughout.
Prior rejected VAE upscale/im2col fusion and PAD-scale reuse remain off.

## Validation and limits

The synthetic test uses the real DupUp3D and addition path. Two live source
tensors remain marked as outputs to catch unintended input mutation. The main
result is a distinct computed tensor that the candidate may update in place.
Input patterns vary across elements and include positive/negative zero,
subnormals, rounding boundaries and large finite values.

All six comparisons pass complete output byte identity with unchanged live
inputs and no non-finite outputs:

- 1×1, 5×33 and 65×129 target inputs, including incomplete final strips.
- 400×224 production input → 800×448×576 output, 825753600 bytes compared.
- An unrelated shortcut prefix, which must take the original fallback.
- A non-first chunk, which must also take the original fallback.

All executed nodes are supported by CUDA. The control metadata graph is
byte-identical to the previous frozen convolution-strip reference. Native
server, synthetic harness, normal probe and allocation probe builds pass;
source/binary hashes and `git diff --check` pass. Only `wan_vae.hpp` changes in
the native source snapshot. These are synthetic and metadata results, not a
model-image acceptance test.

## Decision and next proposal

Keep the shortcut flag off by default and do not run images: the required
whole-VAE memory reduction was not achieved. Retention and block stripping
remain off. No accepted configuration is replaced.

The next focused proposal is to combine this numerically validated shortcut
candidate with strip convolution for the newly identified
`decoder.upsamples.3.upsamples.0.residual.2` im2col allocation. Check that
convolution's bit identity and the complete arena before any image run. That
additional convolution change is not implemented here, and its actual memory
saving is not assumed.

Only synthetic values and existing model-shape metadata were used. No model
weights, private ComfyUI logs/history/workflows/configuration or prompt tensors
were read or persisted. GPU diagnostics used approved sandbox escalation.
No image generation, ComfyUI execution, commit or push occurred.

## Artifacts

- [Synthetic test](test-vae-shortcut-strips.cpp).
- [Evidence validator](validate-vae-shortcut.py).
- [Build, source snapshots, patch, commands, graphs and logs](../runs/vae-shortcut-strips-build/).
- [Validation](../runs/vae-shortcut-strips-build/validation.json).
- [Allocation analysis](../runs/vae-shortcut-strips-build/allocation-analysis.json).
