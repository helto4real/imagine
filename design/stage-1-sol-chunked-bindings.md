# Stage 1: original chunked Sol producer and allocation context

2026-10-18. Source-faithful continuation under the existing authorization;
no new behavior exception, backend substitution or tuning change.

`kitchen_sol.rs` implements the original public chunked producer using the
unchanged native `sol_producer_begin`, `sol_producer_chunk`, `sol_attn_plan`
and `sol_attn_core` entries. It retains BF16 QKV chunks, 64-aligned starts,
RoPE packing, shared 128-element normalization weights, original workspace
views, scale-independent bootstrap statistics and the second quantizing pass.
Either absent statistic bootstraps both. A one-shot iterator is materialized
only when the original needs two passes; a factory is called afresh each pass.
Provided statistics retain the original device/dtype/contiguity conversions
and V-scale clamp. Next-step V-scale retains the original 1.1 margin.

Routing preserves workspace-centroid channel unpermutation, int8 pooled score
construction, source top-k budget, forced sinks and threshold backoff. The
backoff is original code, not an added numerical tolerance. The existing direct
CUDA Sol path shares the extracted pooled helper without changing its math.
Tail handling, token augmentation, block-length clamping and coarse correction
continue through the original native paths and ATen operation order.

The process-wide allocation context retains enter/exit/error/suppression
behavior. RoPE uses a single-entry cache keyed by input tensor-object identity,
T and rotary width, with a weak source reference and strong packed output.
In-place changes deliberately retain the original stale cached coefficients;
a distinct view object misses. A cache hit does not enter the allocation
context, even after the context changes. Frozen ComfyUI's malloc-graph pause
context is an inference memory hook, distinct from excluded workflow/UI graphs;
its integration belongs with the later model-memory stage.

## Differential evidence

- 94 RoPE/cache/context scenarios: CPU and CUDA, FP32/FP16/BF16, widths
  8/64/128, noncontiguous input, seven cache/identity transitions, context entry
  and exit failures, truthy exit callbacks, weak-source release and quantized frequencies.
- 115 producer records: 68 exact output tuples and 47 original errors
  (34 ValueError, 11 RuntimeError and two IndexError). Every ordinary successful
  reference tuple is reproduced on a second unmodified source call.
- Producer coverage includes T=64/129/192, one/two heads, three rotary widths,
  bootstrap/known/one-missing statistics, list/iterator/factory sources,
  noncontiguous/batched/single chunks, CPU statistics, tail policy, top-k,
  forced sinks, block lengths, coarse correction, augmentation, scales and
  iterator/factory errors. Empty chunks reproduce the original zero-grid
  launch error. Per-head normalization weights reproduce the original
  128-element capacity error; they are not silently accepted.
- Focused Rust comparison passes both test functions, including callback order.
  Full [99 crate / 91 unfiltered Python-free tests](stage-1-sol-chunked-checkpoint.json) also pass, with 24,234 physical records (24,211 ordinary, five earlier approved Sol, ten D-04 and eight known-mask-padding records). All four Stage 0 locks and 32 original CUDA objects remain unchanged.

Defined results compare complete tensor bytes and metadata with no tolerance.
The tests do not inspect original unwritten workspace padding. Native guards
reject deliberately invalid argument capacities before kernel consumption.
No short-scale native read, GPU OOM, production data or process inspection was
used. Default-dtype bootstrap with insufficient raw FP32 backing remains an
unvalidated source hazard, not a new public restriction or a repaired kernel.

The public typed adapter and these records are partial coverage, not proof of
every dynamic Python argument, tensor-subclass export or asynchronous lifetime.
General Triton bindings and remaining Stage 1 state/lifetime/restoration work
remain. No model inference or product integration is claimed.
