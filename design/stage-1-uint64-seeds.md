# Stage 1 full-width stochastic seeds

This is source-compatible implementation under the existing authorization. No
new numerical exception or autotuning policy is introduced.

The frozen Generator/native wrappers accept positive seeds throughout uint64.
Rust's former signed value/seed representations could not express the upper
half without disabling stochastic rounding or losing bits. `Value::UInt` now
retains the exact integer; native seed enablement is independent of its raw
64-bit representation. Original signed convenience calls remain supported.
Eager quantizers pass the full unsigned seed into the original ATen generator.
W4A8/W6A8 seed offsets use a wider host integer before checking the destination
boundary, rather than overflowing signed arithmetic or silently wrapping.

The original synthetic oracle has 138 repeated output cases: eager CPU/CUDA,
CUDA and registered Triton paths, tensorwise/rowwise INT8, activation/weight
rotation, ConvRot W4A4 and W4A8/W6A8. It includes 0, i64::MAX, 2^63, 2^63+1,
u64::MAX-1 and u64::MAX. Every output byte and retained input matches. Evidence:
`runtime/stage1-uint64-seeds-focused.log` in the isolated development root.

The bridge constructs a genuine c10 UInt IValue using the frozen ATen ABI;
unsigned values are not represented as floating scalars or negative integers.
A direct ownership/tag round-trip passes across that C++ boundary for seven
boundary values. UInt scalar packing is also supported by the frozen typed
Triton frontend. Separate Python/GPU-free comparisons pass 72 AdaLN and 72
RMS-RoPE source/TTIR cases and 12 complete original cubins per family. NA's
12 source cases and six complete binaries also pass byte-exact comparison.
The 36 additional UInt epsilon wrapper calls pass alongside the existing 144
cases (180 total), including exact output bytes and original cold JIT keys.
Compiler proofs remain separate from the 138 executed seed cases.

All original CUDA object sources and their compilation policy remain untouched.
The adapter bridge changes to expose original ATen scalar operations; its hash
therefore changes and needs a new restoration record. Full unfiltered validation
of the integrated continuation remains pending. Overflow/error and remaining
source argument/configuration obligations are not waived by these successes.
