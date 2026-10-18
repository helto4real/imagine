# Stage 1 structured source validation

This continuation implements frozen source behavior under the existing whole
Stage 1 authorization. It introduces no exception or changed numerical criterion.

The original CPU-only oracle contains 13,072 cases across the eager, Triton,
CUDA and CUDA-with-cuBLASLt constraint tables. It covers devices, empty and
nested tuples/lists/dictionaries, mixed numeric sequences, dtype objects and
initialized CPU tensors. All 13,072 candidate validations match the original
success/failure, parameter, exception class and reason. Only the previously
approved allowed-type set order criterion is used; original raw reasons remain
in the fixture. The focused test and warning-free clippy pass are recorded in
`stage1-structured-metadata-focused.log` under the isolated development root.

Source values retain tuple/list distinctions for Python hashability. Formatting
is deferred until a rejected value needs a diagnostic. Tensor validation keeps
owned handles and reads actual values only when the frozen NA rule does so.
The call context is borrowed for quantized dispatch rather than captured as a
raw pointer or a stale copy of runtime settings. Registry rules execute after
its policy mutex is released. Diagnostic `Debug` does not print tensor payloads.
Existing integer-sequence helpers remain available to the original Sol rules.

The existing metadata-only descriptions remain useful for synthetic descriptors;
they do not claim to contain actual tensor values. `metadata_for_call` carries
runtime dispatch context for values needing deferred operations. The ordinary
public Kitchen call already recursively dequantizes arguments before selection.

The two new focused CPU tests also verify 78 original NVFP4 literal AST keys.
Full integration and Python-free validation of this continuation remain pending.
The earlier full 141-test run used a preserved source/fixture snapshot and passed
its ordinary suite; its isolation run found a missing `nvdisasm` ELF dependency.
It must not be described as a complete passing isolation checkpoint.
