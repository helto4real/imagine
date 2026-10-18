# Stage 1: original local Triton file-cache mechanics

2026-10-18. Necessary source-compatible continuation, with no new approval,
backend substitute, numerical tolerance, kernel or tuning-policy change.

The Rust adapter retains default cache/dump directory creation, override's
read-only manager behavior, lock-path metadata without creating/acquiring a
new lock, original `__grp__` filenames and ordered child groups. A missing child
is removed from the returned group without invalidating existing children.
Directories count as existing paths, exactly as in the source.

Writes use a fresh UUID4/PID temporary directory and POSIX atomic replacement.
The original parent-removal loop remains; no fsync, eviction, failure cleanup
or locking policy is added. Successful writes leave no temporary directory.
Group JSON uses original ASCII escaping and separators, compared as complete
bytes. The source's override `put` AssertionError and empty default-directory
RuntimeError are retained.

Missing/unreadable/invalid-UTF8/malformed-JSON group files produce cache misses.
Wrong but parseable structures retain the original AttributeError, TypeError
or descriptor OverflowError. The source accepts NaN/Infinity and arbitrary-size
JSON integers; the adapter preserves their types before applying the source's
path/descriptor rules. It does not turn those source errors into cache misses.
Duplicate object keys retain their first insertion position and final value.

A cache text value stores Python code points losslessly. This retains unpaired
surrogates, Unicode pairs and the frozen Linux UTF-8/surrogateescape filesystem
encoding. Own non-UTF8 filenames and surrogate-containing group keys/paths are
compared through explicit code-point fixtures, without replacement characters.
No Python interpreter, generic Python VM or new dependency is involved.

## Evidence and limits

Five source scenarios contain 322 ordered filesystem/call transitions across
cache, dump, override and simultaneous override/dump flags, plus empty-base
rejection. Focused Rust comparison passes all transitions and complete JSON
bytes. Source and Rust each create fresh owned directories; relative operand
paths are identical, so no diagnostic path normalization is used. Test cleanup
is confined to the directory created by that test. The original package,
ordinary caches, models, prompts, logs and media are not accessed.

The full [102 crate / 94 unfiltered Python-free checkpoint](stage-1-triton-file-cache-checkpoint.json) passes with 25,112 physical records (25,089 ordinary, five earlier approved Sol, ten D-04 and eight known-mask-padding). Four Stage 0 locks and all 32 original CUDA objects are unchanged.
These are local cache mechanics, not a completed general runtime JIT. Full
compiler/cache integration, dynamic options, restoration and broader failure
states remain Stage 1 obligations. Concurrent writer scheduling, interrupted
writes, I/O failures containing random temporary paths, mutable CPython decoder
limits and custom cache classes are not claimed as runtime-validated here.

The expanded test initially used pathlib with a byte path, which the test
harness rejected; its owned setup now uses Python's byte-capable open. The Rust
harness then assumed all directory-entry names were UTF-8 during temporary-file
inspection; it now filters the known ASCII temporary prefix before decoding.
Transient Rust escaping mistakes were caught during formatting/build and fixed.
Accepted results are the final passing source/Rust comparisons.
