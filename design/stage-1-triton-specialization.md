# Stage 1: original Triton specialization/cache keys

2026-10-18. Source-compatible continuation; no additional approval or policy
exception. This is key construction, not a completed executable JIT/cache.

Rust reproduces the frozen `ASTSource.hash`, compiler key and unpadded base32
cache-directory encoding for the typed FP8 frontend. Inputs retain original
function dependency keys, argument-name-sorted signatures, pointer/count
alignment attributes, 32/64-bit counts, count=1 constexpr and block-size
constants. Runtime tensor contents and ordinary scalar values are not added to
the original key. Backend/compiler/options dependency hashes retain their
original positions; no new cache invalidation policy is introduced.

384 specialization records compare exact AST, compiler and directory strings.
Six environment maps additionally compare original sorted tuple repr and full
compiler digests, including quotes, backslashes, controls, Swedish letters,
emoji, nonbreaking space, format/line controls, private-use and tag characters.

Python's string repr uses a different printable definition from Rust's basic
character predicates. A 712-range immutable table is generated from the locked
CPython 3.12.13 / Unicode 15.0.0. Rust implements quote choice, escapes and
Unicode handling directly. The SHA256 of the complete newline-delimited repr
sequence matches original Python for all 1,112,064 Unicode scalar values.
Unpaired Python surrogates/OS surrogateescape decoding are outside this tested
string component; they are not silently replaced or claimed complete.

SHA256 uses the already locally locked `sha2` 0.10.9 crate, now an explicit root
dependency. There is no new download, Python runtime, remote cache or cloud
service. Source/key metadata remains synthetic development material.

The full [101 crate / 93 unfiltered Python-free checkpoint](stage-1-triton-specialization-checkpoint.json)
passes with 25,107 physical records: 25,084 ordinary, five earlier approved Sol,
ten D-04 and eight known-mask-padding records. Four Stage 0 locks and 32 original
CUDA objects remain unchanged. The preceding frontend/driver proof is retained;
all 28 general registered Triton bodies still require JIT/cache integration.
Local file-cache behavior and full dynamic option/error coverage remain work.
