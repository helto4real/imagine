# Stage 2 publication, 2026-10-19

Tomas explicitly requested committing the completed Stage 2 follow-up before
implementing Stage 3. The publication commit is titled
`Preserve upgraded Qwen Stage 2 acceptance and BF16 product support`.

It preserves ordinary BF16 model selection, the upgraded original-source
BF16/Turbo automatic LoRA comparisons, strict acceptance/rejection tools,
bounded synthetic Float32 fixtures, source fingerprints and documentation.
The independently exported focused source passed 150 workspace tests with one
deliberately ignored live test and all-target Clippy. The larger working tree
separately passed 152 tests with two ignored tests; those counts include
unrelated native/UI work excluded from this publication.

The [follow-up report](stage-2-upgraded-followup.md) and its retained evidence
describe the validation checkpoint before this publication. Their historical
uncommitted flags and fingerprints are preserved, not relabelled. The preceding
working-product baseline remains `f6fc1cd`.

Unrelated native optimizations, scheduler logging and UI layout changes remain
outside the commit. No ordinary runtime data is read, no service is restarted
and no push is performed. Stage 3 implementation is separately authorized by
the same user request, under the existing exact-parity/design constraints.
