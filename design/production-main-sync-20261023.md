# Production source reconciliation, 2026-10-23

The source declared by selected product `20261023T054300Z-f71a3541` is now
integrated into the main checkout. This imports the product implementation;
it does not activate a package, restart services, or reproduce a binary byte
for byte. The existing H3 memory improvements and Multiple prompts draft fix
remain on top of the product source.

## Provenance and scope

The selected package declares `comfyui/ui-groups-20261021/source` as its source
root. Its server and workers are inherited through earlier immutable packages;
its UI was built in that source tree. All selected server/UI/H3/image-worker
and proxy binary hashes were verified against `package.json`. The bundled Comfy
Python scripts and proxy stack script match the imported source exactly.

[The source inventory](production-source-20261023.json) records 760 source,
asset, schema and build inputs, their original product-source hashes and their
integrated hashes. The declared source directory remained unchanged throughout
this operation. This covers the application workspace, contracts, prompt
assets, profiles, tests, H3 product modules and inference dependencies. No
runtime logs, user prompts, job history, user media or user settings were read
or copied.

The old package metadata records executable hashes and source locations, but
not a complete immutable source digest or Git revision. Therefore this is an
audited reconciliation with the package-declared build source, not a claim
that every historical executable can be cryptographically tied to that source.
The newly committed source inventory removes that ambiguity for this snapshot.

## Product capabilities brought into main

- H3 protocol 16, runtime options, sampler selection, adaptive progress,
  reference cache, trained upscale, LoRAs and sampling previews.
- Product text-component residency and the corresponding checkpoint/model
  adapters; image-worker sampling and preview transport.
- Private storage and settings integration across server, UI and proxy,
  persistent media references and server-side path registration.
- Current timeline authoring, navigation, selections, captures, media health,
  reference/cache controls, audio and generation controls.
- Matching static assets, protocol fixtures, test helpers and dependency locks.

## Deliberate retained differences

The product source still uses the older inference core at
`cd0ef53be69dd539ef0ffdf0cb03f4c1c5835fc7`. Eleven generic core files already
contain later parity work on main. Those files are retained; checkpoint.rs is
merged with the product's Aimdo-path/dynamic-text additions. H3's product
sampling implementation retains the existing res_multistep entry point while
using the product's general named-sampler dispatch.

The H3 separate minimum and terminal-release helper are retained in the
static component path. The product's dynamic text path is preserved, rather
than replacing it with the older static-text probe path. Earlier GPU results
are evidence for their documented bounded workloads, not new acceptance of
every combined production option.

The four-file Multiple prompts draft fix is reapplied to the product UI.
Existing design decisions and repository material absent from the product's
build snapshot are preserved.

Required integration repairs are limited to tests/documentation: regenerate
the checked-in schema from the imported Rust types; point cancellation/error
fixtures at the actual `/v1/video/assets/link` contract; share duplicate fixture
module imports; move one test module after production items; use equivalent
array/integer spellings required by current Clippy; format two test tools.
No runtime behavior was altered for these repairs and no checks were disabled.

## Rebuild path

The normal Cargo workspace now builds the production server, UI and proxy
from main. `tools/prepare-h3-product-source.py` includes the complete protocol-16
adapter, shared H3 contracts and required text-residency dependencies on the
frozen inference base. `native/h3-product/checkpoint.patch` contains just the
product additions to that base's checkpoint module, avoiding accidental import
of unrelated later generic checkpoint modules. Repeated preparation restores
the same product checkpoint and copies identical current H3 modules/contracts.

This updates Rust source preparation only. No stable-diffusion.cpp sources,
native C++ patches, Qwen baseline flags or active product artifacts changed.
Packaging and activation remain separate explicit actions.

## Validation

- Workspace tests and examples compile with locked offline dependencies.
- 446 workspace library tests pass, with one pre-existing ignored test.
- The 16 selected server integration suites pass (152 test results), covering
  H3 contracts/options/transport, LoRA catalog, video API/jobs, persistence,
  assembly, schemas, media health and storage.
- Nine UI integration results and two proxy recovery results pass. The three
  explicit large-file streaming tests remain ignored.
- H3 transport/budget CPU suites: 25 pass. Frozen product H3 library tests:
  17 pass, two explicit opt-in tests remain ignored.
- Comfy adapter standard-library CPU tests: 15 pass.
- Warnings-denied Clippy passes for the complete workspace/all targets,
  the UI preview feature, current main image/H3 worker binaries, the frozen
  H3 worker, and the H3 CPU test crate.
- Workspace product binaries and the frozen H3 worker build successfully.
  Formatting, whitespace and source preparation checks pass.

Synthetic fixtures were created only beneath a fresh owned task directory.
Sandbox escalation was used for that directory and local test listeners.
No GPU/model execution, desktop control, private-data access, production
restart, product selection change or push occurred during integration.
