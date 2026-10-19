#!/usr/bin/env python3
"""Strict offline gate for explicitly selected isolated Qwen comparison artifacts."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evaluate(rust, original, comparison):
    report = read(comparison)
    rust_result = read(rust / "result.json")
    original_result = read(original / "result.json")
    assert not report.get("diagnostic_memory_plan"), "Placement replay is not automatic acceptance"
    assert not rust_result.get("diagnostic_memory_plan"), "Rust used a diagnostic placement plan"
    assert rust_result["source_profile"] == original_result["profile"], "Different control operands"
    profile = original_result["profile"]
    boundaries = {"context", "latent", "image-rgba"}
    if profile.get("custom_sigmas") or profile["scheduler"] == "beta":
        boundaries.add("sigmas")
    if "reference_resolution" in profile:
        boundaries.add("negative-context")
        for i, _ in enumerate(profile["references"]):
            boundaries.add(f"reference-vision-{i}")
            if profile["reference_vae"]:
                boundaries.add(f"reference-latent-{i}")
    assert boundaries == report["metrics"].keys() - {"png_rgba8"}, "Missing/alternate numeric boundary"
    assert all(
        v["exact_bytes"] and v["all_finite"]
        for name, v in report["metrics"].items() if name != "png_rgba8"
    ), "Non-exact or non-finite tensor boundary"
    for name in boundaries:
        assert [digest(root / f"{name}-f32.bin") for root in (rust, original)] == report["metrics"][name]["hashes"], f"Stale tensor comparison: {name}"
    assert report["metrics"]["png_rgba8"]["exact"], "PNG pixels differ"
    assert report["component_memory"]["all_observed_fields_exact"], "Component ownership differs"
    components = read(original / "component-memory.json")
    component_phases = {row["phase"] for row in components}
    actual_components = [row for row in read(rust / "memory-manager.json") if row["phase"] in component_phases]
    assert actual_components == components, "Actual component ownership differs"
    if profile.get("lora"):
        patched = 0
        for component in ("text", "dit"):
            x, y = [read(root / f"lora-{component}.json") for root in (rust, original)]
            assert x == y, f"Different LoRA operands: {component}"
            patched += len(x)
        assert patched, "LoRA control did not patch weights"
    assert report["sampler_lifecycle"]["sampler_lifecycle_exact"], "Sampler lifecycle differs"
    states = [read(root / "memory-policy.json") for root in (rust, original)]
    phases = [row["phase"] for row in states[0]]
    assert phases == [row["phase"] for row in states[1]] == [
        "dit_initial", "dit_sampling", "dit_unloaded"
    ], "Missing or reordered DiT lifecycle phase"
    fields = ("loaded_bytes", "buffer_bytes", "resident", "lowvram_passes")
    checks = []
    for x, y in zip(*states):
        for field in fields:
            assert x["placement"][field] == y[field], f"{x['phase']}: {field} differs"
        for field in ("pinned_bytes", "model_size", "model_lowvram", "sampler_buffer"):
            assert x[field] == y[field], f"{x['phase']}: {field} differs"
        assert digest(rust / f"sampler-{x['phase']}-f32.bin") == digest(original / f"sampler-{x['phase']}-f32.bin"), "Sampler buffer values differ"
        checks.append(dict(phase=x["phase"], all_observed_fields_exact=True,
                           loaded_bytes=x["placement"]["loaded_bytes"],
                           pinned_bytes=x["pinned_bytes"], model_lowvram=x["model_lowvram"]))
    assert states[0][0]["model_lowvram"], "Control did not exercise automatic offload"
    return dict(schema=1, accepted=True, diagnostic_memory_plan=False,
                variant=original_result["profile"]["variant"],
                all_numeric_boundaries_and_pixels_exact=True,
                component_memory_exact=True, dit_lifecycle=checks,
                comparison_sha256=hashlib.sha256(comparison.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rust", "original", "comparison", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.rust, args.original, args.comparison)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
