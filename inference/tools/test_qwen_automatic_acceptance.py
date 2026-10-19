#!/usr/bin/env python3
"""Reject poisoned copies of explicitly supplied isolated comparison artifacts.

No models are opened or allocated. Original evidence is never modified.
"""
import argparse
import json
import tempfile
from pathlib import Path

from check_qwen_automatic_acceptance import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rust", "original", "comparison", "failed-rust", "failed-original",
                 "failed-comparison", "own-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.own_root.mkdir(parents=True, exist_ok=False)
    assert evaluate(args.rust, args.original, args.comparison)["accepted"]
    results = []

    def reject(name, rust, original, comparison):
        try:
            evaluate(rust, original, comparison)
        except AssertionError as error:
            results.append(dict(case=name, rejected=True, reason=str(error)))
        else:
            raise AssertionError(name + " was accepted")

    for name, mutate in (
        ("diagnostic_replay", lambda x: x.update(diagnostic_memory_plan=True)),
        ("unequal_latent", lambda x: x["metrics"]["latent"].update(exact_bytes=False)),
        ("missing_boundary", lambda x: x["metrics"].pop("context")),
    ):
        with tempfile.TemporaryDirectory(dir=args.own_root) as own:
            target = Path(own) / "comparison.json"
            report = json.loads(args.comparison.read_text())
            mutate(report)
            target.write_text(json.dumps(report))
            reject(name, args.rust, args.original, target)

    def links(source, target):
        for child in source.iterdir():
            if child.is_file():
                (target / child.name).symlink_to(child.resolve())

    with tempfile.TemporaryDirectory(dir=args.own_root) as own:
        own = Path(own)
        links(args.original, own)
        path = own / "memory-policy.json"
        path.unlink()
        states = json.loads((args.original / path.name).read_text())
        states[0]["loaded_bytes"] += 1
        path.write_text(json.dumps(states))
        reject("stale_memory_report", args.rust, own, args.comparison)
    with tempfile.TemporaryDirectory(dir=args.own_root) as own:
        own = Path(own)
        links(args.rust, own)
        path = own / "latent-f32.bin"
        path.unlink()
        data = bytearray((args.rust / path.name).read_bytes())
        data[0] ^= 1
        path.write_bytes(data)
        reject("stale_tensor_report", own, args.original, args.comparison)
    reject("actual_failed_pair", args.failed_rust, args.failed_original, args.failed_comparison)
    (args.own_root / "checks.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"Strict automatic gate: positive control passes; {len(results)} negative controls rejected")


if __name__ == "__main__":
    main()
