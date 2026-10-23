#!/usr/bin/env python3
"""Focused source-executed Director Stage 2 composition oracle; no Comfy import.

BasicScheduler is a synthetic facade returning planned FP32 curves. This verifies
prefix/manual/remaining-curve composition, not native scheduler generation or
trained model execution. Debug-only schedule descriptions are intentionally stubbed.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import types
import uuid

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "comfyui/ui-groups-20261021/source"
OWN = Path("/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01")
DIRECTOR = Path("/home/thhel/git/comfyui-helto-director/shared/h3/runtime/sampling.py")


def cases():
    def case(name, *, manual_first=None, manual_second=None, first_curve=None,
             second_curve=None, base=2, count=3):
        return {"name": name, "first_curve": first_curve or [1., .875, .625, .25, 0.],
                "second_curve": second_curve or [1., .8, .4, .1, 0.],
                "sampling": {"first": {"steps": 20, "sampler": "res_multistep",
                    "scheduler": "simple", "denoise": .5, "sigmas": manual_first or []},
                    "base_steps_to_run": base,
                    "second": {"sampling": {"steps": count, "sampler": "euler",
                        "scheduler": "karras", "denoise": .2, "sigmas": manual_second or []},
                        "profile": "h3-av-noref-v2", "handoff": "continue_latent"}}}
    manual_unused = case("manual_both_ignore_unused_schedule_metadata", manual_first=[1.2, .6],
                         manual_second=[.6, .2, 0.], base=0)
    for stage in [manual_unused["sampling"]["first"], manual_unused["sampling"]["second"]["sampling"]]:
        stage.update(steps=0, denoise=-1., scheduler="unused manual curve")
    return [
        case("generated_base_prefix_and_remaining_curve"),
        case("manual_first_overrides_steps_denoise_and_base_count", manual_first=[1., .7, .3], base=0),
        case("manual_both_keeps_plateaus_and_overrides_base_count", manual_first=[1., .6],
             manual_second=[.6, .6, .2, 0.], base=999),
        case("generated_first_manual_second", first_curve=[1., .75, .5, .25, 0.],
             manual_second=[.5, .2, 0.], count=2),
        case("one_step_second_bypasses_scheduler", manual_first=[1., .4], count=1),
        case("missing_high_endpoint_anchors_incoming_boundary", manual_first=[1., .8],
             second_curve=[.7, .4, .1, 0.], count=4),
        case("generated_plateaus_removed_before_interpolation", manual_first=[1., .7],
             second_curve=[1., .9, .9, .6, .6, .2, 0.], count=4),
        case("boundary_exactly_on_native_grid", manual_first=[1., .6],
             second_curve=[1., .8, .6, .2, 0.], count=4),
        case("manual_second_must_start_at_handoff", manual_first=[1., .6],
             manual_second=[.5, .3, 0.], count=2),
        case("manual_first_allows_finite_sigma_above_unit", manual_first=[1.2, .6], count=4),
        manual_unused,
        case("generated_interpolation_rejects_exhausted_fp32_resolution", manual_first=[1., 1.e-45],
             second_curve=[1., 1.e-45, 0.], count=4),
    ]


def reference(torch, planned):
    text = DIRECTOR.read_text()
    wanted = {"_validate_stage_sigmas", "_resample_remaining_schedule", "build_two_stage_sampling_schedules"}
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in nodes} == wanted
    # Satisfy only Director's tiny local return-value adapter. No Director package
    # or Comfy module import takes place, and no native/model/runtime file is read.
    package = types.ModuleType("cp39_own_director")
    package.__path__ = []
    native = types.ModuleType("cp39_own_director.native")
    native.node_output_values = lambda result: result
    sys.modules[package.__name__] = package
    sys.modules[native.__name__] = native
    namespace = {"__name__": "cp39_own_director.sampling", "__package__": package.__name__,
                 "torch": torch, "math": math, "SIGMA_TOLERANCE": 1.e-6,
                 "_schedule_debug": lambda *args, **kwargs: {"diagnostics": []}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(DIRECTOR), "exec"), namespace)
    output = []
    for item in planned:
        spec = item["sampling"]
        calls = []

        class Scheduler:
            @staticmethod
            def execute(*, model, scheduler, steps, denoise):
                calls.append({"source": model, "scheduler": scheduler, "steps": steps, "denoise": denoise})
                return (torch.tensor(item[f"{model}_curve"], dtype=torch.float32),)

        def options(stage):
            return {"steps": stage["steps"], "sampler_name": stage["sampler"],
                    "scheduler": stage["scheduler"], "denoise": stage["denoise"],
                    "sigmas": torch.tensor(stage["sigmas"], dtype=torch.float32) if stage["sigmas"] else None}

        try:
            stages, _ = namespace["build_two_stage_sampling_schedules"](
                samplers=types.SimpleNamespace(BasicScheduler=Scheduler), models=["first", "second"],
                base_steps_to_run=spec["base_steps_to_run"], stage1=options(spec["first"]),
                stage2=options(spec["second"]["sampling"]))
            result = {"stages": [{"values": t.tolist(), "bits": [v & 0xffffffff for v in t.view(torch.int32).tolist()]} for t in stages]}
        except (ValueError, RuntimeError) as error:
            result = {"error": type(error).__name__, "message": str(error)}
        output.append({"name": item["name"], "calls": calls, "result": result})
    return output, hashlib.sha256(text.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "comfyui/ui-groups-20261021/target-worker/debug/examples/h3_ui_stage2_schedule_cpu")
    args = parser.parse_args()
    mount = json.loads(subprocess.check_output(["findmnt", "-J", "-T", str(OWN)]))["filesystems"][0]
    assert (mount["target"], mount["source"], mount["fstype"]) == ("/home", "/dev/mapper/cryptroot[/@home]", "btrfs")
    assert not os.statvfs(OWN).f_flag & os.ST_RDONLY
    import torch  # Installed dependency only; never imports ComfyUI.
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    planned = cases()
    expected, source_hash = reference(torch, planned)
    directory = OWN / ("cp39stage2schedules-" + uuid.uuid4().hex[:8])
    directory.mkdir()
    input_file = directory / "cases.json"
    input_file.write_text(json.dumps({"cases": planned}, indent=2) + "\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    env["LD_LIBRARY_PATH"] = str(Path(torch.__file__).parent / "lib")
    actual = json.loads(subprocess.check_output([str(args.binary), str(input_file)], cwd=SOURCE, env=env))
    # Preserve both sides even if the new focused comparison finds a difference.
    (directory / "director.json").write_text(json.dumps(expected, indent=2) + "\n")
    (directory / "rust.json").write_text(json.dumps(actual, indent=2) + "\n")
    assert len(expected) == len(actual["cases"])
    outcomes = []
    for py, rust in zip(expected, actual["cases"], strict=True):
        assert py["name"] == rust["name"]
        error = "error" in py["result"]
        if error != ("error" in rust["result"]):
            (directory / "failure.json").write_text(json.dumps({"python":py,"rust":rust},indent=2))
            raise AssertionError({"source_result_mismatch": py["name"], "evidence": str(directory)})
        if not error:
            assert py["calls"] == rust["calls"], (py, rust)
            assert [stage["bits"] for stage in py["result"]["stages"]] == [stage["bits"] for stage in rust["result"]["stages"]], (py, rust)
        outcomes.append({"name": py["name"], "matching_error": error,
                         "byte_identical_sigmas_and_callback_contract": not error})
    (directory / "director.json").write_text(json.dumps(expected, indent=2) + "\n")
    (directory / "rust.json").write_text(json.dumps(actual, indent=2) + "\n")
    report = {"passed": True, "device": "cpu", "case_count": len(outcomes), "cases": outcomes,
              "director_source": str(DIRECTOR), "director_sha256": source_hash,
              "artifact_root": str(directory), "binary": str(args.binary),
              "scope": "Director source-executed Stage2 schedule composition on planned FP32 curves; callback arguments and all successful sigma bits exact. Invalid cases require rejection in both implementations; error text is not equivalent. No native scheduler generation, real models, GPU, sampling output or full pipeline quality parity."}
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
