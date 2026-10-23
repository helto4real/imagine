#!/usr/bin/env python3
"""Separate diagnostic binary: exact owned Exp input, first masked step only."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OWN = Path("/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e")
CASE = OWN / "exp-step-01"
PRIOR = Path("/run/media/veracrypt1/imagine/.codex-h3-res20-compare-20261020-6f0b84c2")


def native_tools():
    spec = importlib.util.spec_from_file_location("night_native", ROOT / "tools/build-video-night-native.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def prepare():
    native = native_tools()
    env = native.environment()
    overlay = native.prepare()
    assert not CASE.exists(), "Preserve the completed own case; do not prepare it twice."
    diagnostic_source = native.CANDIDATE / "diagnostics/exp-step-01/source"
    assert diagnostic_source.resolve() == diagnostic_source
    assert not diagnostic_source.exists()
    diagnostic_source.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(native.SOURCE, diagnostic_source)
    CASE.mkdir(exist_ok=False)
    for name in ["rust", "python", "tmp", "artifacts", "base", "empty-models"]:
        (CASE / name).mkdir()
    source = OWN / "h3-e2e-08/saved-project"
    document = json.loads((source / "project.json").read_text())
    tail = next(item for item in document["resources"] if item["kind"]["kind"] == "h3_motion" and item["kind"]["take_id"] == "take_4bb77e7e286f409a9b50236a742ca067")
    data = (source / "continuity" / tail["sha256"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == tail["sha256"]
    (CASE / "source-tail.f32").write_bytes(data)
    scene = next(item for item in document["project"]["scenes"] if item["id"] == "own_h3_target_5")
    request = {"protocol": 6, "kind": "generate_video", "id": "own_exp_step", "profile": "h3-av-noref-v2", "prompt": scene["description"], "seed": 42, "shape": {"width": 416, "height": 224, "frames": 124, "fps": 24}, "continuation": {"method": "motion_context_exp", "include_audio": True, "descriptor": tail["kind"]["descriptor"], "sha256": tail["sha256"]}}
    (CASE / "request.json").write_text(json.dumps(request, indent=2))
    runtime = json.loads((OWN / "h3-e2e-08/runtime.json").read_text())["h3"]
    checkpoints = json.loads((ROOT / "inference/profiles/h3-short-av.json").read_text())["checkpoints"]
    (CASE / "model-path.json").write_text(json.dumps({"path": str(Path(runtime["model_root"]) / checkpoints["dit"])}))
    shutil.copyfile(runtime["dependency_map"], CASE / "h3-dependencies.json")
    shutil.copyfile(ROOT / "tools/h3-exp-python-step.py", CASE / "python_worker.py")
    shutil.copyfile("/home/thhel/git/comfyui-helto-director/shared/h3/runtime/attention_window.py", CASE / "attention_window.py")
    product = (ROOT / "inference/src/h3_product/native.rs").read_text()
    product = replace_once(product, "fn observe(&mut self, name: &str, value: &Tensor) -> NResult<()> {", "fn observe(&mut self, name: &str, value: &Tensor) -> NResult<()> {\n        probe_observe(name, value)?;")
    product += "\n" + (ROOT / "tools/h3-exp-step-probe.rs").read_text()
    (diagnostic_source / "inference/src/h3_product/exp_step_probe.rs").write_text(product)
    module = diagnostic_source / "inference/src/h3_product/mod.rs"
    module.write_text(module.read_text() + "\npub mod exp_step_probe;\n")
    (diagnostic_source / "inference/src/bin/h3-exp-step-probe.rs").write_text('fn main() { if let Err(e)=imagine_inference::h3_product::exp_step_probe::probe(){ eprintln!("OWN_EXP_PROBE_FAILED {e}");std::process::exit(1);}}\n')
    sampling = diagnostic_source / "inference/src/models/minimax_h3/sampling.rs"
    text = sampling.read_text()
    text = replace_once(text, "    let (latent, shapes) = request.latent.pack()?;", """    let (latent, shapes) = request.latent.pack()?;
    use crate::h3_product::exp_step_probe::dump;
    for (index,value) in request.latent.unbind().borrow().iter().enumerate() { dump(&format!("input-latent-{index}"), &value.as_tensor()?)?; }
    for (index,value) in request.noise.unbind().borrow().iter().enumerate() { dump(&format!("input-noise-{index}"), &value.as_tensor()?)?; }
    dump("input-context", request.context)?;
    dump("input-tags",request.payload.text_tags.unwrap())?;
    dump("input-mask",request.denoise_mask.unwrap())?;
    dump("input-sigmas",request.sigmas)?;""")
    text = replace_once(text, "        let output = unsafe {\n            self.model.forward(", """        crate::h3_product::exp_step_probe::dump("step1-input-video", &streams[0].as_tensor()?)?;
        crate::h3_product::exp_step_probe::dump("step1-input-audio", &streams[1].as_tensor()?)?;
        crate::h3_product::exp_step_probe::dump("step1-input-context", &context)?;
        crate::h3_product::exp_step_probe::dump("step1-input-timestep", &t)?;
        if let Some(mask)=&v { crate::h3_product::exp_step_probe::dump("step1-denoise_mask",mask)?; }
        if let Some(mask)=&a { crate::h3_product::exp_step_probe::dump("step1-audio_denoise_mask",mask)?; }
        let output = unsafe {
            self.model.forward(""")
    sampling.write_text(text)
    subprocess.run(["cargo", "build", "--manifest-path", str(diagnostic_source / "inference/Cargo.toml"), "--target-dir", str(native.TARGET), "--locked", "--offline", "--bin", "h3-exp-step-probe"], cwd=ROOT, env=env, check=True)
    binary = CASE / "artifacts/h3-exp-step-probe"
    shutil.copy2(native.TARGET / "debug/h3-exp-step-probe", binary)
    (CASE / "build.json").write_text(json.dumps({"activated": False, "instrumentation_only": True, "base_overlay": overlay, "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "native_source_sha256": hashlib.sha256(product.encode()).hexdigest(), "sampling_source_sha256": hashlib.sha256(text.encode()).hexdigest()}, indent=2))
    print(CASE, flush=True)


def run(python_only=False):
    native_tools().environment()  # Verified encrypted mount, no writes outside own paths.
    assert CASE.resolve() == CASE and (CASE / "build.json").is_file()
    status = CASE / ("run-status-python-correct-mask.json" if python_only else "run-status.json")
    assert not status.exists()
    if python_only:
        assert json.loads((CASE / "run-status.json").read_text())["completed"]
        assert json.loads((CASE / "rust-result.json").read_text())["expected_first_step_stop"]
        (CASE / "python-correct-mask").mkdir()
        shutil.copyfile(ROOT / "tools/h3-exp-python-step.py", CASE / "python_worker_correct_mask.py")
    sys.path.insert(0, str(ROOT / "inference/tools"))
    from h3_gpu_commands import original_command, PYTHON_ROOT, CANONICAL_SOCKET
    controls = PRIOR / "components-20261020-6f1db108/build-target/debug"
    runtime = json.loads((OWN / "h3-e2e-08/runtime.json").read_text())["h3"]
    identity = dict(runtime["identity"], worker_sha256=json.loads((CASE / "build.json").read_text())["binary_sha256"])
    env = dict(os.environ, **runtime["environment"], IMAGINE_H3_EXP_STEP=str(CASE), IMAGINE_H3_MODEL_ROOT=runtime["model_root"], IMAGINE_H3_DEPENDENCY_MAP=str(CASE / "h3-dependencies.json"), IMAGINE_H3_DEPLOYMENT=json.dumps(identity), CUDA_VISIBLE_DEVICES="0", TMPDIR=str(CASE / "tmp"), PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for key, folder in [("XDG_CACHE_HOME", "cache"), ("TORCH_HOME", "torch"), ("HF_HOME", "hf"), ("TRITON_CACHE_DIR", "triton"), ("CUDA_CACHE_PATH", "cuda")]:
        (CASE / folder).mkdir(exist_ok=True)
        env[key] = str(CASE / folder)
    record = {"variants": [], "completed": False, "first_step_only": True, "rust_inputs_reused": python_only, "prior_python_packed_mask_invalid": python_only}
    for engine in (["python"] if python_only else ["rust", "python"]):
        suffix = "-correct-mask" if python_only else ""
        manifest = CASE / (engine + suffix + "-reservation.json")
        assert not manifest.exists()
        command = [str(controls / "imagine-native-gpu-reservation"), "--socket", str(CANONICAL_SOCKET), "--job", "own-exp-step-01-" + engine + suffix, "--bytes", "1", "--timeout", "1800", "--manifest", str(manifest)]
        if engine == "rust":
            command += ["--", str(CASE / "artifacts/h3-exp-step-probe")]
        else:
            model_path = Path(json.loads((CASE / "model-path.json").read_text())["path"])
            worker = CASE / ("python_worker_correct_mask.py" if python_only else "python_worker.py")
            args = original_command(ROOT, Path("/home/thhel/git/ComfyUI/.venv/lib/python3.12/site-packages"), CASE, [worker], [(model_path, model_path)])
            args += ["--setenv", "TMPDIR", str(CASE / "tmp"), "--setenv", "CUDA_VISIBLE_DEVICES", "0", "--setenv", "IMAGINE_H3_EXP_STEP", str(CASE), "--setenv", "IMAGINE_H3_EXP_OUTPUT", "python-correct-mask" if python_only else "python", "--chdir", str(CASE), "--", str(PYTHON_ROOT / "bin/python"), "-S", "-B", str(worker)]
            command += ["--process-form", "bwrap-pid-namespace", "--namespace-entry", str(controls / "namespace_entry"), "--", "/usr/bin/bwrap", *args]
        (CASE / (engine + suffix + "-command.json")).write_text(json.dumps(command, indent=2))
        print("BEGIN", engine, flush=True)
        started = time.monotonic()
        with (CASE / (engine + suffix + ".log")).open("x") as log:
            outcome = subprocess.run(command, env=env, cwd=CASE, stdout=log, stderr=subprocess.STDOUT)
        row = {"engine": engine, "exit": outcome.returncode, "seconds_with_diagnostics": time.monotonic() - started}
        if manifest.exists():
            row["reservation"] = json.loads(manifest.read_text())["report"]
        record["variants"].append(row)
        status.write_text(json.dumps(record, indent=2))
        print("END", json.dumps(row), flush=True)
        assert outcome.returncode == 0, "own " + engine + " probe failed"
        report = row["reservation"]
        if engine == "python":
            assert report["namespace_exited"]
            report = report["worker"]
        assert report["child_reaped"] and report["finish_sent"] and report["state"] == "completed"
    record["completed"] = True
    status.write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run", "run-python-correct-mask"])
    options = parser.parse_args()
    prepare() if options.mode == "prepare" else run(options.mode == "run-python-correct-mask")
