#!/usr/bin/env python3
"""Compile H3-only changes on the frozen core into a permanently isolated candidate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import io
import uuid

ROOT = Path(__file__).resolve().parents[1]
BASE = "cd0ef53be69dd539ef0ffdf0cb03f4c1c5835fc7"
CANDIDATE = ROOT / "comfyui/h3-video-night-20261021"
SOURCE = CANDIDATE / "native-source"
TARGET = CANDIDATE / "target-native"
OWN = Path("/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e")
OVERLAY = ["shared/h3_protocol.rs", "shared/h3_geometry.rs", "shared/h3_sampling.rs", "shared/h3_continuation.rs", "shared/h3_references.rs", "design/h3-i2v-model-lock.json", "inference/src/models/minimax_h3/sampling.rs", "inference/examples/h3_stage_cpu.rs", "inference/examples/h3_motion_cpu.rs", "inference/examples/h3_reference_cpu.rs", "inference/examples/h3_reference_window_cpu.rs", "inference/examples/h3_reference_worker_probe.rs"] + [f"inference/src/h3_product/{name}.rs" for name in ["mod", "native", "deployment", "wire", "image", "sampling", "motion", "references", "reference_video", "reference_window"]]


def environment():
    mount = json.loads(subprocess.check_output(["findmnt", "-J", "-T", str(OWN)]))["filesystems"][0]
    if (mount["target"], mount["source"], mount["fstype"]) != ("/run/media/veracrypt1", "/dev/mapper/veracrypt1", "ext3") or os.statvfs(OWN).f_flag & os.ST_RDONLY:
        raise ValueError("The own encrypted destination is unavailable or read-only.")
    for path in [CANDIDATE, SOURCE, TARGET, CANDIDATE / "cargo-native", OWN / "native-build"]:
        if path.resolve() != path:
            raise ValueError("Own build paths must not be symlinks.")
    state = OWN / "native-build"
    state.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(CARGO_HOME=str(CANDIDATE / "cargo-native"), CARGO_TARGET_DIR=str(TARGET), CARGO_INCREMENTAL="0", CARGO_BUILD_JOBS="2", CARGO_PROFILE_DEV_DEBUG="0", CARGO_PROFILE_TEST_DEBUG="0", RUSTFLAGS="-C debuginfo=0", LIBTORCH="/home/thhel/git/ComfyUI/.venv/lib/python3.12/site-packages/torch", LIBTORCH_CXX11_ABI="1", CXX="clang++", IMAGINE_CUDA_INCLUDE=str(ROOT / "comfyui/moss-20261018/cuda13/nvidia/cu13/include"), LD_LIBRARY_PATH="/home/thhel/git/ComfyUI/.venv/lib/python3.12/site-packages/torch/lib", TMPDIR=str(OWN / "tmp"), CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for key, directory in [("XDG_CACHE_HOME", "cache"), ("TORCH_HOME", "torch"), ("TRITON_CACHE_DIR", "triton"), ("CUDA_CACHE_PATH", "cuda"), ("HF_HOME", "hf")]:
        env[key] = str(state / directory)
        (state / directory).mkdir(exist_ok=True)
    for key in ["LIBTORCH_USE_PYTORCH", "LIBTORCH_BYPASS_VERSION_CHECK"]:
        env.pop(key, None)
    return env


def prepare():
    if not SOURCE.exists():
        SOURCE.mkdir(parents=True)
        archive = subprocess.check_output(["git", "archive", BASE, "inference", "shared", "design/h3-stage4-evidence/selected-model-sha256.json"], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(archive)) as files:
            files.extractall(SOURCE, filter="data")
    # Copy dependency source/cache only, never credentials or old target binaries.
    registry = CANDIDATE / "cargo-native/registry"
    if not registry.exists():
        cached = Path("/home/thhel/.codex/worktrees/h3-product-worker/imagine/comfyui/h3-product-20261020/cargo/registry")
        shutil.copytree(cached, registry)
    records = []
    for name in OVERLAY:
        destination = SOURCE / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
        records.append({"path": name, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["build", "cpu", "clippy", "wire"])
    args = parser.parse_args()
    env = environment()
    overlay = prepare()
    if args.mode == "wire":
        target = CANDIDATE / "target-native-wire"
        if target.resolve() != target:
            raise ValueError("Own wire-test target must not be a symlink.")
        env["CARGO_TARGET_DIR"] = str(target)
        subprocess.run(["cargo", "test", "--manifest-path", str(ROOT / "inference/tools/h3-product-cpu/Cargo.toml"), "--target-dir", str(target), "--locked", "--offline"], cwd=ROOT, env=env, check=True)
        return
    command = ["cargo", "clippy" if args.mode == "clippy" else "build", "--manifest-path", str(SOURCE / "inference/Cargo.toml"), "--target-dir", str(TARGET), "--locked", "--offline", "--bin", "imagine-h3-worker", "--example", "h3_stage_cpu", "--example", "h3_motion_cpu", "--example", "h3_reference_cpu", "--example", "h3_reference_window_cpu", "--example", "h3_reference_worker_probe"]
    if args.mode == "clippy":
        command += ["--", "-D", "warnings"]
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    if args.mode == "cpu":
        subprocess.run([str(TARGET / "debug/examples/h3_stage_cpu")], cwd=ROOT, env=env, check=True)
    if args.mode == "build":
        package = CANDIDATE / "native-packages" / uuid.uuid4().hex
        package.mkdir(parents=True)
        binary = package / "imagine-h3-worker"
        shutil.copy2(TARGET / "debug/imagine-h3-worker", binary)
        (package / "candidate.json").write_text(json.dumps({"frozen_base": BASE, "overlay": overlay, "binary_sha256": hashlib.file_digest(binary.open("rb"), "sha256").hexdigest(), "activated": False}, indent=2) + "\n")
        print(package)


if __name__ == "__main__":
    main()
