"""Replay the saved F32 debug reference with optional CUPTI activity tracing."""
import argparse
import datetime
import json
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import time

parser = argparse.ArgumentParser()
parser.add_argument("--trace-library", type=Path)
parser.add_argument("--binary", type=Path)
parser.add_argument("--threads", type=int)
parser.add_argument("--copy-stacks", action="store_true")
parser.add_argument("--default-vram", action="store_true", help="Omit the saved --max-vram override; keep auto-fit enabled")
parser.add_argument("--vram-margin-mib", type=int, help="Override the saved margin for a separate memory diagnostic")
args = parser.parse_args()
if args.vram_margin_mib is not None and (args.vram_margin_mib <= 0 or args.default_vram):
    parser.error("--vram-margin-mib must be positive and cannot be combined with --default-vram")
if args.copy_stacks and not args.trace_library:
    parser.error("--copy-stacks requires --trace-library")
if args.threads is not None and args.threads < 1:
    parser.error("--threads must be positive")
experiment = Path(__file__).resolve().parents[1]
reference = experiment / "runs/20261013T074331Z-debug-7nor7cre"
label = "activity" if args.trace_library else "control"
run = Path(tempfile.mkdtemp(prefix=datetime.datetime.now(datetime.timezone.utc).strftime(
    "%Y%m%dT%H%M%SZ-") + label + "-", dir=experiment / "runs"))
command = shlex.split((reference / "command.sh").read_text())
if args.default_vram:
    index = command.index("--max-vram")
    del command[index:index + 2]
if args.vram_margin_mib is not None:
    command[command.index("--max-vram") + 1] = str(-args.vram_margin_mib / 1024)
if args.binary:
    command[0] = str(args.binary.resolve())
if args.threads is not None:
    command += ["--threads", str(args.threads)]
for flag, filename in [("--prompt-file", "prompt.txt"),
                       ("--negative-prompt-file", "negative-prompt.txt"),
                       ("--output", "image.png")]:
    if filename != "image.png":
        shutil.copyfile(reference / filename, run / filename)
    command[command.index(flag) + 1] = str(run / filename)
environment = os.environ.copy()
if args.trace_library:
    library = args.trace_library.resolve()
    environment["LD_PRELOAD"] = str(library)
    environment["QWEN_ACTIVITY_TRACE"] = str(run / "activity.tsv")
    if args.copy_stacks:
        environment["QWEN_COPY_STACKS"] = str(run / "copy-stacks.txt")
    shutil.copyfile(Path(__file__).with_name("activity.cpp"), run / "activity.cpp")
    shutil.copyfile(library, run / "activity.so")
(run / "command.sh").write_text(shlex.join(command) + "\n")
(run / "trace-environment.json").write_text(json.dumps({
    key: environment[key] for key in ("LD_PRELOAD", "QWEN_ACTIVITY_TRACE", "QWEN_COPY_STACKS", "GGML_CUDA_FUSED_INT8_GATE_UP", "GGML_CUDA_FUSED_INT8_GATE_UP_SWIGLU", "GGML_CUDA_GATE_UP_DIRECT", "GGML_CUDA_FUSED_INT8_PROJECTION", "GGML_CUDA_CONVROT_THREADS", "GGML_CUDA_CONVROT_FUSED_AMAX", "GGML_CUDA_FUSED_SWIGLU_CONVROT", "GGML_CUDA_SWIGLU_CONVROT_VEC", "GGML_CUDA_GATE_UP_CONFIG", "GGML_CUDA_FUSED_INT8_DOWN", "GGML_CUDA_FUSED_RMS_ROPE", "GGML_CUDA_FUSED_NORM_MODULATION", "GGML_CUDA_FUSED_NORM_CONVROT", "GGML_CUDA_FUSED_DOWN_RESIDUAL", "GGML_CUDA_FUSED_ATTENTION_RESIDUAL", "GGML_CUDA_FUSED_QK_RMS_ROPE", "GGML_CUDA_FUSED_GATED_RESIDUAL", "SD_QWEN21_BF16_RESIDUAL", "SD_QWEN21_BF16_GATE_UP", "SD_QWEN21_BF16_ATTENTION", "SD_SAGE_DIRECT_V_CAST", "SD_SAGE_SPLIT_V", "SD_SAGE_SPLIT_K", "SD_MEMORY_TRACE") if key in environment
}, indent=2) + "\n")
with open(command[0], "rb") as binary:
    (run / "binary-sha256.txt").write_text(hashlib.file_digest(binary, "sha256").hexdigest() + "\n")
shutil.copyfile(__file__, run / "run.py")
print(run, flush=True)
with (run / "gpu.csv").open("w") as gpu, (run / "gpu-monitor.log").open("w") as errors:
    monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,memory.used,utilization.gpu,"
        "clocks.sm,clocks.mem,power.draw,temperature.gpu", "--format=csv", "-l", "1"],
        stdout=gpu, stderr=errors)
    try:
        start = time.monotonic_ns()
        with (run / "generation.log").open("wb") as log, (run / "log-timestamps.jsonl").open("w") as stamps:
            process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, bufsize=0)
            pending = b""
            while True:
                chunk = os.read(process.stdout.fileno(), 65536)
                now = time.monotonic_ns()
                if not chunk:
                    break
                log.write(chunk)
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    stamps.write(json.dumps({"monotonic_ns": now, "line": line.decode(errors="replace")}) + "\n")
            if pending:
                stamps.write(json.dumps({"monotonic_ns": now, "line": pending.decode(errors="replace")}) + "\n")
            code = process.wait()
        end = time.monotonic_ns()
        (run / "result.json").write_text(json.dumps({"exit_code": code,
            "wall_seconds": (end-start)/1e9, "start_ns": start, "end_ns": end,
            "phase_timestamps": "parent pipe receipt on CLOCK_MONOTONIC; approximate boundaries"}, indent=2) + "\n")
    finally:
        monitor.terminate()
        monitor.wait()
print(json.dumps({"run": str(run), "exit_code": code, "wall_seconds": (end-start)/1e9}), flush=True)
raise SystemExit(code)
