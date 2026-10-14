"""Replay one warmup and three blocks of CPU-thread comparisons, without tracing."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile

from PIL import Image

experiment = Path(__file__).resolve().parents[1]
binary = experiment / "runs/projection-fusion-build/sd-cli"
expected_binary = "33aa286e9ff42e4b6e61af134bed5a1fa64b1d7d98ce32d27ea486ff7d225805"
expected_pixels = "253d91ada4f98feb50bc9139619d00526594123006d37a9b4589e15cfe43a7f2"
assert hashlib.sha256(binary.read_bytes()).hexdigest() == expected_binary
out = Path(tempfile.mkdtemp(prefix=datetime.datetime.now(datetime.timezone.utc).strftime(
    "%Y%m%dT%H%M%SZ-cpu-threads-"), dir=experiment / "runs"))
shutil.copyfile(__file__, out / "compare-cpu-threads.py")
shutil.copyfile(Path(__file__).with_name("run.py"), out / "replay-run.py")
orders = [[11, 4, 8, 16], [16, 8, 4, 11], [8, 11, 16, 4]]
env = {**os.environ, "GGML_CUDA_FUSED_INT8_GATE_UP": "1", "GGML_CUDA_FUSED_INT8_PROJECTION": "1"}
assert not any(env.get(k) for k in ("LD_PRELOAD", "QWEN_ACTIVITY_TRACE", "QWEN_COPY_STACKS"))
(out / "plan.json").write_text(json.dumps({
    "warmup_threads": 11, "measured_orders": orders, "binary": str(binary),
    "binary_sha256": expected_binary, "expected_rgb_sha256": expected_pixels,
    "cpu_affinity": sorted(os.sched_getaffinity(0)),
    "thread_environment": {k: v for k, v in env.items() if k.startswith(("OMP_", "MKL_", "OPENBLAS_", "GGML_"))},
    "timing": "CLI process start to successful save log receipt; wall time also includes process exit. Loader timings overlap conditioning, sampling, and VAE totals; do not add them.",
}, indent=2) + "\n")
for name, command in [("cpu.txt", ["lscpu"]), ("gpu.txt", ["nvidia-smi"] )]:
    (out / name).write_text(subprocess.check_output(command, text=True))
print(out, flush=True)
results = []
sequence = [(0, 11)] + [(block, threads) for block, order in enumerate(orders, 1) for threads in order]
for index, (block, threads) in enumerate(sequence):
    print(f"START {index + 1}/{len(sequence)} block={block} threads={threads}", flush=True)
    command = [sys.executable, str(experiment / "profiling/run.py"), "--binary", str(binary), "--threads", str(threads)]
    process = subprocess.run(command, env=env, text=True, capture_output=True)
    (out / f"replay-{index:02d}.log").write_text(process.stdout + process.stderr)
    process.check_returncode()
    item = json.loads(process.stdout.splitlines()[-1])
    run = Path(item["run"])
    log = (run / "generation.log").read_text()
    assert int(re.search(r"n_threads: (\d+)", log)[1]) == threads
    assert "Using SageAttention in the diffusion model" in log
    result = json.loads((run / "result.json").read_text())
    stamps = [json.loads(line) for line in (run / "log-timestamps.jsonl").read_text().splitlines()]
    saved = next(s for s in stamps if "save result image 0" in s["line"] and "(success)" in s["line"])
    image = Image.open(run / "image.png").convert("RGB")
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    validation = {"mode": "RGB", "size": image.size, "sha256": digest,
                  "bit_identical_reference": digest == expected_pixels}
    (run / "image-validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    def seconds(pattern):
        return float(re.search(pattern, log)[1])
    item.update(block=block, warmup=block == 0, threads=threads, image=validation,
        saved_seconds=(saved["monotonic_ns"] - result["start_ns"]) / 1e9,
        sampling_seconds=seconds(r"sampling completed, taking ([\d.]+)s"),
        conditioning_seconds=seconds(r"get_learned_condition completed, taking ([\d.]+)s"),
        vae_seconds=seconds(r"decode_first_stage completed, taking ([\d.]+)s"),
        engine_seconds=seconds(r"generate_image completed in ([\d.]+)s"),
        tensor_load_seconds=[float(v) for v in re.findall(r"loading tensors completed, taking ([\d.]+)s", log)])
    results.append(item)
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(item), flush=True)
    assert image.size == (1600, 896) and digest == expected_pixels, "Output differs: saved evidence; stopping comparison"

summary = {}
for threads in [4, 8, 11, 16]:
    rows = [r for r in results if not r["warmup"] and r["threads"] == threads]
    summary[threads] = {key: {"mean": statistics.mean(r[key] for r in rows),
        "median": statistics.median(r[key] for r in rows), "min": min(r[key] for r in rows),
        "max": max(r[key] for r in rows)} for key in
        ["saved_seconds", "wall_seconds", "sampling_seconds", "conditioning_seconds", "vae_seconds"]}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("COMPLETE " + str(out), flush=True)
