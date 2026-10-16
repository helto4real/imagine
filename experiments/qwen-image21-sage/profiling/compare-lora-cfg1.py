"""Compare a single LoRA with no LoRA on frozen Qwen INT8, CFG 1.

By default only client timings, HTTP status and GPU counters are observed.
Optional profiling collects kernel/copy timings with explicit user permission.
Native logs and response bodies (including images) are discarded, never inspected.
Dry-run by default; no existing service is contacted or stopped.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import statistics
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
BASELINE = ROOT / "baselines/1600x896-v2"
PROMPT = (
    "A photographic portrait of an adult woman wearing a blue cotton shirt, "
    "neutral expression, looking at the camera, soft natural window light, "
    "realistic skin texture, simple light gray background, waist-up composition."
)


def gpu():
    values = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip().splitlines()
    if len(values) != 1:
        raise RuntimeError("This benchmark requires one GPU")
    free, utilization = (float(v.strip()) for v in values[0].split(","))
    return {"free_mib": free, "utilization_percent": utilization}


def request(port, body=None, timeout=900):
    command = [
        "curl", "--silent", "--show-error", "--noproxy", "*",
        "--max-time", str(timeout), "--output", os.devnull,
        "--write-out", "%{http_code}",
    ]
    if body is None:
        command += [f"http://127.0.0.1:{port}/v1/models"]
        data = None
    else:
        command += ["--header", "Content-Type: application/json", "--data-binary", "@-",
                    f"http://127.0.0.1:{port}/v1/images/generations"]
        data = json.dumps(body).encode()
    start = time.perf_counter()
    result = subprocess.run(command, input=data, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, timeout=timeout + 5)
    elapsed = time.perf_counter() - start
    status = result.stdout.decode("ascii").strip()
    if result.returncode != 0 or status != "200":
        raise RuntimeError(f"HTTP request failed: curl={result.returncode}, status={status}")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--lora", required=True, type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--trace-library", type=Path,
                        help="Opt-in kernel/copy timings only; requires permission to read diagnostic runtime data")
    args = parser.parse_args()
    manifest = json.loads((BASELINE / "manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        if hashlib.sha256((BASELINE / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen source changed: {name}")
    binary = BASELINE / "sd-server"
    if hashlib.sha256(binary.read_bytes()).hexdigest() != manifest["binary_sha256"]:
        raise RuntimeError("Frozen binary hash mismatch")
    lora = args.lora.resolve(strict=True)
    if not lora.is_file():
        raise RuntimeError("LoRA is not a file")
    command = shlex.split((BASELINE / "command.sh").read_text())
    command[0] = str(binary)
    for option, relative in manifest["models"].items():
        path = args.model_root.resolve() / relative
        if not path.is_file():
            raise RuntimeError(f"Missing model component: {relative}")
        command[command.index(option) + 1] = str(path)
    command[command.index("--cfg-scale") + 1] = "1"
    command[command.index("--log-level") + 1] = "error"
    if any(os.environ.get(k) for k in ("LD_PRELOAD", "QWEN_ACTIVITY_TRACE", "QWEN_COPY_STACKS", "QWEN_ACTIVITY_FLUSH")):
        raise RuntimeError("Profiling injection must be disabled")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SD_", "GGML_"))}
    env.update(json.loads((BASELINE / "baseline-environment.json").read_text()))
    library = args.trace_library.resolve(strict=True) if args.trace_library else None
    plan = {
        "baseline": manifest["id"], "binary_sha256": manifest["binary_sha256"],
        "width": 1600, "height": 896, "steps": 40, "cfg": 1, "seed": 42,
        "sampler": "er_sde", "scheduler": "beta", "references": 0,
        "prompt": PROMPT, "negative_prompt": "", "lora": str(lora),
        "lora_bytes": lora.stat().st_size, "strength": 1.0,
        "measurement": "client wall time through synchronous HTTP response discarded to /dev/null",
        "trace_library": str(library) if library else None,
        "sequence": ["without_first", "without_warm_1", "without_warm_2",
                     "with_first", "with_warm_1", "with_warm_2",
                     "without_return_first", "without_return_warm"],
    }
    if not args.run:
        print(json.dumps(plan, indent=2))
        return
    stable = 0
    for _ in range(120):
        state = gpu()
        stable = stable + 1 if state["free_mib"] >= 13000 and state["utilization_percent"] < 20 else 0
        if stable >= 4:
            break
        time.sleep(0.5)
    else:
        raise RuntimeError("GPU is not idle with enough free memory; no service was stopped")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command += ["--listen-ip", "127.0.0.1", "--listen-port", str(port),
                "--conditioning-cache-size", "4", "--disable-image-metadata"]
    (ROOT / "runs").mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="lora-cfg1-", dir=ROOT / "runs"))
    if library:
        env.update(LD_PRELOAD=str(library), QWEN_ACTIVITY_TRACE=str(out / "activity.tsv"),
                   QWEN_ACTIVITY_FLUSH=str(out / "flush.request"))
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"RESULTS {out}", flush=True)
    results = []
    proc = None
    try:
        start = time.perf_counter()
        proc = subprocess.Popen(command, cwd=PROJECT, env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        while True:
            if proc.poll() is not None or time.perf_counter() - start > 180:
                raise RuntimeError("Test server failed to become ready; output was discarded")
            try:
                request(port, timeout=1)
                break
            except RuntimeError:
                time.sleep(0.2)
        startup = time.perf_counter() - start
        print(f"READY startup_seconds={startup:.3f}", flush=True)
        for name in plan["sequence"]:
            enabled = name.startswith("with_")
            extra = {"negative_prompt": "", "seed": 42,
                     "lora": [{"path": str(lora), "multiplier": 1.0}] if enabled else []}
            body = {"prompt": PROMPT + "<sd_cpp_extra_args>" + json.dumps(extra) + "</sd_cpp_extra_args>",
                    "size": "1600x896", "n": 1, "output_format": "png"}
            before = gpu()
            print(f"START {name}", flush=True)
            start_ns = time.monotonic_ns()
            elapsed = request(port, body)
            end_ns = time.monotonic_ns()
            results.append({"case": name, "seconds": elapsed, "http_status": 200, "gpu_before": before,
                            "start_ns": start_ns, "end_ns": end_ns})
            (out / "timings.json").write_text(json.dumps({"startup_seconds": startup, "runs": results}, indent=2) + "\n")
            print(f"DONE {name} seconds={elapsed:.3f}", flush=True)
        without = statistics.mean(r["seconds"] for r in results if r["case"] in ("without_warm_1", "without_warm_2"))
        with_lora = statistics.mean(r["seconds"] for r in results if r["case"] in ("with_warm_1", "with_warm_2"))
        summary = {"without_warm_mean_seconds": without, "with_warm_mean_seconds": with_lora,
                   "ratio": with_lora / without, "runs": results, "startup_seconds": startup}
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"SUMMARY without={without:.3f} with={with_lora:.3f} ratio={with_lora / without:.3f}", flush=True)
        if library:
            checkpoint = out / "flush.request"
            checkpoint.touch()
            deadline = time.monotonic() + 30
            while checkpoint.exists():
                if proc.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("Profiler checkpoint did not complete")
                time.sleep(0.1)
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


if __name__ == "__main__":
    main()
