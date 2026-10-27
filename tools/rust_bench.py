#!/usr/bin/env python3
"""Old-versus-new regression bench for the Rust image workers, through the proxy's MCP.

Runs fixed synthetic workloads with fixed seeds against whatever server is running
and records, per generation: client wall time, peak GPU memory (device and worker
processes), peak worker RSS and the output image hashes. The first run of each
workload follows a model switch and includes loading; later runs are warm.

Raw results and the bench's own images go to comfyui/bench/<label>-<time>/ (ignored
by Git). Only the bench's synthetic jobs are created and read; no job history,
logs or other runtime data are accessed.

    python3 tools/rust_bench.py --label control
    python3 tools/rust_bench.py --label candidate --workloads moss-t2i moss-sheet
"""
import argparse
import hashlib
import json
import statistics
import subprocess
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Image and H3 workers, and the native Qwen server; /proc comm is cut to 15 bytes.
WORKER_NAMES = {"imagine-worker", "imagine-h3-work", "sd-server"}

# Synthetic prompts only. Seeds are fixed so that runs are comparable across builds.
WORKLOADS = {
    "moss-t2i": {"profile": "moss", "parameters": {
        "prompt": "A red ceramic teapot on a wooden table beside a window, soft morning light, photograph.",
        "width": 1024, "height": 1024, "seed": 1001}},
    "moss-sheet": {"profile": "moss-character-sheet", "parameters": {
        "character_sheet": {
            "character_description": "A man in his late sixties, tall and lean, weathered tanned face with deep lines, short white beard, pale grey eyes, close-cropped grey hair",
            "clothing_description": "dark navy wool fisherman's sweater, black knit watch cap, heavy grey canvas trousers, worn brown leather boots"},
        "seed": 1002}},
    "qwen-t2i": {"profile": "qwen", "parameters": {
        "prompt": "A lighthouse on a rocky coast at dusk, waves breaking on the rocks, photograph.",
        "seed": 1003}},
    "qwen-edit": {"profile": "qwen", "reference": "qwen-t2i", "parameters": {
        "prompt": "Make it a snowy winter evening with warm light in the lighthouse windows.",
        "seed": 1004}},
}


class Mcp:
    def __init__(self, url):
        self.url = url

    def call(self, name, arguments, timeout=3600):
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                           "params": {"name": name, "arguments": arguments}}).encode()
        request = urllib.request.Request(self.url, body, {
            "Content-Type": "application/json", "Accept": "application/json, text/event-stream"})
        text = urllib.request.urlopen(request, timeout=timeout).read().decode()
        data = next(line[6:] for line in text.splitlines() if line.startswith("data: "))
        result = json.loads(data)["result"]
        return result.get("structuredContent", result)


class Sampler:
    """Peak GPU memory (device and worker processes) and worker RSS while running."""

    def __init__(self):
        self.device = []
        self.workers_gpu = {}
        self.rss = {}
        self.stop = threading.Event()

    def __enter__(self):
        self.streams = [
            subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                              "-lms", "100"], stdout=subprocess.PIPE, text=True),
            subprocess.Popen(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits", "-lms", "200"], stdout=subprocess.PIPE, text=True),
        ]
        self.threads = [threading.Thread(target=self.read_device, daemon=True),
                        threading.Thread(target=self.read_apps, daemon=True),
                        threading.Thread(target=self.read_rss, daemon=True)]
        for thread in self.threads:
            thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        for stream in self.streams:
            stream.terminate()
            stream.wait()
        for thread in self.threads:
            thread.join(timeout=2)

    def read_device(self):
        for line in self.streams[0].stdout:
            if line.strip().isdigit():
                self.device.append(int(line))

    def read_apps(self):
        for line in self.streams[1].stdout:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit() and worker(int(parts[0])):
                pid, mib = int(parts[0]), int(parts[1])
                self.workers_gpu[pid] = max(self.workers_gpu.get(pid, 0), mib)

    def read_rss(self):
        while not self.stop.is_set():
            for proc in Path("/proc").iterdir():
                if proc.name.isdigit() and worker(int(proc.name)):
                    try:
                        status = (proc / "status").read_text()
                    except OSError:
                        continue
                    kib = next((int(l.split()[1]) for l in status.splitlines() if l.startswith("VmRSS:")), 0)
                    self.rss[proc.name] = max(self.rss.get(proc.name, 0), kib)
            self.stop.wait(0.1)

    def summary(self):
        return {"device_gpu_mib": {"before": self.device[0] if self.device else None,
                                   "peak": max(self.device, default=None)},
                "worker_gpu_peak_mib": max(self.workers_gpu.values(), default=None),
                "worker_rss_peak_mib": round(max(self.rss.values(), default=0) / 1024) or None}


def worker(pid):
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip() in WORKER_NAMES
    except OSError:
        return False


def server_release():
    for proc in Path("/proc").iterdir():
        try:
            if proc.name.isdigit() and (proc / "comm").read_text().strip() == "imagine-server":
                return str(Path((proc / "exe").readlink()).parent)
        except OSError:
            continue
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True, help="control or candidate, recorded with the results")
    parser.add_argument("--runs", type=int, default=4, help="runs per workload; the first follows a model switch")
    parser.add_argument("--workloads", nargs="+", default=list(WORKLOADS), choices=list(WORKLOADS))
    parser.add_argument("--mcp", default="http://127.0.0.1:8190/mcp")
    args = parser.parse_args()
    out = ROOT / "comfyui/bench" / f"{args.label}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    (out / "images").mkdir(parents=True)
    mcp = Mcp(args.mcp)
    key = mcp.call("workspace_create", {"title": f"Rust bench {args.label}"})["workspace_key"]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (out / "run.json").write_text(json.dumps({
        "label": args.label, "server_release": server_release(), "git_head": commit,
        "runs": args.runs, "workloads": args.workloads, "workspace_key": key}, indent=2) + "\n")
    outputs = {}
    results = []
    for name in args.workloads:
        spec = WORKLOADS[name]
        for run in range(args.runs):
            arguments = {"workspace_key": key, "job_id": f"bench-{out.name}-{name}-{run}",
                         "profile": spec["profile"], "parameters": spec["parameters"]}
            if "reference" in spec:
                file_id = outputs[spec["reference"]]
                image = mcp.call("register_image", {"workspace_key": key, "profile": spec["profile"],
                                                    "file_id": file_id})["image_id"]
                arguments["references"] = [image]
            with Sampler() as sampler:
                start = time.monotonic()
                result = mcp.call("generate", arguments)
                wall = time.monotonic() - start
            row = {"workload": name, "run": run, "phase": "cold" if run == 0 else "warm",
                   "status": result.get("status"), "wall_s": round(wall, 3), **sampler.summary()}
            if result.get("status") != "succeeded":
                row["error"] = result.get("error")
            else:
                images = []
                for i, image in enumerate(result["images"]):
                    url = image["download_url"]
                    url = url if url.startswith("http") else args.mcp.rsplit("/mcp", 1)[0] + url
                    data = urllib.request.urlopen(urllib.request.Request(
                        url, headers={"Authorization": f"Bearer {key}"}), timeout=600).read()
                    path = out / "images" / f"{name}-{run}-{i}.png"
                    path.write_bytes(data)
                    images.append(hashlib.sha256(data).hexdigest())
                outputs.setdefault(name, result["images"][0]["file_id"])
                row.update(seed=result["parameters"].get("seed"), model=result["parameters"].get("model"),
                           images_sha256=images)
            results.append(row)
            print(json.dumps(row), flush=True)
            with (out / "results.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
    summary = {}
    for name in args.workloads:
        rows = [r for r in results if r["workload"] == name and r["status"] == "succeeded"]
        warm = [r for r in rows if r["phase"] == "warm"]
        summary[name] = {
            "cold_s": next((r["wall_s"] for r in rows if r["phase"] == "cold"), None),
            "warm_median_s": statistics.median(r["wall_s"] for r in warm) if warm else None,
            "worker_gpu_peak_mib": max((r["worker_gpu_peak_mib"] or 0 for r in rows), default=None),
            "device_gpu_peak_mib": max((r["device_gpu_mib"]["peak"] or 0 for r in rows), default=None),
            "worker_rss_peak_mib": max((r["worker_rss_peak_mib"] or 0 for r in rows), default=None),
            "deterministic": len({tuple(r["images_sha256"]) for r in rows}) == 1 if rows else None,
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(out)


if __name__ == "__main__":
    main()
