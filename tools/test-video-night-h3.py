#!/usr/bin/env python3
"""Explicit real-GPU acceptance in a fresh owned encrypted instance; never a live API."""
import argparse
import array
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import time
import traceback
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OWN = Path("/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e")
PREVIOUS = Path("/run/media/veracrypt1/imagine/.codex-h3-res20-compare-20261020-6f0b84c2")


def copy_own_motion_source(destination):
    source = OWN / "h3-e2e-06/saved-project"
    assert source.resolve() == source and destination.resolve() == destination
    files = []
    for path in source.rglob("*"):
        assert not path.is_symlink(), "own source must not contain symlinks"
        if path.is_file() and path.name != ".imagine-lock":
            assert path.stat().st_size <= 1024**3
            relative = path.relative_to(source)
            assert str(relative) == "project.json" or relative.parts[0] in {"takes", "media", "continuity"}
            files.append((path, relative))
        else:
            assert path.is_dir() or path.name == ".imagine-lock"
    assert 0 < len(files) <= 16
    records = []
    for path, relative in files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists()
        shutil.copyfile(path, target)
        digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
        assert hashlib.file_digest(target.open("rb"), "sha256").hexdigest() == digest
        records.append({"path": str(relative), "sha256": digest, "bytes": target.stat().st_size})
    return {"source": str(source), "destination": str(destination), "files": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--continuity", action="store_true", help="Generate a fresh source, accept it and drive an I2V target with its raw saved endpoint")
    parser.add_argument("--motion", action="store_true", help="Fresh source then both native Motion Context methods, with and without explicit paired audio")
    parser.add_argument("--reuse-motion-source", action="store_true", help="Copy only our fixed h3-e2e-06 source into this new case; generate the four targets without regenerating the source")
    options = parser.parse_args()
    if not re.fullmatch(r"h3-e2e-[0-9]{2,3}", options.case):
        parser.error("Use a new h3-e2e-NN owned case.")
    if options.reuse_motion_source and not options.motion:
        parser.error("The fixed own source can only be used with --motion.")
    mount = json.loads(subprocess.check_output(["findmnt", "-J", "-T", str(OWN)]))["filesystems"][0]
    assert (mount["target"], mount["source"], mount["fstype"]) == ("/run/media/veracrypt1", "/dev/mapper/veracrypt1", "ext3") and not os.statvfs(OWN).f_flag & os.ST_RDONLY
    case = OWN / options.case
    assert case.resolve() == case
    case.mkdir(exist_ok=False)
    for name in ["project", "state", "output", "tmp", "cache", "config", "share", "models", "saved-project", "evidence"]:
        (case / name).mkdir()
    package = Path((OWN / ("evidence/audio-trim-product-package.log" if options.reuse_motion_source else "evidence/motion-product-package2.log" if options.motion else "evidence/endpoint-product-package.log")).read_text().splitlines()[-1])
    native = Path((OWN / ("evidence/audio-trim-native-package-formatted.log" if options.reuse_motion_source else "evidence/motion-native-package2.log" if options.motion else "evidence/h3-stages-native-package.log")).read_text().splitlines()[-1])
    assert package.is_relative_to(ROOT / "comfyui/h3-video-night-20261021/packages") and native.is_relative_to(ROOT / "comfyui/h3-video-night-20261021/native-packages")
    for path in [package, native]:
        assert path.resolve() == path and json.loads((path / "candidate.json").read_text())["activated"] is False
    runtime = json.loads((PREVIOUS / "full-fp16-20261020-5e4a923b/candidate-runtime.json").read_text())
    h3 = runtime["h3"]
    h3["files"].pop(h3["binary"])
    h3["binary"] = str(native / "imagine-h3-worker")
    digest = hashlib.file_digest(Path(h3["binary"]).open("rb"), "sha256").hexdigest()
    h3["identity"].update(protocol=6 if options.motion else 5, worker_sha256=digest)
    h3["files"][h3["binary"]] = digest
    dependency = Path(h3["dependency_map"])
    assert dependency.is_relative_to(PREVIOUS)
    (case / "h3-dependencies.json").write_bytes(dependency.read_bytes())
    h3["dependency_map"] = str(case / "h3-dependencies.json")
    (case / "runtime.json").write_text(json.dumps(runtime, indent=2))
    destination = json.loads((OWN / "q22/evidence/endpoint.json").read_text())["destination"]
    destination["root"] = str(case / "output")
    (case / "state/video-output.json").write_text(json.dumps(destination))
    env = dict(os.environ, TMPDIR=str(case / "tmp"), XDG_CACHE_HOME=str(case / "cache"), XDG_CONFIG_HOME=str(case / "config"), XDG_DATA_HOME=str(case / "share"), QWEN_MODEL_ROOT=str(case / "models"), CUDA_VISIBLE_DEVICES="0", PYTHONDONTWRITEBYTECODE="1", IMAGINE_COORDINATOR_SOCKET=str(ROOT / ".imagine/gpu.sock"))
    for key, name in [("TORCH_HOME", "torch"), ("HF_HOME", "hf"), ("TRITON_CACHE_DIR", "triton"), ("CUDA_CACHE_PATH", "cuda")]:
        path = case / "cache" / name
        path.mkdir()
        env[key] = str(path)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    acceptance = {"frames": 124, "fps": 24, "duration": 31 / 6, "duration_tolerance": 0.002, "container": "mp4", "video_codec": "h264", "pixel_format": "yuv420p", "audio_codec": "aac", "sample_rate": 32000, "channels": 2, "color_space": "bt709", "color_range": "tv", "audio_rms_min": 0.0001, "native_worker_pid_after_each_job": None}
    if options.motion:
        assert json.loads((OWN / "evidence/motion-director-cpu-parity.json").read_text())["passed"]
        if options.reuse_motion_source:
            assert json.loads((OWN / "evidence/motion-audio-trim-director-cpu-parity.json").read_text())["passed"]
        acceptance["motion"] = {"tail_schema": 1, "video_steps": 7, "audio_steps": 37, "sampled_frames": 158, "context_head": 34, "exp_head": 21, "exp_tail": 13, "attention": "full16+fade16 output blend", "paired_audio": "explicit opt in", "source": "Accepted, frozen resource/hash/model/shape/version"}
    (case / "evidence/acceptance-before-run.json").write_text(json.dumps(acceptance, indent=2))
    result = {"case": str(case), "package": str(package), "native_package": str(native), "gpu_coordination": "real shared scheduler Ticket; no standalone fallback", "variants": [], "completed": False}
    if options.reuse_motion_source:
        result["source_copy"] = copy_own_motion_source(case / "saved-project")
        (case / "evidence/source-copy.json").write_text(json.dumps(result["source_copy"], indent=2))
    server = None
    api = None
    session = ""

    def request(method, route, value=None, raw=None, mime="application/json", key=None):
        headers = {"Content-Type": mime}
        if session:
            headers["X-Imagine-Session"] = session
        if key:
            headers["Idempotency-Key"] = key
        data = raw if raw is not None else json.dumps(value).encode() if value is not None else None
        req = urllib.request.Request(api + route, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"own {route} returned {error.code}: {error.read().decode()}") from error

    try:
        log_path = case / "evidence/server.log"
        with log_path.open("w") as log:
            server = subprocess.Popen([str(package / "imagine-server"), "--project", str(case / "project"), "--data", str(case / "state"), "--profiles", str(ROOT / "config/image/profiles"), "--bind", "127.0.0.1:0", "--backend", "rust", "--rust-runtime", str(case / "runtime.json")], env=env, stdout=log, stderr=subprocess.STDOUT)
        result["server_pid"] = server.pid
        deadline = time.monotonic() + 30
        while api is None:
            assert server.poll() is None, "own server exited before ready"
            found = re.search(r"Imagine listening on (http://127[.]0[.]0[.]1:[0-9]+)", log_path.read_text())
            if found:
                api = found.group(1)
                break
            assert time.monotonic() < deadline, "own startup timeout"
            time.sleep(0.05)
        result["api"] = api
        session = request("GET", "/v1/session")["id"]
        proof = request("GET", "/v1/video-destination")
        assert proof["config"]["root"] == str(case / "output")
        model = {"adapter": "h3", "model": "minimax-h3", "variant": "i2v", "profile": "h3-fl2va-i2va-v1", "profile_revision": "2"}
        stage = {"steps": 20, "sampler": "res_multistep", "scheduler": "simple", "denoise": 1.0, "sigmas": []}
        single = {"stage_one": stage, "stage_two": None, "base_steps_to_run": 4}
        location = request("POST", "/v1/video/locations", {"schema_version": 1, "path": str(case / "saved-project")})
        if options.reuse_motion_source:
            document = request("POST", "/v1/video/projects/open", {"schema_version": 1, "location_id": location["location_id"]})
            project = document["project"]["id"]
            assert len(document["takes"]) == 1 and document["takes"][0]["id"] == "take_4bb77e7e286f409a9b50236a742ca067"
            scene = next(item for item in document["project"]["scenes"] if item["id"] == "own_h3_scene")
            assert scene["accepted_take"]["id"] == document["takes"][0]["id"], "own source must remain Accepted after reopen"
        else:
            document = request("POST", "/v1/video/projects", {"schema_version": 1, "default_model": model, "delivery": {"width": 416, "height": 224, "fps": {"num": 24, "den": 1}}})
            project = document["project"]["id"]
            image = request("POST", "/v1/video/assets", raw=(PREVIOUS / "own-portrait.png").read_bytes(), mime="image/png")
            document = request("POST", f"/v1/video/projects/{project}/assets", {"schema_version": 1, "expected_project_revision": document["project"]["project_revision"], "asset_id": image["id"]})
            assert len(document["project"]["assets"]) == 1
            project_asset = next(iter(document["project"]["assets"]))
            prompt = json.loads((PREVIOUS / "settings.json").read_text())["prompt"]
            scene = {"id": "own_h3_scene", "start": {"num": 0, "den": 1}, "duration": {"num": 31, "den": 6}, "description": prompt, "model": "project_default", "settings": [{"model": model, "schema_version": 1, "values": single}], "sections": [], "bindings": [{"id": "own_first_frame", "asset": project_asset, "role": "first_frame", "enabled": True}], "continuity": None, "accepted_take": None}
            document = request("POST", f"/v1/video/projects/{project}/edits", {"schema_version": 1, "expected_project_revision": document["project"]["project_revision"], "commands": [{"operation": "add_scene", "scene": scene}]})
            document = request("POST", f"/v1/video/projects/{project}/save", {"schema_version": 1, "expected_project_revision": document["project"]["project_revision"], "location_id": location["location_id"]})
        baseline_takes = len(document["takes"])
        baseline_motion = len([r for r in document["resources"] if r["kind"]["kind"] == "h3_motion"])
        variants = ["single", "motion_context", "motion_context_audio", "motion_context_exp", "motion_context_exp_audio"] if options.motion else ["single", "last_frame_first_frame"] if options.continuity else ["single", "continue_latent", "re_noise_video_prediction"]
        if options.reuse_motion_source:
            variants.remove("single")
        frozen_endpoint = None
        source_scene = copy.deepcopy(scene)
        for name in variants:
            settings = copy.deepcopy(single)
            if name == "last_frame_first_frame" or name.startswith("motion_context"):
                source_take = document["takes"][0]["id"]
                kind = "h3_motion" if options.motion else "h3_endpoint"
                frozen_endpoint = next(resource for resource in document["resources"] if resource["kind"]["kind"] == kind and resource["kind"]["take_id"] == source_take)
                assert (frozen_endpoint["kind"]["descriptor"]["schema_version"] if options.motion else frozen_endpoint["kind"]["schema_version"]) == 1
                assert re.fullmatch(r"[0-9a-f]{64}", frozen_endpoint["sha256"])
                endpoint_file = case / "saved-project/continuity" / frozen_endpoint["sha256"]
                assert endpoint_file.resolve().is_relative_to(case / "saved-project/continuity")
                assert hashlib.sha256(endpoint_file.read_bytes()).hexdigest() == frozen_endpoint["sha256"]
                document = request("POST", f"/v1/video/projects/{project}/review", {"schema_version": 1, "expected_project_revision": document["project"]["project_revision"], "scene_id": source_scene["id"], "take_id": source_take, "action": "accept"})
                target = copy.deepcopy(source_scene)
                method = name.removesuffix("_audio") if options.motion else "last_frame_first_frame"
                target_index = len(document["project"]["scenes"])
                target.update(id="own_h3_target_" + str(target_index), start={"num": 31 * target_index, "den": 6}, bindings=[], accepted_take=None, continuity={"source_scene": source_scene["id"], "method": method, "include_audio": name.endswith("_audio")})
                if options.motion:
                    reference_model = dict(model, variant="reference", profile="h3-av-noref-v2")
                    target["model"] = {"override": reference_model}
                    target["settings"] = [{"model": reference_model, "schema_version": 1, "values": single}]
                target["description"] = "The same adult woman continues smiling, gently lowers her raised hand and looks calmly toward the camera. Quiet room ambience, no speech."
                document = request("POST", f"/v1/video/projects/{project}/edits", {"schema_version": 1, "expected_project_revision": document["project"]["project_revision"], "commands": [{"operation": "add_scene", "scene": target}]})
                scene = target
                result["frozen_endpoint"] = frozen_endpoint
            elif name != "single":
                settings["stage_two"] = {"sampling": dict(stage, steps=16), "model": model, "handoff": name, "trained_latent_upscale_model": None}
                document = request("POST", f"/v1/video/projects/{project}/edits", {"schema_version": 1, "expected_project_revision": document["project"]["project_revision"], "commands": [{"operation": "set_model_settings", "scene_id": scene["id"], "settings": {"model": model, "schema_version": 1, "values": settings}}]})
            started = time.monotonic()
            admitted = request("POST", "/v1/jobs", {"kind": "video_scene", "schema_version": 1, "project_id": project, "expected_project_revision": document["project"]["project_revision"], "scene_id": scene["id"], "seed": 42}, key=options.case + "-" + name)
            job_id = admitted["id"]
            observations = []
            last = None
            while True:
                job = request("GET", f"/v1/jobs/{job_id}")
                stamp = (job["status"], job.get("phase"), (job.get("progress") or {}).get("completed"))
                if stamp != last:
                    item = {"status": stamp[0], "phase": stamp[1], "step": stamp[2], "seconds": time.monotonic() - started}
                    observations.append(item)
                    print(json.dumps({"variant": name, **item}), flush=True)
                    last = stamp
                (case / "evidence/current.json").write_text(json.dumps({"variant": name, "server_pid": server.pid, "job_id": job_id, "observations": observations}, indent=2))
                if job["status"] in {"succeeded", "failed", "cancelled", "interrupted"}:
                    break
                assert time.monotonic() - started < 1800, "own GPU job timed out"
                time.sleep(0.25)
            (case / "evidence" / (name + "-job.json")).write_text(json.dumps(job, indent=2))
            assert job["status"] == "succeeded", job.get("error")
            runtime_status = request("GET", "/v1/runtime")
            assert runtime_status["engine"]["pid"] is None, "own worker exit unconfirmed"
            output = Path(job["output"]["saved_path"])
            assert output.resolve().is_relative_to(case / "output") and output.suffix == ".mp4"
            probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(output)], env=env))
            video = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
            audio = next(stream for stream in probe["streams"] if stream["codec_type"] == "audio")
            assert video["codec_name"] == "h264" and video["pix_fmt"] == "yuv420p" and int(video["nb_read_frames"]) == 124
            assert video["color_space"] == "bt709" and video["color_range"] == "tv" and video["r_frame_rate"] == "24/1"
            assert audio["codec_name"] == "aac" and int(audio["sample_rate"]) == 32000 and audio["channels"] == 2
            assert abs(float(probe["format"]["duration"]) - 31 / 6) <= acceptance["duration_tolerance"]
            pcm = array.array("f", subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(output), "-map", "0:a:0", "-ar", "32000", "-ac", "2", "-f", "f32le", "pipe:1"], env=env))
            assert pcm and all(math.isfinite(value) for value in pcm)
            rms = math.sqrt(sum(value * value for value in pcm) / len(pcm))
            (case / "evidence" / (name + "-audio-metrics.json")).write_text(json.dumps({"rms": rms, "peak": max(abs(value) for value in pcm), "samples_per_channel": len(pcm) // 2, "minimum_rms": acceptance["audio_rms_min"], "passed": rms > acceptance["audio_rms_min"]}, indent=2))
            assert rms > acceptance["audio_rms_min"], f"unexpected near-silent own output: RMS {rms:.12g} <= {acceptance['audio_rms_min']}"
            seek = subprocess.check_output(["ffmpeg", "-v", "error", "-ss", "4.5", "-i", str(output), "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"], env=env)
            assert len(seek) == 416 * 224 * 3
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(output), "-frames:v", "1", str(case / (name + "-preview.png"))], env=env, check=True)
            variant = {"name": name, "settings": settings, "job_id": job_id, "seconds": time.monotonic() - started, "output": str(output), "probe": probe, "decoded_audio_samples_per_channel": len(pcm) // 2, "decoded_audio_rms": rms, "decoded_audio_peak": max(abs(value) for value in pcm), "seek_frame_bytes": len(seek), "worker_reaped_before_ticket_release": runtime_status["engine"]["pid"] is None, "observations": observations}
            result["variants"].append(variant)
            (case / "evidence/results-in-progress.json").write_text(json.dumps(result, indent=2))
            document = request("GET", f"/v1/video/projects/{project}")
            assert len(document["takes"]) == baseline_takes + len(result["variants"])
            if options.motion:
                tail = next(resource for resource in document["resources"] if resource["kind"]["kind"] == "h3_motion" and resource["kind"]["take_id"] == document["takes"][-1]["id"])
                descriptor = tail["kind"]["descriptor"]
                head = 0 if name == "single" else 21 if name.startswith("motion_context_exp") else 34
                assert descriptor["shape"] == {"width": 416, "height": 224, "frames": 124, "fps": 24}
                assert descriptor["sampled_frames"] == (124 if name == "single" else 158)
                assert descriptor["visible_end_frame"] == descriptor["selected_end_frame"] == head + 124
                data = (case / "saved-project/continuity" / tail["sha256"]).read_bytes()
                assert hashlib.sha256(data).hexdigest() == tail["sha256"]
                assert len(data) == (24 * 7 * 14 * 26 + 32 * 2 * 37) * 4
                assert all(math.isfinite(value) for value in array.array("f", data))
                result["variants"][-1]["continuation_artifact"] = tail
            if name == "last_frame_first_frame" or name.startswith("motion_context"):
                provenance = document["takes"][-1]["provenance"]
                assert provenance["source_take"] == frozen_endpoint["kind"]["take_id"]
                assert provenance["continuity"]["resource_id"] == frozen_endpoint["id"] and provenance["continuity"]["sha256"] == frozen_endpoint["sha256"]
                assert provenance["continuity"]["method"] == (name.removesuffix("_audio") if options.motion else "last_frame_first_frame")
                assert provenance["continuity"].get("include_audio", False) == name.endswith("_audio")
                assert len([resource for resource in document["resources"] if resource["kind"]["kind"] == kind]) == (baseline_motion if options.motion else 0) + len(result["variants"])
                result["target_provenance"] = provenance
                result["continuity_verified"] = True
            print(json.dumps({"variant_passed": name, "output": str(output), "rms": rms}), flush=True)
        result["completed"] = True
    except BaseException as error:
        result["error"] = str(error) or repr(error)
        result["traceback"] = traceback.format_exc()
        print(json.dumps({"own_test_error": result["error"], "traceback": result["traceback"]}), flush=True)
    finally:
        if server is not None:
            if server.poll() is None:
                server.terminate()
            try:
                result["server_exit"] = server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                result["server_exit_uncertain"] = True
            result["own_server_reaped"] = server.poll() is not None
        (case / "evidence/result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({"completed": result["completed"], "variants_passed": len(result["variants"]), "own_server_reaped": result.get("own_server_reaped"), "error": result.get("error")}), flush=True)
    return 0 if result["completed"] and result.get("own_server_reaped") else 1


if __name__ == "__main__":
    def stop_owned_test(_signum, _frame):
        raise KeyboardInterrupt("own test interrupted")
    signal.signal(signal.SIGTERM, stop_owned_test)
    raise SystemExit(main())
