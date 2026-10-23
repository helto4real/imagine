#!/usr/bin/env python3
"""Build/check only the isolated video candidate; never publish or launch it."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "comfyui/h3-video-night-20261021"
TARGET = CANDIDATE / "target"
EVIDENCE = Path("/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e")


def validate(arguments, environ):
    if not arguments or arguments[0] not in {"build", "check", "clippy", "test", "fmt", "metadata"}:
        raise ValueError("Only candidate build/check/test commands are supported.")
    forbidden = ("--target-dir", "--manifest-path", "--config", "--out-dir", "--artifact-dir", "--lockfile-path", "-C", "-Z")
    if any(a == key or a.startswith(key + "=") for a in arguments for key in forbidden):
        raise ValueError("Build destination/configuration overrides are not allowed.")
    if TARGET.resolve() != TARGET or CANDIDATE.resolve() != CANDIDATE:
        raise ValueError("Candidate build directories must not be symlinks.")
    if environ.get("CARGO_TARGET_DIR", str(TARGET)) != str(TARGET):
        raise ValueError("An inherited Cargo target override conflicts with isolation.")
    return TARGET


def environment(arguments):
    validate(arguments, os.environ)
    mount = json.loads(subprocess.check_output(["findmnt", "-J", "-T", str(EVIDENCE)]))["filesystems"][0]
    if (mount["target"], mount["source"], mount["fstype"]) != ("/run/media/veracrypt1", "/dev/mapper/veracrypt1", "ext3") or os.statvfs(EVIDENCE).f_flag & os.ST_RDONLY:
        raise ValueError("The own encrypted test destination is unavailable or read-only.")
    temporary = EVIDENCE / "tmp"
    if temporary.resolve() != temporary:
        raise ValueError("The own temporary directory must not be a symlink.")
    temporary.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(CARGO_TARGET_DIR=str(TARGET), CARGO_BUILD_JOBS="2", CARGO_INCREMENTAL="0", CARGO_PROFILE_DEV_DEBUG="0", CARGO_PROFILE_TEST_DEBUG="0", TMPDIR=str(temporary))
    return env


if __name__ == "__main__":
    try:
        args = sys.argv[1:]
        env = environment(args)
        # The command-line target wins over every Cargo config/environment source.
        command = ["cargo", *args]
        if args[0] not in {"fmt", "metadata"}:
            command += ["--target-dir", str(TARGET)] if "--" not in args else []
            if "--" in args:
                index = command.index("--")
                command[index:index] = ["--target-dir", str(TARGET)]
        raise SystemExit(subprocess.call(command, cwd=ROOT, env=env))
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
