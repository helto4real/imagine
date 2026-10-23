#!/usr/bin/env python3
"""Build the isolated Director UI candidate, without publishing or launching it."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "comfyui/ui-groups-20261021"
SOURCE = CANDIDATE / "source"
EVIDENCE = Path("/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01")

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] not in {"build", "check", "clippy", "test", "fmt"}:
        raise SystemExit("Expected a candidate build/check/test command.")
    mount=json.loads(subprocess.check_output(["findmnt","-J","-T",str(EVIDENCE)]))["filesystems"][0]
    if (mount["target"],mount["source"],mount["fstype"])!=("/home","/dev/mapper/cryptroot[/@home]","btrfs") or os.statvfs(EVIDENCE).f_flag & os.ST_RDONLY:
        raise SystemExit("Own encrypted test destination is unavailable or read-only.")
    test_tmp=EVIDENCE/'tmp'
    if 'video_jobs' in args:
        # Linux Unix-domain socket paths must fit SUN_LEN. Keep these synthetic
        # coordinator fixtures on the same verified encrypted volume.
        test_tmp=EVIDENCE/'g15'
        test_tmp.mkdir(exist_ok=True)
    env = os.environ.copy()
    integrated = EVIDENCE / 'integrated-a-tests'
    for directory in [integrated / 'fixtures', integrated / 'evidence']:
        directory.mkdir(parents=True, exist_ok=True)
    env.update(CARGO_TARGET_DIR=str(CANDIDATE / "target"), CARGO_BUILD_JOBS="2",
               CARGO_INCREMENTAL="0", CARGO_PROFILE_DEV_DEBUG="0", CARGO_PROFILE_TEST_DEBUG="0",
               TMPDIR=str(test_tmp), PYTHONDONTWRITEBYTECODE="1",
               IMAGINE_WORKER_A_FIXTURES=str(integrated / 'fixtures'),
               IMAGINE_WORKER_A_EVIDENCE=str(integrated / 'evidence'))
    env['IMAGINE_LORA_TEST_ROOT'] = str(EVIDENCE / 'integrated-b-lora-tests')
    raise SystemExit(subprocess.call(["cargo", *args], cwd=SOURCE, env=env))
