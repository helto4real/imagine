#!/usr/bin/env python3
"""Produce a new isolated code-only package, without changing launch selection."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--qa", action="store_true", help="Include the opt-in native QA harness")
options = parser.parse_args()
root = Path(__file__).resolve().parents[1]
wrapper = root / "tools/video-night-cargo.py"
spec = importlib.util.spec_from_file_location("night_cargo", wrapper)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
arguments = ["build", "--locked", "--offline", "--workspace", "--bins", "--examples"]
if options.qa:
    arguments += ["--features", "imagine-ui/ui-preview"]
environment = module.environment(arguments)
subprocess.run([sys.executable, str(wrapper), *arguments], cwd=root, env=environment, check=True)
package = module.CANDIDATE / "packages" / uuid.uuid4().hex
package.mkdir(parents=True, exist_ok=False)
records = []
for name, source in (("imagine-server", module.TARGET / "debug/imagine-server"), ("imagine-ui", module.TARGET / "debug/imagine-ui"), ("timeline_fixture_server", module.TARGET / "debug/examples/timeline_fixture_server")):
    destination = package / name
    shutil.copy2(source, destination)
    digest = hashlib.file_digest(destination.open("rb"), "sha256").hexdigest()
    records.append({"name": name, "sha256": digest})
(package / "candidate.json").write_text(json.dumps({"schema_version": 1, "qa": options.qa, "target": str(module.TARGET), "binaries": records, "activated": False, "runtime": "Supply an explicit own instance; no production config is bundled."}, indent=2) + "\n")
print(package)
