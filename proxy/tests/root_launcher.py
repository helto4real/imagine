#!/usr/bin/env python3
"""Check root launcher dispatch without starting services or touching the GPU."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class RootLauncher(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="imagine-root-launcher-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        shutil.copy2(ROOT / "start-proxy.sh", self.root / "start-proxy.sh")
        for directory in ("bin", "proxy", "target/release"):
            (self.root / directory).mkdir(parents=True)
        cargo = self.root / "bin/cargo"
        cargo.write_text("#!/bin/sh\nexit 0\n")
        cargo.chmod(0o755)
        script = self.root / "target/release/st-proxy-launch"
        script.write_text(
            "#!/usr/bin/env python3\nimport json, os, sys\n"
            'print(json.dumps({"args":sys.argv[1:], "env":{k:v for k,v in os.environ.items() if k.startswith("ST_PROXY_")}}))\n'
        )
        script.chmod(0o755)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("ST_PROXY_")}
        self.env["PATH"] = str(self.root / "bin") + ":" + self.env["PATH"]

    def run_launcher(self, *args, **env):
        result = subprocess.check_output(
            [str(self.root / "start-proxy.sh"), *args],
            env=self.env | env, text=True,
        )
        return json.loads(result.splitlines()[-1])

    def test_default_starts_all_through_supervisor(self):
        result = self.run_launcher()
        self.assertEqual(result["args"], [])
        env = result["env"]
        self.assertEqual(env["ST_PROXY_LLM_BACKEND"], "all")
        self.assertNotIn("ST_PROXY_LLM_URL", env)
        self.assertEqual([env[f"ST_PROXY_{kind}_PORT"] for kind in ("CHAT", "IMAGE", "MCP")],
                         ["5002", "8189", "8190"])
        self.assertEqual(env["ST_PROXY_COORDINATOR_SOCKET"], str(self.root / ".imagine/gpu.sock"))

    def test_stack_controls_and_environment_selection(self):
        for args in (("--log",), ("--stop",), ("--log", "--stop")):
            result = self.run_launcher(*args, ST_PROXY_LLM_BACKEND="sc117")
            self.assertEqual(result["args"], list(args))
            self.assertEqual(result["env"]["ST_PROXY_LLM_BACKEND"], "sc117")

    def test_options_are_forwarded_to_shared_parser(self):
        for args in [("--llm-b", "sc117", "--image-port", "18189"),
                     ("--proxy-only",), ("--help",), ("--check-backend",)]:
            self.assertEqual(self.run_launcher(*args)["args"], list(args))

    def test_original_image_root_and_imagine_only_defaults(self):
        env = self.run_launcher()["env"]
        self.assertEqual(env["ST_PROXY_MCP_OUTPUT_DIR"], "/run/media/veracrypt1/mcp-images")
        self.assertEqual(env["ST_PROXY_COMFY_ENABLED"], "false")
        env = self.run_launcher(ST_PROXY_MCP_OUTPUT_DIR="/tmp/custom", ST_PROXY_COMFY_ENABLED="true")["env"]
        self.assertEqual(env["ST_PROXY_MCP_OUTPUT_DIR"], "/tmp/custom")
        self.assertEqual(env["ST_PROXY_COMFY_ENABLED"], "true")


if __name__ == "__main__":
    unittest.main()
