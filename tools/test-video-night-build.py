"""Isolation regression: ordinary metadata and guarded builds select no product path."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "tools/video-night-cargo.py"
spec = importlib.util.spec_from_file_location("night_cargo", WRAPPER)
night = importlib.util.module_from_spec(spec)
spec.loader.exec_module(night)
native_spec = importlib.util.spec_from_file_location("night_native", ROOT / "tools/build-video-night-native.py")
native = importlib.util.module_from_spec(native_spec)
native_spec.loader.exec_module(native)


class Isolation(unittest.TestCase):
    def test_native_build_has_separate_source_target_and_rejects_redirect(self):
        self.assertNotEqual(native.TARGET, night.TARGET)
        self.assertTrue(native.TARGET.is_relative_to(night.CANDIDATE))
        self.assertTrue(native.SOURCE.is_relative_to(night.CANDIDATE))
        self.assertIn("shared/h3_sampling.rs", native.OVERLAY)
        self.assertIn("inference/src/h3_product/sampling.rs", native.OVERLAY)
        with tempfile.TemporaryDirectory(dir=night.EVIDENCE / "tmp") as folder:
            root = Path(folder)
            (root / "other").mkdir()
            (root / "redirect").symlink_to(root / "other", target_is_directory=True)
            with mock.patch.object(native, "TARGET", root / "redirect"), self.assertRaises(ValueError):
                native.environment()

    def test_override_cannot_redirect_candidate_to_product(self):
        for arguments in (["build", "--target-dir", "comfyui/h3-main-20261020/target-product"], ["test", "--config=build.target-dir=bad"], ["build", "--manifest-path=elsewhere/Cargo.toml"], ["run"], ["install"]):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                night.validate(arguments, {})
        with self.assertRaises(ValueError):
            night.validate(["check"], {"CARGO_TARGET_DIR": "/tmp/other-target"})

    def test_normal_and_guarded_cargo_report_separate_target(self):
        configuration = tomllib.loads((ROOT / ".cargo/config.toml").read_text())
        self.assertEqual(ROOT / configuration["build"]["target-dir"], night.TARGET)
        env = os.environ.copy()
        env.pop("CARGO_TARGET_DIR", None)
        for command in (["cargo", "metadata", "--locked", "--offline", "--no-deps", "--format-version", "1"], [sys.executable, str(WRAPPER), "metadata", "--locked", "--offline", "--no-deps", "--format-version", "1"]):
            value = json.loads(subprocess.check_output(command, cwd=ROOT, env=env))
            self.assertEqual(Path(value["target_directory"]), night.TARGET)
            self.assertNotIn("target-product", value["target_directory"])


if __name__ == "__main__":
    unittest.main()
