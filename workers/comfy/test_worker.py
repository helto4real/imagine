"""CPU/stdlib-only checks. Does not import ComfyUI, Torch, NumPy or Pillow."""
import contextlib
import copy
import io
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock

from worker import (ATTENTION, MAX_HEADER, MAX_PAYLOAD, WorkerError, read_frame,
                    component_load, references, serve, write_frame,
                    private_temporary_directory)
from runtime import Runtime, require_kitchen
from sampling_preview import State as PreviewState


def request(**values):
    return {"protocol": 1, "type": "generate", "id": "synthetic", "reference_lengths": [], **values}


class ProtocolTests(unittest.TestCase):
    def test_private_temporary_directory_never_probes_an_unencrypted_fallback(self):
        import os
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / "own-missing-volume")
            with patch.dict(os.environ, {"IMAGINE_PRIVATE_TMP": missing}), \
                 patch.object(tempfile, "tempdir", None), \
                 patch.object(tempfile, "_get_default_tempdir", side_effect=AssertionError("fallback")):
                private_temporary_directory()
                with self.assertRaises(FileNotFoundError):
                    tempfile.TemporaryFile()
            self.assertFalse(Path(missing).exists())

    def test_preview_payload_is_separate_and_does_not_replace_final_result(self):
        class PreviewRuntime:
            def generate(self, request, references, emit):
                emit("preview", metadata={"frames": 1}, payload=b"\x01\x02\x03")
                emit("preview_warning")
                return b"unchanged final PNG"
        incoming, outgoing = io.BytesIO(), io.BytesIO()
        write_frame(incoming, request())
        incoming.seek(0)
        serve(incoming, outgoing, PreviewRuntime())
        outgoing.seek(0)
        read_frame(outgoing)
        self.assertEqual(read_frame(outgoing), ({"type":"preview", "id":"synthetic", "metadata":{"frames":1}}, b"\x01\x02\x03"))
        self.assertEqual(read_frame(outgoing), ({"type":"preview_warning", "id":"synthetic"}, b""))
        self.assertEqual(read_frame(outgoing), ({"type":"result", "id":"synthetic"}, b"unchanged final PNG"))

    def test_preview_cadence_precedes_render_and_failure_is_nonfatal(self):
        now = [0.]
        events = []
        emit = lambda event, **data: events.append((event, data))
        renderer = Mock(return_value=(1, 1, b"rgb"))
        disabled = PreviewState(None, clock=lambda: now[0])
        now[0] = 100.
        disabled.update(1, object(), emit, renderer)
        renderer.assert_not_called()
        state = PreviewState({"enabled":True, "interval_seconds":2., "max_resolution":1}, clock=lambda: now[0])
        now[0] = 101.
        state.update(1, object(), emit, renderer)
        renderer.assert_not_called()
        now[0] = 102.
        renderer.side_effect = ValueError("private preview detail")
        state.update(2, object(), emit, renderer)
        self.assertEqual(events, [("preview_warning", {})])
        renderer.side_effect = None
        now[0] = 104.
        state.update(4, object(), emit, renderer)
        self.assertEqual(events[1][0], "preview")
        self.assertEqual(events[1][1]["metadata"]["sequence"], 1)
        self.assertEqual(events[1][1]["payload"], b"rgb")

    def test_component_failures_keep_role_and_safe_cause_without_exception_text(self):
        for component in ("text_encoder", "vae", "diffusion_model"):
            for error, reason, tensor in (
                (FileNotFoundError("synthetic private sentinel"), "file_not_found", None),
                (PermissionError("synthetic private sentinel"), "permission_denied", None),
                (MemoryError("synthetic private sentinel"), "out_of_memory", None),
                (KeyError("layers.0.weight"), "missing_tensor", "layers.0.weight"),
                (KeyError("/private/synthetic sentinel"), "missing_tensor", None),
                (RuntimeError("size mismatch: synthetic private sentinel"), "tensor_shape_mismatch", None),
                (ValueError("synthetic private sentinel"), "invalid_checkpoint", None),
                (RuntimeError("VAE is invalid: synthetic private sentinel"), "unsupported_architecture", None),
                (RuntimeError("unsupported quantization synthetic private sentinel"), "unsupported_quantization", None),
                (RuntimeError("synthetic private sentinel"), "load_failed", None),
            ):
                with self.subTest(component=component, reason=reason):
                    runtime = Mock()
                    runtime.generate.side_effect = lambda *_: component_load(component, Mock(side_effect=error))
                    incoming, outgoing = io.BytesIO(), io.BytesIO()
                    write_frame(incoming, request())
                    incoming.seek(0)
                    serve(incoming, outgoing, runtime)
                    self.assertNotIn(b"synthetic private sentinel", outgoing.getvalue())
                    outgoing.seek(0)
                    read_frame(outgoing)
                    event, payload = read_frame(outgoing)
                    diagnostic = {"component": component, "reason": reason}
                    if tensor is not None:
                        diagnostic["tensor"] = tensor
                    self.assertEqual(event, {"type": "error", "id": "synthetic", "code": "model_component_load_failed", "diagnostic": diagnostic})
                    self.assertEqual(payload, b"")
    def test_real_bootstrap_keeps_import_output_private_and_requires_kitchen(self):
        # Exercise the actual -I entrypoint against tiny source doubles, never
        # the installed Comfy/Torch modules or a GPU.
        sources = {
            "torch.py": "print('synthetic import noise')\n",
            "numpy.py": "",
            "PIL/__init__.py": "Image = object()\n",
            "comfy/__init__.py": "",
            "comfy/options.py": "def enable_args_parsing(value):\n    assert value is False\n",
            "comfy/sd.py": "",
            "comfy/sample.py": "",
            "comfy/utils.py": "",
            "comfy/model_management.py": "from types import SimpleNamespace\ndef get_torch_device():\n    return SimpleNamespace(type='cuda')\n",
            "comfy/ldm/__init__.py": "",
            "comfy/ldm/modules/__init__.py": "",
            "comfy/ldm/modules/attention.py": "def get_attention_function(name, default):\n    assert name == 'comfy_kitchen_int8'\n    return object()\n",
            "comfy_extras/__init__.py": "",
            "comfy_extras/nodes_qwen.py": "TextEncodeQwenImage21 = QwenImage21Cache = object()\n",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path, source in sources.items():
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(source)
            for available in (True, False):
                (root / "comfy_kitchen.py").write_text(
                    f"def int8_attention_is_available(device):\n    return {available}\n")
                child = subprocess.run([sys.executable, "-I", "-u", str(Path(__file__).with_name("worker.py")),
                                        "--comfy-root", directory], input=b"", capture_output=True, timeout=5)
                output = io.BytesIO(child.stdout)
                event, payload = read_frame(output)
                self.assertEqual(payload, b"")
                self.assertEqual(output.read(), b"")
                self.assertEqual(child.stderr, b"")
                if available:
                    self.assertEqual(child.returncode, 0)
                    self.assertEqual(event, {"type": "ready", "protocol": 1, "attention": ATTENTION})
                else:
                    self.assertEqual(child.returncode, 1)
                    self.assertEqual(event, {"type": "error", "id": None, "code": "comfy_kitchen_unavailable"})

    def test_fragmented_binary_transport_and_order(self):
        class Fragmented(io.BytesIO):
            def read(self, size=-1):
                return super().read(min(size, 3))
        raw = io.BytesIO()
        write_frame(raw, request(prompt="synthetic \u2603", reference_lengths=[3, 2]), b"\x00a\xffbc")
        header, payload = read_frame(Fragmented(raw.getvalue()))
        self.assertEqual(header["prompt"], "synthetic \u2603")
        self.assertEqual(references(header, payload), [b"\x00a\xff", b"bc"])

    def test_limits_truncation_and_bad_lengths(self):
        for raw in (struct.pack(">I", MAX_HEADER + 1), struct.pack(">I", 2) + b"{}" + struct.pack(">I", MAX_PAYLOAD + 1)):
            with self.assertRaises(WorkerError):
                read_frame(io.BytesIO(raw))
        with self.assertRaises(EOFError):
            read_frame(io.BytesIO(b"\x00\x00"))
        for lengths in ([0], [-1], [True], [2], [1] * 11):
            with self.assertRaises(WorkerError):
                references(request(reference_lengths=lengths), b"x")

    def test_persistent_requests_and_redacted_failure(self):
        incoming, outgoing = io.BytesIO(), io.BytesIO()
        for index in range(3):
            write_frame(incoming, request(id=str(index)))
        incoming.seek(0)
        runtime = Mock()
        runtime.generate.side_effect = [b"first", b"second", RuntimeError("private synthetic sentinel")]
        serve(incoming, outgoing, runtime)
        raw = outgoing.getvalue()
        self.assertNotIn(b"private synthetic sentinel", raw)
        outgoing.seek(0)
        self.assertEqual(read_frame(outgoing)[0]["attention"], ATTENTION)
        self.assertEqual(read_frame(outgoing), ({"type": "result", "id": "0"}, b"first"))
        self.assertEqual(read_frame(outgoing), ({"type": "result", "id": "1"}, b"second"))
        self.assertEqual(read_frame(outgoing)[0], {"type": "error", "id": "2", "code": "comfy_generation_failed"})


class Patch:
    def __init__(self, patches=()):
        self.patches = list(patches)
        self.attention = None

    def clone(self):
        return copy.deepcopy(self)

    def set_model_optimized_attention(self, attention):
        self.attention = attention


class RuntimeTests(unittest.TestCase):
    def test_kitchen_is_required_without_fallback(self):
        function = object()
        kitchen = Mock()
        kitchen.int8_attention_is_available.return_value = True
        device = types.SimpleNamespace(type="cuda")
        self.assertIs(require_kitchen(device, kitchen, function), function)
        for kind, available, attention in (("cpu", True, function), ("cuda", False, function), ("cuda", True, None)):
            kitchen.int8_attention_is_available.return_value = available
            with self.assertRaises(WorkerError) as failure:
                require_kitchen(types.SimpleNamespace(type=kind), kitchen, attention)
            self.assertEqual(failure.exception.code, "comfy_kitchen_unavailable")

    def test_lora_order_strength_and_removal_do_not_mutate_base(self):
        runtime = Runtime.__new__(Runtime)
        runtime.model, runtime.clip = Patch(), Patch()
        runtime.attention = object()
        runtime.utils = Mock()
        runtime.utils.load_torch_file.side_effect = lambda path, **kw: (path, {"source": path})

        def load_lora(model, clip, weights, strength_model, strength_clip, **kw):
            model, clip = model.clone(), clip.clone()
            model.patches.append((weights, strength_model))
            clip.patches.append((weights, strength_clip))
            return model, clip
        runtime.sd = types.SimpleNamespace(load_lora_for_models=load_lora)
        runtime.cache = Mock()
        runtime.cache.execute.side_effect = lambda model, **kw: types.SimpleNamespace(result=(model.clone(),))
        model, clip = runtime.patched({"loras": [{"path": "b", "strength": 0.7}, {"path": "a", "strength": -0.2}]})
        self.assertEqual(model.patches, [("b", 0.7), ("a", -0.2)])
        self.assertEqual(clip.patches, model.patches)
        self.assertIs(model.attention, runtime.attention)
        clean, clean_clip = runtime.patched({"loras": []})
        self.assertEqual((clean.patches, clean_clip.patches, runtime.model.patches, runtime.clip.patches), ([], [], [], []))
        runtime.cache.execute.assert_called_with(unittest.mock.ANY, device="auto", dtype="default")

    def test_component_reuse_and_replacement_at_same_path(self):
        runtime = Runtime.__new__(Runtime)
        runtime.loaded_key = None
        runtime.sd, runtime.utils, runtime.mm = Mock(), Mock(), Mock()
        runtime.utils.load_torch_file.return_value = ({}, {"vae": "metadata"})
        with tempfile.TemporaryDirectory() as directory:
            paths = {name: str(Path(directory) / name) for name in ("diffusion_model", "text_encoder", "vae")}
            for path in paths.values():
                Path(path).write_bytes(b"synthetic")
            runtime.load(paths)
            runtime.load(paths)
            self.assertEqual(runtime.sd.load_diffusion_model.call_count, 1)
            runtime.sd.VAE.assert_called_once_with(sd={}, metadata={"vae": "metadata"})
            Path(paths["diffusion_model"]).write_bytes(b"synthetic replacement")
            runtime.load(paths)
            self.assertEqual(runtime.sd.load_diffusion_model.call_count, 2)

    def test_real_load_boundaries_identify_encoder_and_vae(self):
        for failing in ("text_encoder", "vae"):
            runtime = Runtime.__new__(Runtime)
            runtime.loaded_key = None
            runtime.sd, runtime.utils, runtime.mm = Mock(), Mock(), Mock()
            runtime.utils.load_torch_file.return_value = ({}, {})
            if failing == "text_encoder":
                runtime.sd.load_clip.side_effect = KeyError("layers.0.weight")
            else:
                runtime.sd.VAE.return_value.throw_exception_if_invalid.side_effect = RuntimeError("VAE is invalid")
            with tempfile.TemporaryDirectory() as directory:
                paths = {name: str(Path(directory) / name) for name in ("diffusion_model", "text_encoder", "vae")}
                for path in paths.values():
                    Path(path).write_bytes(b"synthetic")
                with self.assertRaises(WorkerError) as result:
                    runtime.load(paths)
                self.assertEqual(result.exception.diagnostic["component"], failing)
                self.assertIsNone(runtime.loaded_key)

    @staticmethod
    def generation_runtime():
        # Shared with the Rust IPC integration fixture: actual adapter, CPU doubles.
        runtime = Runtime.__new__(Runtime)
        runtime.load = Mock()
        runtime.patched = Mock(return_value=("model", "clip"))
        runtime.torch, runtime.np, runtime.Image = Mock(), MagicMock(), Mock()
        runtime.torch.inference_mode.side_effect = contextlib.nullcontext
        runtime.mm = Mock()
        runtime.mm.intermediate_device.return_value = "cpu"
        runtime.vae, runtime.sample, runtime.encode = MagicMock(), Mock(), Mock()
        runtime.encode.execute.return_value.result = ("positive", [[object(), {"pooled_output": None}]], "ignored reference-sized latent")
        def zeros_like(value):
            if value is None:
                raise TypeError("zeros_like requires a tensor, not None")
            return object()
        runtime.torch.zeros_like.side_effect = zeros_like
        runtime.Image.open.side_effect = [contextlib.nullcontext(Mock()), contextlib.nullcontext(Mock())]
        tensor1, tensor2 = object(), object()
        runtime.torch.from_numpy.side_effect = [Mock(unsqueeze=Mock(return_value=tensor1)), Mock(unsqueeze=Mock(return_value=tensor2))]
        runtime.Image.fromarray.return_value.save.side_effect = lambda output, **kw: output.write(b"png")
        return runtime, tensor1, tensor2

    def test_generate_without_optional_custom_sigmas_uses_comfy_schedule(self):
        for sampler, scheduler in (("er_sde", "beta"), ("euler", "simple")):
            with self.subTest(sampler=sampler, scheduler=scheduler):
                runtime, _, _ = self.generation_runtime()
                config = request(prompt="synthetic", negative_prompt="", width=1600, height=896, seed=42,
                                 zero_negative_conditioning=False,
                                 sampling={"steps": 40, "cfg_scale": 2.5, "sampler": sampler, "scheduler": scheduler})
                self.assertEqual(runtime.generate(config, [], lambda *_args, **_kw: None), b"png")
                runtime.torch.tensor.assert_not_called()
                self.assertIsNone(runtime.sample.sample.call_args.kwargs["sigmas"])
                self.assertEqual(runtime.sample.sample.call_args.args[2:6], (40, 2.5, sampler, scheduler))

    def test_generate_keeps_target_geometry_reference_order_and_custom_sigmas(self):
        runtime, tensor1, tensor2 = self.generation_runtime()
        events = []
        sigmas = [1, 0.978453, 0.954180, 0.926626, 0.895080, 0.845148, 0.704534, 0.414568, 0]
        config = request(prompt="synthetic", negative_prompt="", width=1600, height=896, seed=42,
                         zero_negative_conditioning=False,
                         sampling={"steps": 8, "cfg_scale": 1.0, "sampler": "euler", "scheduler": "simple", "custom_sigmas": sigmas})
        self.assertEqual(runtime.generate(config, [b"first", b"second"], lambda event, **kw: events.append(event)), b"png")
        runtime.torch.zeros.assert_called_once_with([1, 64, 56, 100], device="cpu")
        runtime.encode.execute.assert_called_once_with("clip", "synthetic", "", vae=runtime.vae,
                                                       resolution=0, images={"image_1": tensor1, "image_2": tensor2})
        runtime.torch.tensor.assert_called_once_with(sigmas, dtype=runtime.torch.float32)
        self.assertEqual(runtime.sample.sample.call_args.args[2:6], (8, 1.0, "euler", "simple"))
        self.assertIs(runtime.sample.sample.call_args.kwargs["sigmas"], runtime.torch.tensor.return_value)
        self.assertTrue(runtime.sample.sample.call_args.kwargs["disable_pbar"])
        self.assertEqual(events[-1], "phase")

    def test_turbo_zero_negative_accepts_missing_none_and_tensor_pooled_outputs(self):
        runtime, _, _ = self.generation_runtime()
        conditions = [object(), object(), object()]
        pooled = object()
        references, mask = object(), object()
        negative = [
            [conditions[0], {"pooled_output": None, "reference_latents": references}],
            [conditions[1], {"attention_mask": mask}],
            [conditions[2], {"pooled_output": pooled}],
        ]
        runtime.encode.execute.return_value.result = ("positive", negative, "unused latent")
        zeros = {id(value): object() for value in [*conditions, pooled]}

        def zeros_like(value):
            if value is None:
                raise TypeError("zeros_like requires a tensor, not None")
            return zeros[id(value)]

        runtime.torch.zeros_like.side_effect = zeros_like
        sigmas = [1, 0.978453, 0.954180, 0.926626, 0.895080, 0.845148, 0.704534, 0.414568, 0]
        config = request(prompt="synthetic", negative_prompt="", width=1600, height=896, seed=42,
                         zero_negative_conditioning=True,
                         sampling={"steps": 8, "cfg_scale": 1.0, "sampler": "res_multistep", "scheduler": "simple", "custom_sigmas": sigmas})
        self.assertEqual(runtime.generate(config, [], lambda *_args, **_kw: None), b"png")
        args = runtime.sample.sample.call_args.args
        self.assertEqual(args[2:6], (8, 1.0, "res_multistep", "simple"))
        self.assertEqual(args[7][0], [zeros[id(conditions[0])], {"pooled_output": None, "reference_latents": references}])
        self.assertEqual(args[7][1], [zeros[id(conditions[1])], {"attention_mask": mask}])
        self.assertEqual(args[7][2], [zeros[id(conditions[2])], {"pooled_output": zeros[id(pooled)]}])
        self.assertIs(negative[2][1]["pooled_output"], pooled)
        runtime.torch.tensor.assert_called_once_with(sigmas, dtype=runtime.torch.float32)


if __name__ == "__main__":
    unittest.main()
