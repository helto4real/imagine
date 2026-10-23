"""Qwen-only adapter to Comfy core, without main/server/execution/custom nodes."""
import gc
import io
import os
import sys
import time
from pathlib import Path

from worker import ATTENTION, WorkerError, component_load
from sampling_preview import State as PreviewState, render as render_preview


def require_kitchen(device, kitchen, attention):
    if device.type != "cuda" or not kitchen.int8_attention_is_available(device) or attention is None:
        raise WorkerError("comfy_kitchen_unavailable")
    return attention


class Runtime:
    def __init__(self, root):
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
            os.environ[key] = "1"
        sys.path.insert(0, root)
        sys.argv = ["imagine-comfy-worker"]
        import comfy.options
        comfy.options.enable_args_parsing(False)
        import torch
        import numpy as np
        from PIL import Image
        import comfy.sd as sd
        import comfy.sample as sample
        import comfy.utils as utils
        import comfy.model_management as mm
        import comfy_kitchen
        from comfy.ldm.modules.attention import get_attention_function
        from comfy_extras.nodes_qwen import TextEncodeQwenImage21, QwenImage21Cache

        device = mm.get_torch_device()
        attention = get_attention_function(ATTENTION, None)
        require_kitchen(device, comfy_kitchen, attention)
        self.torch, self.np, self.Image = torch, np, Image
        self.sd, self.sample, self.utils, self.mm = sd, sample, utils, mm
        self.encode, self.cache = TextEncodeQwenImage21, QwenImage21Cache
        self.attention = attention
        self.loaded_key = None
        self.model = self.clip = self.vae = None

    def load(self, request):
        paths = [request[name] for name in ("diffusion_model", "text_encoder", "vae")]
        # Replacement at the same path must not reuse old tensors.
        stamps = [component_load(name, lambda path=path: Path(path).stat())
                  for name, path in zip(("diffusion_model", "text_encoder", "vae"), paths)]
        key = tuple((path, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
                    for path, stat in zip(paths, stamps))
        if self.loaded_key == key:
            return
        self.loaded_key = None
        self.model = self.clip = self.vae = None
        self.mm.unload_all_models()
        gc.collect()
        self.mm.soft_empty_cache()
        self.model = component_load("diffusion_model", lambda: self.sd.load_diffusion_model(paths[0]))
        self.clip = component_load("text_encoder", lambda: self.sd.load_clip([paths[1]], clip_type=self.sd.CLIPType.QWEN_IMAGE))

        def load_vae():
            weights, metadata = self.utils.load_torch_file(paths[2], safe_load=True, return_metadata=True)
            vae = self.sd.VAE(sd=weights, metadata=metadata)
            vae.throw_exception_if_invalid()
            return vae

        self.vae = component_load("vae", load_vae)
        self.loaded_key = key

    def patched(self, request):
        # Always start from pristine patches, including after LoRA -> no LoRA.
        model, clip = self.model.clone(), self.clip.clone()
        for entry in request["loras"]:
            weights, metadata = self.utils.load_torch_file(entry["path"], safe_load=True, return_metadata=True)
            model, clip = self.sd.load_lora_for_models(
                model, clip, weights, entry["strength"], entry["strength"], lora_metadata=metadata)
        model = self.cache.execute(model, device="auto", dtype="default").result[0]
        model.set_model_optimized_attention(self.attention)
        return model, clip

    def generate(self, request, references, emit):
        with self.torch.inference_mode():
            emit("phase", phase="loading_model")
            self.load(request)
            model, clip = self.patched(request)
            emit("phase", phase="preparing_inputs")
            images = {}
            for index, raw in enumerate(references):
                with self.Image.open(io.BytesIO(raw)) as image:
                    # Rust has already resized/cropped; keep alpha for the VAE.
                    pixels = self.np.array(image.convert("RGBA"), dtype=self.np.float32) / 255.0
                images[f"image_{index + 1}"] = self.torch.from_numpy(pixels).unsqueeze(0)
            emit("phase", phase="encoding")
            positive, negative, _ = self.encode.execute(
                clip, request["prompt"], request["negative_prompt"], vae=self.vae,
                resolution=0, images=images).result
            if request["zero_negative_conditioning"]:
                negative = [[self.torch.zeros_like(cond), {
                    **data, **({"pooled_output": self.torch.zeros_like(data["pooled_output"])}
                               if data.get("pooled_output") is not None else {})}] for cond, data in negative]
            latent = self.torch.zeros([1, 64, request["height"] // 16, request["width"] // 16],
                                      device=self.mm.intermediate_device())
            noise = self.sample.prepare_noise(latent, request["seed"])
            settings = request["sampling"]
            custom_sigmas = settings.get("custom_sigmas", [])
            sigmas = (self.torch.tensor(custom_sigmas, dtype=self.torch.float32)
                      if custom_sigmas else None)
            emit("phase", phase="inference")
            started = time.monotonic()
            preview = PreviewState(request.get("sampling_preview"))

            def callback(step, _x0, _x, total):
                emit("progress", completed=step + 1, total=total, elapsed_seconds=time.monotonic() - started)
                preview.update(step + 1, _x0, emit,
                               lambda denoised, edge: render_preview(self.torch, denoised, edge))

            samples = self.sample.sample(
                model, noise, settings["steps"], settings["cfg_scale"], settings["sampler"],
                settings["scheduler"], positive, negative, latent, sigmas=sigmas,
                callback=callback, disable_pbar=True, seed=request["seed"])
            emit("phase", phase="saving_output")
            pixels = self.vae.decode(samples)[0].detach().cpu().numpy()
            pixels = self.np.clip(pixels * 255.0, 0, 255).astype(self.np.uint8)
            output = io.BytesIO()
            self.Image.fromarray(pixels).save(output, format="PNG")
            return output.getvalue()
