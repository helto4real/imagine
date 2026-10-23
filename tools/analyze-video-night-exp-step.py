#!/usr/bin/env python3
"""Compare only completed own first-step captures, including actual model inputs."""
import hashlib
import json
from pathlib import Path
import numpy as np

CASE = Path("/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e/exp-step-01")


def read(folder, name):
    meta = json.loads((CASE / folder / (name + ".json")).read_text())
    path = CASE / folder / (name + ".f32")
    assert path.resolve().is_relative_to(CASE / folder) and path.stat().st_size <= 256 * 1024**2
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["sha256"]
    return np.fromfile(path, dtype="<f4").reshape(meta["shape"])


def metrics(x, y):
    if x.shape != y.shape:
        return {"shape_equal": False, "rust_shape": list(x.shape), "python_shape": list(y.shape)}
    a, b = x.astype(np.float64).ravel(), y.astype(np.float64).ravel()
    delta = a - b
    rms = float(np.sqrt(np.mean(b*b)))
    rmse = float(np.sqrt(np.mean(delta*delta)))
    return {"shape_equal": True, "byte_exact": x.tobytes() == y.tobytes(), "mae": float(np.mean(np.abs(delta))), "rmse": rmse, "max": float(np.max(np.abs(delta))), "relative_rmse": rmse / rms if rms else None, "correlation": float(np.corrcoef(a, b)[0, 1]) if a.size > 1 and np.std(a) and np.std(b) else None, "rust_rms": float(np.sqrt(np.mean(a*a))), "python_rms": rms, "finite": bool(np.isfinite(a).all() and np.isfinite(b).all())}


def main():
    assert json.loads((CASE / "run-status-python-correct-mask.json").read_text())["completed"]
    names = ["input-context", "input-tags", "input-latent-0", "input-latent-1", "input-noise-0", "input-noise-1", "input-mask", "input-sigmas", "step1-input-video", "step1-input-audio", "step1-input-context", "step1-input-timestep", "step1-denoise_mask", "step1-audio_denoise_mask", "h3-packed", "h3-time-basis", "h3-rope", "h3-block-0", "h3.sampling.0.x", "h3.sampling.0.denoised"]
    rows = []
    for name in names:
        rows.append({"name": name, **metrics(read("rust", name), read("python-correct-mask", name))})
    x, y = read("rust", names[-1]).flatten(), read("python-correct-mask", names[-1]).flatten()
    split = read("rust", "input-latent-0").size
    streams = {"video": metrics(x[:split], y[:split]), "audio": metrics(x[split:], y[split:])}
    result = {"source": str(CASE), "correct_nested_mask": True, "one_model_forward_each": True, "rows": rows, "denoised_streams": streams, "first_divergence": next((row["name"] for row in rows if not row.get("byte_exact", False)), None)}
    (CASE / "comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
