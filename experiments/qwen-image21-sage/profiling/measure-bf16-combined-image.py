"""Compare the combined BF16 image with the attention-only and F32 references."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

parser = argparse.ArgumentParser()
parser.add_argument('candidate', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
refs = {
    'attention_only': ('20261013T162749Z-activity-v2vqxp1a', 'b07e26e2c6ea38bb8c69df093d7f4f27c0190784d35b647fbd7463746903e8a8'),
    'f32': ('20261013T145345Z-control-vqy6q9ct', '253d91ada4f98feb50bc9139619d00526594123006d37a9b4589e15cfe43a7f2'),
}
y = np.array(Image.open(args.candidate).convert('RGB'))
assert y.shape == (896, 1600, 3)
result = {'candidate': str(args.candidate.resolve()), 'candidate_rgb_sha256': hashlib.sha256(y.tobytes()).hexdigest(), 'references': {}}
for label, (run, expected) in refs.items():
    path = root / 'runs' / run / 'image.png'
    x = np.array(Image.open(path).convert('RGB'))
    digest = hashlib.sha256(x.tobytes()).hexdigest()
    assert digest == expected and x.shape == y.shape
    delta = x.astype(np.float64) - y
    rmse = float(np.sqrt(np.mean(delta * delta)))
    result['references'][label] = dict(path=str(path), rgb_sha256=digest,
        mae_0_255=float(np.abs(delta).mean()), rmse_0_255=rmse,
        max_abs_0_255=int(np.abs(delta).max()),
        psnr_db=float(20 * np.log10(255 / rmse)) if rmse else None,
        ssim_rgb=float(structural_similarity(x, y, channel_axis=2, data_range=255)),
        fraction_pixels_changed=float(np.any(x != y, axis=2).mean()))
(root / 'runs/bf16-combined-check/image-metrics.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
