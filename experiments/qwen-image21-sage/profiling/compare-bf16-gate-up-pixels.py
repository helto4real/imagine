"""Compare only the synthetic benchmark images from a completed paired test."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

paired = Path(sys.argv[1]).resolve()
rows = json.loads((paired / 'results.json').read_text())
images = [Path(next(r['run'] for r in rows if not r['warmup'] and r['variant'] == v)) / 'image.png'
          for v in (0, 1)]
a, b = [np.asarray(Image.open(p).convert('RGB'), dtype=np.float64) for p in images]
assert a.shape == b.shape == (896, 1600, 3)
diff = b - a
mse = float(np.mean(diff**2))
# Per-channel SSIM: Gaussian 11x11, sigma=1.5, population covariance, valid interior.
filter_args = dict(sigma=(1.5, 1.5, 0), radius=(5, 5, 0))
mu_a, mu_b = [gaussian_filter(x, **filter_args) for x in (a, b)]
var_a = gaussian_filter(a*a, **filter_args) - mu_a*mu_a
var_b = gaussian_filter(b*b, **filter_args) - mu_b*mu_b
cov = gaussian_filter(a*b, **filter_args) - mu_a*mu_b
c1, c2 = (0.01*255)**2, (0.03*255)**2
ssim = ((2*mu_a*mu_b+c1)*(2*cov+c2)) / ((mu_a*mu_a+mu_b*mu_b+c1)*(var_a+var_b+c2))
result = {
    'images': [str(p) for p in images],
    'rgb_sha256': [hashlib.sha256(x.astype(np.uint8).tobytes()).hexdigest() for x in (a, b)],
    'size': [1600, 896],
    'changed_pixels_percent': float(100*np.mean(np.any(diff != 0, axis=2))),
    'changed_channels_percent': float(100*np.mean(diff != 0)),
    'rgb_mae_0_255': float(np.mean(np.abs(diff))),
    'rgb_rmse_0_255': float(np.sqrt(mse)),
    'rgb_max_abs_0_255': float(np.max(np.abs(diff))),
    'psnr_db': float(10*np.log10(255**2/mse)),
    'ssim_rgb_mean': float(np.mean(ssim[5:-5, 5:-5])),
    'ssim_definition': 'Gaussian 11x11 sigma 1.5, per RGB channel, population covariance, valid interior, range 255',
    'interpretation': 'Pixel similarity on one prompt and seed, not a perceptual quality score.'
}
(paired / 'image-difference.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
