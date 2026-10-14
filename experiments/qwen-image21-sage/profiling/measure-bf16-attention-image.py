"""Compare this experiment's candidate to the fixed native F32 benchmark image."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
p=argparse.ArgumentParser();p.add_argument('candidate',type=Path);a=p.parse_args()
root=Path(__file__).resolve().parents[1]
baseline=root/'runs/20261013T145345Z-control-vqy6q9ct/image.png'
x=np.array(Image.open(baseline).convert('RGB'));y=np.array(Image.open(a.candidate).convert('RGB'))
assert x.shape==y.shape==(896,1600,3)
d=x.astype(np.float64)-y;rmse=float(np.sqrt(np.mean(d*d)))
out=dict(baseline=str(baseline),candidate=str(a.candidate.resolve()),baseline_rgb_sha256=hashlib.sha256(x.tobytes()).hexdigest(),candidate_rgb_sha256=hashlib.sha256(y.tobytes()).hexdigest(),mae_0_255=float(np.abs(d).mean()),rmse_0_255=rmse,max_abs_0_255=int(np.abs(d).max()),psnr_db=float(20*np.log10(255/rmse)),ssim_rgb=float(structural_similarity(x,y,channel_axis=2,data_range=255)),fraction_pixels_changed=float(np.any(x!=y,axis=2).mean()))
assert out['baseline_rgb_sha256']=='253d91ada4f98feb50bc9139619d00526594123006d37a9b4589e15cfe43a7f2'
(root/'runs/bf16-attention-build/image-metrics.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
