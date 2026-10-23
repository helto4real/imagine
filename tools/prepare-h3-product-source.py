#!/usr/bin/env python3
"""Build only H3 adapter changes on the approved frozen native base."""
from pathlib import Path
import subprocess,tarfile,io,shutil
root=Path(__file__).resolve().parents[1]
dest=root/'comfyui/h3-build/source'
base='cd0ef53be69dd539ef0ffdf0cb03f4c1c5835fc7'
if not dest.exists():
 dest.mkdir(parents=True)
 data=subprocess.check_output(['git','archive',base,'inference','shared','design/h3-stage4-evidence/selected-model-sha256.json'],cwd=root)
 with tarfile.open(fileobj=io.BytesIO(data))as archive:archive.extractall(dest,filter='data')
paths=['inference/examples/h3_res20_cpu.rs','inference/src/models/minimax_h3/sampling.rs','shared/h3_protocol.rs','shared/h3_geometry.rs','shared/h3_sampling.rs','shared/h3_continuation.rs','shared/h3_references.rs','design/h3-i2v-model-lock.json']+[f'inference/src/h3_product/{name}.rs'for name in ['mod','native','lifecycle','memory_budget','deployment','wire','image','sampling','motion','references','reference_video','reference_window']]
# Protocol 16's product components and text residency dependencies must travel
# with the adapter. The remaining frozen inference core stays at the base above.
paths += [str(path.relative_to(root)) for pattern in (
 'inference/src/h3_product/*.rs',
 'inference/src/models/minimax_h3/sampling/*.rs',
 'shared/h3_*.rs',
) for path in sorted(root.glob(pattern))]
paths += [
 'inference/src/checkpoint/prepared.rs',
 'inference/src/checkpoint/dynamic.rs',
 'inference/src/checkpoint/dynamic_vbar.rs',
 'inference/src/models/minimax_h3/checkpoint.rs',
 'inference/src/models/qwen_image21/layers.rs',
 'inference/src/models/qwen_image21/text_encoder.rs',
 'inference/src/models/qwen_image21/vision_encoder.rs',
 'shared/image_preview_rgb.rs',
]
for name in dict.fromkeys(paths):
 target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,target)
# Keep later generic checkpoint-parity modules on main out of this frozen build.
# Apply only the product's Aimdo path/dynamic-text additions to its pinned base.
checkpoint='inference/src/checkpoint.rs'
(dest/checkpoint).write_bytes(subprocess.check_output(['git','show',f'{base}:{checkpoint}'],cwd=root))
subprocess.run(['git','apply','--unsafe-paths','--directory',str(dest),str(root/'native/h3-product/checkpoint.patch')],cwd=root,check=True)
(dest/'inference/src/lib.rs').touch()
