#!/usr/bin/env python3
"""Static, hash-locked inventory; never imports ComfyUI or scans runtime files."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from reference_paths import reference_dir

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--comfy', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
project = Path(__file__).resolve().parents[2]
lock = reference_dir(project)
manifest = json.loads((lock / 'source-manifest.json').read_text())
groups = {
    'sampling': ['comfy/samplers.py', 'comfy/sample.py', 'comfy/sampler_helpers.py', 'comfy/model_sampling.py', 'comfy/k_diffusion/sampling.py', 'comfy/k_diffusion/deis.py', 'comfy/k_diffusion/sa_solver.py', 'comfy/k_diffusion/utils.py', 'comfy/extra_samplers/uni_pc.py'],
    'conditioning': ['comfy/conds.py', 'comfy_extras/nodes_cfg.py', 'comfy_extras/nodes_tcfg.py', 'comfy_extras/nodes_perpneg.py', 'comfy_extras/nodes_differential_diffusion.py'],
    'patching': ['comfy/model_patcher.py', 'comfy/patcher_extension.py', 'comfy/hooks.py', 'comfy/lora.py', *['comfy/weight_adapter/' + name + '.py' for name in ['__init__', 'base', 'bypass', 'lora', 'loha', 'lokr', 'glora', 'oft', 'boft']]],
    'loading_memory': ['comfy/sd.py', 'comfy/model_detection.py', 'comfy/model_management.py', 'comfy/memory_management.py', 'comfy/quant_ops.py', 'comfy/utils.py', 'comfy/float.py', 'comfy/system_memory.py'],
    'inference_extras': ['comfy_extras/nodes_model_advanced.py', 'comfy_extras/nodes_custom_sampler.py', 'comfy_extras/nodes_gits.py', 'comfy_extras/nodes_align_your_steps.py', 'comfy_extras/nodes_advanced_samplers.py'],
}
files = []
trees = {}
for group, paths in groups.items():
    for path in paths:
        source = a.comfy / path
        expected = next(e['sha256'] for e in manifest if e['root'] == 'comfy' and e['path'] == path)
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, path
        tree = ast.parse(source.read_text())
        trees[path] = tree
        definitions = []
        def walk(body, owner=''):
            for n in body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    name = owner + n.name
                    definitions.append(dict(name=name, line=n.lineno, end_line=n.end_lineno, kind=type(n).__name__, status='inventory_only'))
                    if isinstance(n, ast.ClassDef): walk(n.body, name + '.')
        walk(tree.body)
        imports = [ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        files.append(dict(group=group, path=path, sha256=expected, imports=imports, definitions=definitions))
tree = trees['comfy/samplers.py']
samplers = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'KSAMPLER_NAMES' for t in n.targets)) + ['ddim', 'uni_pc', 'uni_pc_bh2']
# This is an inventory, not an executable support declaration.
result = dict(schema=1, reference=lock.name, scope='shared algorithms; graph and UI schemas excluded; model forward implementations belong to Stage 4', samplers=samplers, files=files)
a.output.write_text(json.dumps(result, indent=2) + '\n')
print(f'Inventoried {len(files)} source modules and {sum(len(f["definitions"]) for f in files)} definitions, {len(samplers)} samplers')
