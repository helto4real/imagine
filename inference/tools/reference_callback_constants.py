"""Frozen callback names for own source-body oracles, without core imports."""
import ast
import hashlib
from types import SimpleNamespace

def _constants(comfy, manifest, class_name):
    relative = 'comfy/patcher_extension.py'
    path = comfy / relative
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == next(v['sha256'] for v in manifest if v['root']=='comfy' and v['path']==relative)
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name)
    values = {}
    for node in cls.body:
        if isinstance(node,ast.Assign):
            assert len(node.targets)==1 and isinstance(node.targets[0],ast.Name)
            values[node.targets[0].id] = ast.literal_eval(node.value)
    return SimpleNamespace(**values), digest

def callback_constants(comfy, manifest):
    return _constants(comfy, manifest, 'CallbacksMP')

def wrapper_constants(comfy, manifest):
    return _constants(comfy, manifest, 'WrappersMP')
