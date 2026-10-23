"""Install only the accepted Qwen executable; never accept a new hash implicitly."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
lock = json.loads((HERE / 'source-lock.json').read_text())
manifest = json.loads((ROOT / '.imagine/qwen-default/manifest.json').read_text())
binary = ROOT / manifest['binary_path']
assert hashlib.sha256((HERE / 'source-lock.json').read_bytes()).hexdigest() == manifest['source_lock_sha256']
assert hashlib.sha256(binary.read_bytes()).hexdigest() == manifest['binary_sha256'] == lock['candidate_binary_sha256']
destination = HERE / 'bin/sd-server'
destination.parent.mkdir(exist_ok=True)
shutil.copy2(binary, destination)
print('Installed the accepted Qwen default executable.')
