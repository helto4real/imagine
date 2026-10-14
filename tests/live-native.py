"""Explicit GPU acceptance test. Only synthetic prompts and this run's own media.

Build first, then run: python3 tests/live-native.py --run
Requires Pillow for RGBA hashing; Python is test tooling, not an API dependency.
The test stops its own API/worker and leaves an ignored evidence directory.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', action='store_true')
args = parser.parse_args()
if not args.run:
    parser.exit(message='Pass --run to perform three real generations and lifecycle checks.\n')
root = Path(__file__).resolve().parents[1]
baseline = root / 'experiments/qwen-image21-sage/baselines/1600x896-v1'
manifest = json.loads((baseline / 'manifest.json').read_text())
(root / '.imagine/test-runs').mkdir(parents=True, exist_ok=True)
out = Path(tempfile.mkdtemp(prefix='native-api-', dir=root / '.imagine/test-runs'))
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
base = f'http://127.0.0.1:{port}'
log = (out / 'server.log').open('w')
proc = subprocess.Popen([str(root / 'start-server.sh'), '--data', str(out / 'data'), '--bind', f'127.0.0.1:{port}'], cwd=root, stdout=log, stderr=log)
access = None
results = {'profile': manifest['id'], 'tests': []}

def api(method, path, data=None, admin=False):
    token = access['admin_token'] if admin else access['clients']['local']
    headers = {'Authorization': f'Bearer {token}'}
    if data is not None:
        headers['Content-Type'] = 'application/json'
        data = json.dumps(data).encode()
    with urllib.request.urlopen(urllib.request.Request(base+path, data=data, headers=headers, method=method), timeout=30) as response:
        raw = response.read()
        return json.loads(raw) if 'json' in response.headers['Content-Type'] else raw

def until(fn, timeout=600):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        result = fn()
        if result:
            return result
        time.sleep(.1)
    raise TimeoutError('acceptance condition timed out')

def finished(job):
    value = api('GET', f'/v1/jobs/{job}')
    return value if value['status'] in ('succeeded', 'failed', 'cancelled', 'interrupted') else None

def consume(job, label, started, expected=None):
    value = until(lambda: finished(job))
    assert value['status'] == 'succeeded', value['error']
    asset = value['output']['assets'][0]['id']
    data = api('GET', f'/v1/assets/{asset}')
    elapsed = time.monotonic()-started
    (out / f'{label}.png').write_bytes(data)
    image = Image.open(io.BytesIO(data)).convert('RGBA')
    assert image.size == (1600,896)
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    if expected:
        assert digest == expected, (label,digest,expected)
    result = {'case':label, 'submission_to_download_seconds':round(elapsed,3), 'execution_to_saved_ms':value['execution_to_saved_ms'],
              'rgba_sha256':digest, 'reference_preprocessing':value['effective']['reference_preprocessing']}
    results['tests'].append(result)
    print(json.dumps(result),flush=True)
    return asset

def unloaded(op,pid):
    value = api('GET',f'/v1/operations/{op}',admin=True)
    if value['status']!='succeeded':
        return None
    state = api('GET','/v1/runtime')
    assert state['paused'] and state['engine']['pid'] is None
    assert not Path(f'/proc/{pid}').exists(), 'owned native process still exists'
    return state

try:
    def ready():
        global access
        if proc.poll() is not None:
            raise RuntimeError('API startup failed; inspect this test run server.log')
        path = out/'data/access.json'
        if not path.exists():
            return False
        access = json.loads(path.read_text())
        try:
            return api('GET','/v1/models')
        except (OSError,urllib.error.URLError):
            return False
    until(ready,180)
    assert api('GET','/v1/runtime')['engine']['pid'] is None
    request = {'model':'qwen-image-2.1-int8','operation':'text_to_image',
               'prompt':(baseline/'prompt.txt').read_text(), 'negative_prompt':(baseline/'negative-prompt.txt').read_text(), 'seed':42}
    start = time.monotonic()
    first = api('POST','/v1/jobs',request)['id']
    asset = consume(first,'baseline',start,manifest['reference_rgba_sha256'])
    pid = api('GET','/v1/runtime')['engine']['pid']
    edit = dict(request, operation='image_edit', prompt='Preserve the scene and composition in the reference image. Change the lighting to a soft golden sunset. Photorealistic, no text, no watermark.',
                inputs={'reference_images':[asset]},model_options={'qwen_image_2_1':{'reference_max_pixels_per_image':1000000}})
    start = time.monotonic()
    second = api('POST','/v1/jobs',edit)['id']
    until(lambda: api('GET',f'/v1/jobs/{second}')['status']=='running')
    queued = api('POST','/v1/jobs',request)['id']
    op = api('POST','/v1/runtime/unload',{'active_job':'wait'},admin=True)['id']
    consume(second,'reference-edit-1mp',start)
    until(lambda: unloaded(op,pid))
    assert api('GET',f'/v1/jobs/{queued}')['status']=='queued'
    api('POST',f'/v1/jobs/{queued}/cancel')
    assert api('GET',f'/v1/jobs/{queued}')['status']=='cancelled'
    results['tests'].append({'case':'unload_wait','worker_exit_confirmed':True,'queued_job_preserved':True})
    api('POST','/v1/runtime/resume',admin=True)
    start = time.monotonic()
    third = api('POST','/v1/jobs',request)['id']
    consume(third,'baseline-after-resume',start,manifest['reference_rgba_sha256'])
    pid2 = api('GET','/v1/runtime')['engine']['pid']
    assert pid2 and pid2 != pid
    fourth = api('POST','/v1/jobs',request)['id']
    until(lambda: api('GET',f'/v1/jobs/{fourth}')['status']=='running')
    op = api('POST','/v1/runtime/unload',{'active_job':'cancel'},admin=True)['id']
    until(lambda: unloaded(op,pid2))
    assert api('GET',f'/v1/jobs/{fourth}')['status']=='cancelled'
    results['tests'].append({'case':'unload_cancel','worker_exit_confirmed':True,'active_job_cancelled':True})
    results['passed'] = True
finally:
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    log.close()
    (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(f'EVIDENCE {out}',flush=True)
