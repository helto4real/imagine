"""Two-image smoke test: frozen Qwen baseline portrait, then reference editing.

Press Enter after inspecting portrait.png to submit the reference edit. Results
are local ignored artifacts; no ComfyUI, private inputs or external image API.
"""
import argparse
import base64
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reference-max-pixels', type=int, help='native VAE reference pixel budget, rounded to model alignment')
args = parser.parse_args()
if args.reference_max_pixels is not None and args.reference_max_pixels <= 0:
    parser.error('--reference-max-pixels must be positive')
spec = importlib.util.spec_from_file_location('baseline', ROOT / 'run-baseline.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
manifest, flags, env, command, _ = baseline.prepare(os.environ)
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
command[command.index('--listen-port') + 1] = str(port)
(ROOT / 'runs').mkdir(exist_ok=True)
OUT = Path(tempfile.mkdtemp(prefix=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ-reference-edit-'), dir=ROOT / 'runs'))
print(f'OUTPUT {OUT}', flush=True)
portrait_prompt = (
    'Photorealistic editorial head-and-shoulders portrait of one adult woman, age 35. '
    'She has an oval face, green eyes, fine natural freckles across her nose and cheeks, '
    'and shoulder-length wavy copper-auburn hair parted on her left. '
    'She wears small gold hoop earrings, an ivory ribbed turtleneck and a dark forest-green wool coat. '
    'She looks directly at the camera with a relaxed closed-mouth smile. '
    'Her entire head and shoulders are visible, her face is large and sharply focused, '
    'centered in a horizontal composition. Plain warm gray studio background, '
    'soft window light, realistic skin texture, natural proportions, 85mm portrait photography. '
    'One person, opaque photograph, no text, no watermark.'
)
edit_prompt = (
    'Use the woman in the reference image as the same person in a new photorealistic scene. '
    'Preserve her facial identity, face shape, green eyes, freckles, copper-auburn wavy hair, '
    'hair parting and apparent adult age. Keep her small gold hoop earrings, ivory ribbed '
    'turtleneck and dark forest-green wool coat. Change the plain studio background into '
    'a picturesque Scandinavian harbor on a clear autumn morning: wet cobblestones, '
    'red timber waterfront houses, moored wooden fishing boats and calm water. '
    'Show her from the waist up in the foreground, positioned slightly left of center. '
    'Her body is turned slightly toward the harbor while her face looks toward the camera '
    'with the same gentle closed-mouth smile. Keep her face clearly visible and recognizable. '
    'The wider horizontal composition shows the harbor around her, with natural soft sunlight '
    'and realistic depth. One woman only, no other people, no text, no watermark, opaque photograph.'
)
negative = (baseline.BASELINE / 'negative-prompt.txt').read_text()
common = {
    'negative_prompt': negative, 'width': 1600, 'height': 896, 'seed': 42, 'batch_count': 1,
    'sample_params': {'sample_method': 'er_sde', 'scheduler': 'beta', 'sample_steps': 40,
                      'guidance': {'txt_cfg': 2.5}},
    'preview': 'none', 'output_format': 'png', 'embed_image_metadata': False,
}
for name, value in [('portrait-prompt.txt', portrait_prompt), ('edit-prompt.txt', edit_prompt), ('negative-prompt.txt', negative)]:
    (OUT / name).write_text(value)
(OUT / 'plan.json').write_text(json.dumps({
    'baseline': manifest['id'], 'binary_sha256': manifest['binary_sha256'],
    'environment': flags, 'common_request': common,
    'reference': 'portrait.png passed unmodified as a base64 ref_images entry; no init_image',
    'ref_image_args': (f'vae_input_max_pixels={args.reference_max_pixels}' if args.reference_max_pixels else
                       'native defaults; Qwen 2.1 resizes VAE reference to target pixel area'),
    'scope': 'New prompts and editing workload; not an identical-pixel baseline replay',
}, indent=2) + '\n')
(OUT / 'command.sh').write_text(shlex.join(command) + '\n')
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def http(path, body=None):
    request = urllib.request.Request(f'http://127.0.0.1:{port}' + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={'Content-Type': 'application/json'})
    with opener.open(request, timeout=30) as response:
        return json.load(response)

results = []
process = monitor = None
log_path = OUT / 'server.log'
def generate(name, prompt, reference=None):
    body = {**common, 'prompt': prompt}
    if reference is not None:
        body['ref_images'] = [base64.b64encode(reference.read_bytes()).decode()]
        if args.reference_max_pixels:
            body['ref_image_args'] = f'vae_input_max_pixels={args.reference_max_pixels}'
    offset = log_path.stat().st_size
    started = time.perf_counter()
    job = http('/sdcpp/v1/img_gen', body)
    poll = job['poll_url']
    assert poll.startswith('/sdcpp/v1/jobs/') and '://' not in poll
    while True:
        if process.poll() is not None:
            raise RuntimeError('Native server exited; see local server.log')
        state = http(poll)
        if state['status'] in ('completed', 'failed', 'cancelled'):
            break
        if time.perf_counter() - started > 900:
            http(poll + '/cancel', {})
            raise TimeoutError(f'{name} exceeded 15 minutes')
        time.sleep(.1)
    if state['status'] != 'completed':
        raise RuntimeError(f'{name}: {state.get("error", state["status"])}')
    images = state['result']['images']
    assert len(images) == 1
    target = OUT / (name + '.png')
    target.write_bytes(base64.b64decode(images[0]['b64_json'], validate=True))
    saved_seconds = time.perf_counter() - started
    with Image.open(target) as image:
        assert image.size == (1600, 896)
        rgba = image.convert('RGBA')
        image_hash = hashlib.sha256(rgba.tobytes()).hexdigest()
        alpha_range = rgba.getchannel('A').getextrema()
    text = log_path.read_bytes()[offset:].decode(errors='replace')
    (OUT / (name + '.log')).write_text(text)
    def durations(pattern):
        return [float(x) for x in re.findall(pattern, text)]
    result = {
        'name': name, 'saved_seconds': saved_seconds, 'rgba_sha256': image_hash,
        'size': [1600, 896], 'alpha_range': alpha_range,
        'reference_file_sha256': hashlib.sha256(reference.read_bytes()).hexdigest() if reference else None,
        'edit_mode_logged': 'EDIT mode' in text,
        'reference_resizes': re.findall(r'resize vae ref image[^\r\n]*', text),
        'ref_image_args': body.get('ref_image_args', ''),
        'sampling_seconds': durations(r'sampling completed, taking ([\d.]+)s'),
        'conditioning_seconds': durations(r'get_learned_condition completed, taking ([\d.]+)s'),
        'vae_encode_seconds': durations(r'encode_first_stage completed, taking ([\d.]+)s'),
        'vae_decode_seconds': durations(r'decode_first_stage completed, taking ([\d.]+)s'),
        'vae_tiled_retry': 'retrying with spatial tiling' in text,
        'prefix_fallback': 'disabling prefix cache' in text,
    }
    results.append(result)
    (OUT / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    if reference:
        assert result['edit_mode_logged'], 'Reference editing was not confirmed by native logging'
    return target

try:
    with log_path.open('wb', buffering=0) as log, (OUT / 'gpu.csv').open('w') as gpu:
        monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,memory.used,utilization.gpu', '--format=csv', '-lms', '500'], stdout=gpu, stderr=subprocess.DEVNULL)
        started = time.perf_counter()
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=PROJECT)
        while True:
            if process.poll() is not None:
                raise RuntimeError('Native server exited during startup; see local server.log')
            if time.perf_counter() - started > 180:
                raise TimeoutError('Native server startup')
            try:
                capabilities = http('/sdcpp/v1/capabilities')
                break
            except (OSError, urllib.error.URLError):
                time.sleep(.2)
        (OUT / 'startup.json').write_text(json.dumps({'ready_seconds': time.perf_counter() - started}, indent=2) + '\n')
        print('SERVER READY; generating portrait', flush=True)
        portrait = generate('portrait', portrait_prompt)
        input('PORTRAIT READY. Inspect portrait.png, then press Enter to run the reference edit.\n')
        print('EDIT START', flush=True)
        generate('environment', edit_prompt, portrait)
finally:
    for child in (process, monitor):
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
    assert hashlib.sha256((baseline.BASELINE / 'sd-server').read_bytes()).hexdigest() == manifest['binary_sha256']
print(f'DONE {OUT}; owned server and GPU monitor stopped', flush=True)
