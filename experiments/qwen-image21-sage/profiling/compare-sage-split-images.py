"""Compare contiguous Sage inputs, split V, and split V+K with fixed F32 gate/up."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
from PIL import Image

experiment = Path(__file__).resolve().parents[1]
build = experiment / 'runs/sage-split-kv-build'
binary = build / 'sd-cli'
expected_binary = '041cfc02dff7d58d328af73e7e8a9c240c504cc7e4889305ec9172d29a90e900'
expected_pixels = 'c672ca9049d2826d3c713c3ead1f41edaca4498e201b0249663341f6cbd7db1c'
assert hashlib.sha256(binary.read_bytes()).hexdigest() == expected_binary
out = Path(tempfile.mkdtemp(prefix=datetime.datetime.now(datetime.timezone.utc).strftime(
    '%Y%m%dT%H%M%SZ-sage-split-images-'), dir=experiment / 'runs'))
shutil.copyfile(__file__, out / Path(__file__).name)
shutil.copyfile(experiment / 'profiling/run.py', out / 'replay-run.py')
orders = [[0, 1, 2], [2, 0, 1], [1, 2, 0]]
(out / 'plan.json').write_text(json.dumps({'orders': orders, 'excluded_warmup': 0,
    'cpu_threads': 11, 'default_vram': True, 'binary': str(binary),
    'binary_sha256': expected_binary}, indent=2) + '\n')
print(out, flush=True)
results = []
reference_config = None
base_env = json.loads((experiment / 'runs/20261013T203045Z-activity-q759cwrf/trace-environment.json').read_text())
for key in ('LD_PRELOAD', 'QWEN_ACTIVITY_TRACE'):
    base_env.pop(key, None)
sequence = [(0, 0)] + [(block, mode) for block, order in enumerate(orders, 1) for mode in order]
for index, (block, mode) in enumerate(sequence):
    print(f'START {index+1}/{len(sequence)} block={block} mode={mode}', flush=True)
    env = {**os.environ, **base_env, 'SD_SAGE_SPLIT_V': str(int(mode >= 1)),
           'SD_SAGE_SPLIT_K': str(int(mode >= 2)), 'SD_QWEN21_BF16_GATE_UP': '0',
           'PYTHONDONTWRITEBYTECODE': '1'}
    assert not any(env.get(k) for k in ('LD_PRELOAD', 'QWEN_ACTIVITY_TRACE', 'QWEN_COPY_STACKS'))
    cmd = [sys.executable, str(experiment / 'profiling/run.py'), '--binary', str(binary),
           '--threads', '11', '--default-vram']
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    (out / f'replay-{index:02d}.log').write_text(proc.stdout + proc.stderr)
    proc.check_returncode()
    item = json.loads(proc.stdout.splitlines()[-1])
    run = Path(item['run'])
    assert (run / 'binary-sha256.txt').read_text().strip() == expected_binary
    command = shlex.split((run / 'command.sh').read_text())
    assert '--max-vram' not in command and command[command.index('--auto-fit')+1] == 'on'
    log = (run / 'generation.log').read_text()
    assert int(re.search(r'n_threads: (\d+)', log)[1]) == 11
    normalized = command.copy()
    for flag in ('--output', '--prompt-file', '--negative-prompt-file'):
        normalized[normalized.index(flag)+1] = '<run-local>'
    config = {'command': normalized,
              'prompt': hashlib.sha256((run / 'prompt.txt').read_bytes()).hexdigest(),
              'negative': hashlib.sha256((run / 'negative-prompt.txt').read_bytes()).hexdigest(),
              'environment': json.loads((run / 'trace-environment.json').read_text())}
    assert config['environment'].pop('SD_SAGE_SPLIT_V') == str(int(mode >= 1))
    assert config['environment'].pop('SD_SAGE_SPLIT_K') == str(int(mode >= 2))
    assert config['environment']['SD_QWEN21_BF16_GATE_UP'] == '0'
    if reference_config is None:
        reference_config = config
    assert config == reference_config, 'Configuration drift between variants'
    result = json.loads((run / 'result.json').read_text())
    stamps = [json.loads(line) for line in (run / 'log-timestamps.jsonl').read_text().splitlines()]
    saved = next(s for s in stamps if 'save result image 0' in s['line'] and '(success)' in s['line'])
    image = Image.open(run / 'image.png').convert('RGB')
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    validation = {'mode': 'RGB', 'size': image.size, 'sha256': digest,
                  'bit_identical_reference': digest == expected_pixels}
    (run / 'image-validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    def seconds(pattern):
        return float(re.search(pattern, log)[1])
    item.update(block=block, warmup=block == 0, variant=mode, image=validation,
                saved_seconds=(saved['monotonic_ns'] - result['start_ns']) / 1e9,
                sampling_seconds=seconds(r'sampling completed, taking ([\d.]+)s'),
                conditioning_seconds=seconds(r'get_learned_condition completed, taking ([\d.]+)s'),
                vae_seconds=seconds(r'decode_first_stage completed, taking ([\d.]+)s'))
    results.append(item)
    (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(item), flush=True)
    assert image.size == (1600, 896) and digest == expected_pixels, 'Image mismatch; evidence saved'
summary = {}
for mode in range(3):
    rows = [r for r in results if not r['warmup'] and r['variant'] == mode]
    summary[mode] = {key: {'mean': statistics.mean(r[key] for r in rows),
                         'median': statistics.median(r[key] for r in rows),
                         'min': min(r[key] for r in rows), 'max': max(r[key] for r in rows)}
                     for key in ('saved_seconds', 'wall_seconds', 'sampling_seconds',
                                 'conditioning_seconds', 'vae_seconds')}
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print('COMPLETE ' + str(out), flush=True)
