"""Profile both gate/up precision variants with the paired test's exact environment."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from PIL import Image

experiment = Path(__file__).resolve().parents[1]
paired = Path(sys.argv[1]).resolve()
results = json.loads((paired / 'results.json').read_text())
assert len(results) == 7
binary = experiment / 'runs/bf16-gate-up-build-v2/sd-cli'
expected = json.loads((binary.parent / 'binary-hashes.json').read_text())['sd-cli']
assert hashlib.sha256(binary.read_bytes()).hexdigest() == expected
profiles = []
variants = [int(sys.argv[2])] if len(sys.argv) > 2 else [0, 1]
assert all(v in (0, 1) for v in variants)
for variant in variants:
    ref = next(r for r in results if not r['warmup'] and r['variant'] == variant)
    env = {**os.environ, **json.loads((Path(ref['run']) / 'trace-environment.json').read_text()),
           'PYTHONDONTWRITEBYTECODE': '1'}
    assert env['SD_QWEN21_BF16_GATE_UP'] == str(variant)
    cmd = [sys.executable, str(experiment / 'profiling/run.py'), '--binary', str(binary),
           '--threads', '11', '--default-vram', '--trace-library',
           str(experiment / 'runs/activity-profiler/activity.so')]
    run = subprocess.run(cmd, env=env, capture_output=True, text=True)
    (paired / f'profile-{variant}.log').write_text(run.stdout + run.stderr)
    run.check_returncode()
    item = json.loads(run.stdout.splitlines()[-1])
    path = Path(item['run'])
    assert (path / 'binary-sha256.txt').read_text().strip() == expected
    rgb = Image.open(path / 'image.png').convert('RGB')
    assert rgb.size == (1600, 896)
    digest = hashlib.sha256(rgb.tobytes()).hexdigest()
    assert digest == ref['image']['sha256']
    flags = ['--fused-gate-up', '--fused-projection', '--fused-down', '--fused-rms-rope',
             '--fused-norm-modulation', '--fused-gated-residual', '--bf16-residual',
             '--bf16-attention', '--fused-norm-convrot', '--fused-down-residual',
             '--fused-attention-residual', '--paired-rms-rope']
    if variant:
        flags.append('--bf16-gate-up')
    analyze = [sys.executable, str(experiment / 'profiling/analyze.py'), str(path), *flags]
    checked = subprocess.run(analyze, capture_output=True, text=True)
    (paired / f'profile-analysis-{variant}.log').write_text(checked.stdout + checked.stderr)
    checked.check_returncode()
    item.update(variant=variant, image_sha256=digest, analyze_command=analyze)
    profiles.append(item)
    (paired / 'profiles.json').write_text(json.dumps(profiles, indent=2) + '\n')
    print(json.dumps(item), flush=True)
