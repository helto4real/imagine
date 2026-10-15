"""Build the verified stage-2 candidate and isolated checks; preserve baseline v1."""
from pathlib import Path
import hashlib, json, shlex, shutil, subprocess
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'stable-diffusion.cpp'
BUILD = SRC / 'build-cuda'
OUT = ROOT / 'experiments/qwen-image21-sage/runs/vae-stage2-strips-build'
BASE = ROOT / 'experiments/qwen-image21-sage/baselines/1600x896-v1'
EVIDENCE = ROOT / 'experiments/qwen-image21-sage/evidence/vae-stage2-strips'
OUT.mkdir(exist_ok=True)
expected = json.loads((BASE / 'after-hashes.json').read_text())
candidate = json.loads((EVIDENCE / 'source-hashes.json').read_text())
expected['stable-diffusion.cpp/' + candidate['file']] = candidate['candidate']
for name, digest in expected.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
with (OUT / 'native-build.log').open('w') as log:
    subprocess.run(['cmake', '--build', str(BUILD), '--target', 'sd-server', '--parallel', '2'],
                   stdout=log, stderr=subprocess.STDOUT, check=True)
shutil.copy2(BUILD / 'bin/sd-server', OUT / 'sd-server')
(OUT / 'binary-hashes.json').write_text(json.dumps({'sd-server': hashlib.sha256((OUT / 'sd-server').read_bytes()).hexdigest()}, indent=2) + '\n')
(OUT / 'after-hashes.json').write_text(json.dumps(expected, indent=2) + '\n')
flags = json.loads((BASE / 'baseline-environment.json').read_text())
(OUT / 'baseline-environment.json').write_text(json.dumps({**flags, 'SD_VAE_STRIP_STAGE2': '1'}, indent=2) + '\n')
entries = json.loads((BUILD / 'compile_commands.json').read_text())
entry = next(e for e in entries if e['file'].endswith('/src/core/ggml_extend.cpp'))
link = subprocess.check_output(['ninja', '-t', 'commands', 'bin/sd-server'], cwd=BUILD, text=True).splitlines()[-1]
assert link.startswith(': && ') and link.endswith(' && :')
base = [s for s in shlex.split(link[5:-5]) if not s.endswith('.o') and not s.startswith('-Wl,--dependency-file=')]
commands = []
for name in ['measure-vae-stage2', 'test-vae-stage2-strips']:
    source = ROOT / 'experiments/qwen-image21-sage/profiling' / (name + '.cpp')
    obj = OUT / (name + '.o')
    cmd = shlex.split(entry['command'])
    cmd.remove('-DNDEBUG')
    cmd[cmd.index('-o') + 1] = str(obj)
    cmd[cmd.index('-c') + 1] = str(source)
    commands.append(cmd)
    with (OUT / (name + '.build.log')).open('w') as log:
        subprocess.run(cmd, cwd=BUILD, stdout=log, stderr=subprocess.STDOUT, check=True)
        cmd = base.copy()
        cmd[cmd.index('-o') + 1] = str(OUT / name)
        cmd.insert(1, str(obj))
        commands.append(cmd)
        subprocess.run(cmd, cwd=BUILD, stdout=log, stderr=subprocess.STDOUT, check=True)
(OUT / 'check-build-commands.json').write_text(json.dumps(commands, indent=2) + '\n')
print(OUT)
