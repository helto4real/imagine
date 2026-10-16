"""Build and compare deterministic synthetic CUDA graphs, flag off/on.

This reads only synthetic output bytes and test-only kernel counters, never
model runtime logs, images or prompt/job data. --run explicitly uses the GPU.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / '.imagine/lora-output'
BUILD = OUT / 'build'
SOURCE = OUT / 'source'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--trace-library', type=Path)
    args = parser.parse_args()
    obj = OUT / 'test-lora-output.o'
    binary = OUT / 'test-lora-output'
    subprocess.run(['/usr/bin/g++', '-std=c++17', '-O2', '-Wall', '-Wextra',
        '-I' + str(SOURCE / 'ggml/include'), '-c', str(Path(__file__).with_suffix('.cpp')),
        '-o', str(obj)], check=True)
    link = subprocess.check_output(['ninja', '-t', 'commands', 'bin/sd-server'], cwd=BUILD, text=True).splitlines()[-1]
    assert link.startswith(': && ') and link.endswith(' && :')
    cmd = [s for s in shlex.split(link[5:-5]) if not s.endswith('.o') and not s.startswith('-Wl,--dependency-file=')]
    cmd[cmd.index('-o') + 1] = str(binary)
    cmd.insert(1, str(obj))
    subprocess.run(cmd, cwd=BUILD, check=True)
    if not args.run:
        print('Built synthetic check:', binary)
        return
    state = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free,utilization.gpu',
                                     '--format=csv,noheader,nounits'], text=True).strip()
    free, utilization = map(float, state.split(','))
    if free < 13000 or utilization >= 20:
        raise RuntimeError('GPU is busy; no existing process was stopped')
    results = []
    for flag in ['0', '1']:
        output = OUT / ('synthetic-' + flag + '.bin')
        env = {k: v for k, v in os.environ.items() if not k.startswith(('SD_', 'GGML_', 'QWEN_')) and k != 'LD_PRELOAD'}
        env['GGML_CUDA_FUSED_LORA_OUTPUT'] = flag
        if args.trace_library:
            env['LD_PRELOAD'] = str(args.trace_library.resolve(strict=True))
            env['QWEN_ACTIVITY_TRACE'] = str(OUT / ('synthetic-' + flag + '.tsv'))
        subprocess.run([str(binary), str(output)], env=env, check=True, stderr=subprocess.DEVNULL)
        with output.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        results.append({'flag': flag, 'bytes': output.stat().st_size, 'sha256': digest})
        if args.trace_library:
            fused_calls = 0
            checkpoints = []
            with (OUT / ('synthetic-' + flag + '.tsv')).open() as f:
                for line in f:
                    if line.startswith('#'):
                        checkpoints.append(line.strip())
                    elif 'lora_output_f16_f32' in line:
                        fused_calls += 1
            assert checkpoints == ['# dropped_records=0'], checkpoints
            assert fused_calls == (10 if flag == '1' else 0), fused_calls
            results[-1]['fused_calls'] = fused_calls
    assert results[0]['sha256'] == results[1]['sha256'], results
    assert results[0]['bytes'] == results[1]['bytes']
    print('PASS: all synthetic output and retained-intermediate bytes identical.')
    (OUT / 'synthetic-summary.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
