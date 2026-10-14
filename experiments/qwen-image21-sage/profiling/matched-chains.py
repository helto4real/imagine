"""Compare exclusive GPU work in 5600-token target blocks, not profiler wall time.

Native input must first pass analyze.py with the current fusion flags. This
analyzer recognizes that graph by kernel identities, shapes and strict counts.
Kitchen input comes from analyze.py's CUDA-launch-correlated chain scopes.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import subprocess

CHAINS = ('attention_norm_qkv', 'attention_inner', 'attention_output_residual',
          'mlp_norm_gate_up', 'mlp_down_residual')


def native(run):
    summary = json.loads((run/'profile-summary.json').read_text())
    assert summary['dropped_records'] == summary['boundary_crossing_events'] == 0
    logs = [json.loads(x) for x in (run/'log-timestamps.jsonl').read_text().splitlines()]
    start = next(x['monotonic_ns'] for x in logs if 'qwen_image_2_1 executing segment' in x['line'])
    end = next(x['monotonic_ns'] for x in logs if 'sampling completed' in x['line'])
    with (run/'activity.tsv').open() as f:
        rows = list(csv.DictReader((x for x in f if not x.startswith('#')), delimiter='\t'))
    names = sorted({x['name'] for x in rows})
    demangled = subprocess.check_output(['c++filt'], input='\n'.join(names)+'\n', text=True).splitlines()
    assert len(names) == len(demangled)
    names = dict(zip(names, demangled))
    rows = [dict(x, name=names[x['name']], start_ns=int(x['start_ns']), end_ns=int(x['end_ns'])) for x in rows]
    rows = sorted((x for x in rows if start <= x['start_ns'] and x['end_ns'] <= end), key=lambda x: x['start_ns'])
    anchors = [i for i,x in enumerate(rows) if 'norm_mul_convrot_bf16' in x['name']]
    assert len(anchors) == 4992
    chains = defaultdict(float)
    kernels = defaultdict(lambda: dict(count=0, seconds=0.))
    assigned = set()
    for j in range(0, len(anchors), 2):
        a, m = anchors[j:j+2]
        limit = anchors[j+2] if j+2 < len(anchors) else len(rows)
        ends = [i for i in range(m,limit) if 'DownResidual::' in rows[i]['name']]
        assert len(ends) == 1
        z = ends[0]+1
        block = rows[a:z]
        assert rows[a]['grid'] == rows[m]['grid'] == '5600,1,1'
        assert sum('cutlass::bfloat16_t' in x['name'] and 'DefaultGemmWithVisitor' in x['name'] and 'DownResidual::' not in x['name'] for x in block) == 3
        assert sum('rms_norm_rope_apply_bf16' in x['name'] for x in block) == 2
        assert sum('sageattention::' in x['name'] for x in block) == 1
        outputs = [i for i in range(a,m) if 'quantize_rowwise_i8_convrot_cuda<512, false, __nv_bfloat16>' in rows[i]['name']]
        swiglus = [i for i in range(m,z) if 'quantize_rowwise_i8_convrot_cuda<1024, true, float>' in rows[i]['name']]
        assert len(outputs) == len(swiglus) == 1
        out, swiglu = outputs[0], swiglus[0]
        assert sum('DefaultGemmWithVisitor' in x['name'] and x['grid']=='44,96,1' for x in block) == 1
        assert sum('DefaultGemmWithVisitor' in x['name'] for x in block) == 6
        for i in range(a,z):
            x = rows[i]
            # Q/K/V GEMMs are interleaved with RoPE and prefix concatenation.
            if i < out:
                c = CHAINS[0] if i == a or ('DefaultGemmWithVisitor' in x['name'] and 'cutlass::bfloat16_t' in x['name']) else CHAINS[1]
            elif i < m: c = CHAINS[2]
            elif i < swiglu: c = CHAINS[3]
            else: c = CHAINS[4]
            assert i not in assigned
            assigned.add(i)
            seconds = (x['end_ns']-x['start_ns'])/1e9
            chains[c] += seconds
            k = kernels[c,x['name'],x['grid']]
            k['count'] += 1; k['seconds'] += seconds
    total = sum((x['end_ns']-x['start_ns'])/1e9 for x in rows)
    assert abs(total-summary['summed_gpu_activity_seconds']) < 1e-6
    result = dict(blocks=2496, chains_seconds=dict(chains),
                  other_seconds=total-sum(chains.values()), total_gpu_seconds=total,
                  note='78 cached-prefix forwards x 32 target blocks; first two joint prefix/target forwards excluded. Tiny gate setup after a block stays in other.')
    result['kernels'] = [dict(chain=k[0], name=k[1], grid=k[2], **v) for k,v in sorted(kernels.items())]
    return result


def comfy(run):
    s = json.loads((run/'diffusion-profile.json').read_text())
    assert s['unmapped_in_diffusion_window'] == 0
    assert s['target_block_calls'] == 2560 and s['prefix_block_calls'] == 64
    parts = {'attention_norm': CHAINS[0], 'qkv_q': CHAINS[0], 'qkv_k': CHAINS[0], 'qkv_v': CHAINS[0],
             'attention_inner': CHAINS[1], 'attention_output': CHAINS[2], 'attention_gated': CHAINS[2],
             'mlp_norm': CHAINS[3], 'mlp_gate_up': CHAINS[3], 'mlp_down': CHAINS[4],
             'mlp_gated': CHAINS[4], 'mlp_inner': CHAINS[4]}
    chains = defaultdict(float)
    for x in s['chain_parts']:
        if x['domain'] == 'target':
            assert x['part'] in parts, x
            chains[parts[x['part']]] += x['seconds']
    return dict(blocks=s['target_block_calls'], chains_seconds=dict(chains),
                other_seconds=s['summed_gpu_activity_seconds']-sum(chains.values()),
                total_gpu_seconds=s['summed_gpu_activity_seconds'],
                note='80 forwards x 32 target blocks; separate prefix blocks excluded; launch-correlated exclusive GPU activities.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('native_run', type=Path)
    p.add_argument('--comfy-run', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    n = native(a.native_run)
    result = dict(native=n)
    if a.comfy_run:
        c = comfy(a.comfy_run)
        result['comfy'] = c
        result['comparison'] = [dict(chain=k, native_ms_per_block=n['chains_seconds'][k]*1000/n['blocks'],
                                     comfy_ms_per_block=c['chains_seconds'][k]*1000/c['blocks']) for k in CHAINS]
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k=='comparison' or k in ('native','comfy') and not v.get('kernels')}, indent=2))
