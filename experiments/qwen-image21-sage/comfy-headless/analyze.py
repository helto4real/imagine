"""Attribute actual GPU activities via CPU launch correlations, never GPU annotations.

Usage: python analyze.py RUN_DIRECTORY
The input is our own torch profiler trace; no ComfyUI history or logs are read.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


def category(event, labels):
    name = event['name']
    if event['cat'] == 'gpu_memcpy':
        if 'DtoD' in name:
            return 'Layout and device copies'
        return 'Host transfers'
    if event['cat'] == 'gpu_memset':
        return 'Other and memset'
    if 'bench::cuda_cutlass_int8_dequant' in labels:
        return 'INT8 GEMM and output dequantization'
    if 'bench::cuda_quantize_int8_rowwise_convrot64' in labels:
        return 'Linear ConvRot and activation quantization'
    if 'bench::cuda_sage_sdpa_prequantized' in labels:
        return 'Sage masked prefix' if '(MaskMode)2' in name else 'Sage core'
    if any(s in name for s in ('quant_qk_fused', 'quant_v_int8_kernel', 'detect_k_anchor')):
        return 'Sage preparation'
    if any(s in name for s in ('copy_kernel', 'Copy', 'transpose')):
        return 'Layout and device copies'
    if any(s in labels for s in ('bench::kitchen_rms_rope', 'bench::kitchen_adaln',
                                 'bench::_gated_residual', 'bench::_modulated_norm')):
        return 'Normalization, RoPE and elementwise'
    if any(s in name for s in ('elementwise', 'layer_norm', 'reduce_kernel')):
        return 'Normalization, RoPE and elementwise'
    return 'Other and memset'


def union_us(events):
    end = float('-inf')
    total = 0.
    for e in sorted(events, key=lambda x: x['ts']):
        stop = e['ts'] + e['dur']
        total += max(0., stop - max(end, e['ts']))
        end = max(end, stop)
    return total


def analyze(run, profile_index=3):
    trace = json.loads((run/f'trace-{profile_index}.json').read_text())['traceEvents']
    threads = defaultdict(list)
    for e in trace:
        if e.get('cat') in ('cpu_op', 'user_annotation', 'cuda_runtime', 'cuda_driver') and e.get('ph') == 'X':
            threads[e['pid'], e['tid']].append(e)
    external, launches = {}, {}
    for events in threads.values():
        stack = []
        for e in sorted(events, key=lambda x: (x['ts'], -x['dur'])):
            while stack and e['ts'] >= stack[-1]['ts'] + stack[-1]['dur']:
                stack.pop()
            if e['cat'] == 'user_annotation' and e['name'].startswith('bench::'):
                stack.append(e)
            labels = tuple(x['name'] for x in stack)
            args = e.get('args', {})
            if 'External id' in args:
                external[args['External id']] = labels
            if e['cat'] in ('cuda_runtime', 'cuda_driver') and 'correlation' in args:
                assert args['correlation'] not in launches, 'Ambiguous CUDA correlation'
                launches[args['correlation']] = labels

    selected, unmatched = [], []
    groups = defaultdict(lambda: dict(count=0, seconds=0.))
    kernels = defaultdict(lambda: dict(count=0, seconds=0.))
    scopes = defaultdict(lambda: dict(count=0, seconds=0.))
    parts = defaultdict(lambda: dict(count=0, seconds=0.))
    mapping = Counter()
    for e in trace:
        if e.get('cat') not in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            continue
        args = e.get('args', {})
        if args.get('correlation') in launches:
            labels = launches[args['correlation']]
            mapping['runtime_correlation'] += 1
        elif args.get('External id') in external:
            labels = external[args['External id']]
            mapping['external_id_fallback'] += 1
        else:
            unmatched.append(e)
            continue
        if 'bench::diffusion_forward' not in labels:
            continue
        selected.append(e)
        part = next((x.removeprefix('bench::part_') for x in reversed(labels) if x.startswith('bench::part_')), 'other')
        domain = 'target' if 'bench::target_block' in labels else 'prefix' if 'bench::prefix_block' in labels else 'outside_blocks'
        if part == 'other' and 'bench::attention_chain' in labels:
            part = 'attention_inner'
        if part == 'other' and 'bench::mlp_chain' in labels:
            part = 'mlp_inner'
        item=parts[(domain,part,category(e,labels))];item['count']+=1;item['seconds']+=e['dur']/1e6
        group = category(e, labels)
        for bucket in (groups[group], kernels[(group, e['name'])], scopes[labels[-1]]):
            bucket['count'] += 1
            bucket['seconds'] += e['dur']/1e6
    assert selected, 'No diffusion GPU activities found'
    start = min(e['ts'] for e in selected)
    end = max(e['ts'] + e['dur'] for e in selected)
    unknown_in_window = [e for e in unmatched if e['ts'] < end and e['ts'] + e['dur'] > start]
    assert not unknown_in_window, 'Unmapped GPU activities overlap diffusion'
    forward_count = sum(e.get('cat') == 'user_annotation' and e['name'] == 'bench::diffusion_forward' for e in trace)
    assert forward_count == 80
    assert groups['Sage core']['count'] == 2560
    assert groups['Sage masked prefix']['count'] == 64
    assert groups['INT8 GEMM and output dequantization']['count'] == 15744
    assert groups['Linear ConvRot and activation quantization']['count'] == 15744
    summary = dict(method='GPU activity -> CUDA launch correlation -> nested CPU bench scopes; external-id fallback if required',
                   forwards=forward_count, activity_count=len(selected),
                   mapping_all_phases=dict(mapping), unmapped_all_phases=len(unmatched),
                   unmapped_in_diffusion_window=len(unknown_in_window),
                   gpu_activity_span_seconds=(end-start)/1e6,
                   summed_gpu_activity_seconds=sum(e['dur'] for e in selected)/1e6,
                   gpu_activity_union_seconds=union_us(selected)/1e6,
                   groups=dict(groups), exclusive_innermost_scopes=dict(scopes))
    summary['profile_index']=profile_index
    summary['target_block_calls']=sum(e.get('cat')=='user_annotation' and e['name']=='bench::target_block' for e in trace)
    summary['prefix_block_calls']=sum(e.get('cat')=='user_annotation' and e['name']=='bench::prefix_block' for e in trace)
    summary['chain_parts']=[dict(domain=k[0],part=k[1],category=k[2],**v) for k,v in sorted(parts.items())]
    assert abs(sum(x['seconds'] for x in parts.values())-summary['summed_gpu_activity_seconds'])<1e-6
    (run/'diffusion-profile.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (run/'diffusion-kernels.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['category', 'count', 'seconds', 'kernel'])
        for (group, name), row in sorted(kernels.items(), key=lambda x: -x[1]['seconds']):
            writer.writerow([group, row['count'], row['seconds'], name])
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--profile-index',type=int,default=3)
    args=parser.parse_args()
    analyze(args.run_directory,args.profile_index)
