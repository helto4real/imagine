"""Whole-trace, exact-name/count checks for the fixed Qwen split-prefix experiment.

This deliberately reports only identifiable Sage preparation/copy kernels. It does
not use approximate parent-pipe phase timestamps or claim total diffusion time.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
parser.add_argument('mode', type=int, choices=(0, 1, 2))
parser.add_argument('--gate-up-swiglu', action='store_true')
parser.add_argument('--gate-up-direct', action='store_true')
args = parser.parse_args()
assert not args.gate_up_direct or args.gate_up_swiglu
run = args.run
lines = (run / 'activity.tsv').read_text().splitlines()
assert lines[-1] == '# dropped_records=0'
rows = list(csv.DictReader((line for line in lines if not line.startswith('#')), delimiter='\t'))
assert all(int(r['end_ns']) >= int(r['start_ns']) > 0 for r in rows)
names = sorted({r['name'] for r in rows})
demangled = subprocess.check_output(['c++filt', '--no-recurse-limit'], input='\n'.join(names)+'\n', text=True).splitlines()
assert len(names) == len(demangled)
name_map = dict(zip(names, demangled))
for row in rows:
    row['name'] = name_map[row['name']]
logs = [json.loads(line) for line in (run / 'log-timestamps.jsonl').read_text().splitlines()]
assert sum('qwen_image_2_1 executing segment 1/1' in x['line'] for x in logs) == 80
result = {'scope': 'whole-trace named diffusion kernels; no phase-window total',
          'mode': args.mode, 'dropped_records': 0, 'kernels': {}}
def selected(key, pattern, count, exclude=(), grids=()):
    matches = [r for r in rows if r['kind'] == 'kernel' and pattern in r['name'] and not any(x in r['name'] for x in exclude) and (not grids or r['grid'] in grids)]
    assert len(matches) == count, (key, len(matches), count)
    item = {'count': len(matches), 'seconds': sum((int(r['end_ns'])-int(r['start_ns']))/1e9 for r in matches)}
    result['kernels'][key] = item
    return item
selected('attention_core', 'sageattention::', 2560)
selected('key_mean', 'sage_key_mean<128, __nv_bfloat16,', 2560)
selected('key_quant', 'sage_quantize_per_thread<128, true, __nv_bfloat16,', 2560)
selected('query_quant', 'sage_quantize_per_thread<128, false, __nv_bfloat16,', 2560)
selected('value_scale', 'sage_value_scale<128, __nv_bfloat16,', 2560)
selected('value_quant', 'sage_quantize_value<128, __nv_bfloat16,', 2560)
selected('key_concat_and_initial_result_join', 'concat_cont<unsigned short, 1>', 64 if args.mode == 2 else 2560)
selected('value_concat', 'concat_cont<unsigned short, 2>', 0 if args.mode >= 1 else 2496)
selected('split_key_mean', 'sage_key_mean<128, __nv_bfloat16, true>', 2496 if args.mode == 2 else 0)
selected('split_key_quant', 'sage_quantize_per_thread<128, true, __nv_bfloat16, true>', 2496 if args.mode == 2 else 0)
selected('split_value_scale', 'sage_value_scale<128, __nv_bfloat16, true>', 2496 if args.mode >= 1 else 0)
selected('split_value_quant', 'sage_quantize_value<128, __nv_bfloat16, true>', 2496 if args.mode >= 1 else 0)
selected('paired_qk_norm_rope', 'paired_rms_norm_rope_apply_bf16', 2560)
selected('norm_convrot', 'norm_mul_convrot_bf16', 4992)
selected('down_residual', 'DownResidual::', 2496, ('AttentionResidual::',))
selected('attention_residual', 'AttentionResidual::', 2496)
selected('f32_swiglu_convrot', 'quantize_rowwise_i8_convrot_cuda<1024, true, float>', 0 if args.gate_up_swiglu else 2560)
selected('gate_up_swiglu', 'VisitorSwiGLUStore', 2560 if args.gate_up_swiglu else 0)
selected('gate_up_pack', 'gate_up_swiglu_pack', 2560 if args.gate_up_swiglu and not args.gate_up_direct else 0)
selected('gate_up_direct', 'GateUpDirect::WeightPermutation', 2560 if args.gate_up_direct else 0)
if args.gate_up_swiglu:
    selected('post_swiglu_convrot', 'quantize_rowwise_i8_convrot_cuda<1024, false, float>', 2560, grids=('5600,1,1','5657,1,1','5830,1,1'))
selected('bf16_swiglu_convrot', 'quantize_rowwise_i8_convrot_cuda<1024, true, __nv_bfloat16>', 0)
# Check the exact accepted GEMM paths without including the text encoder's GEMMs.
for dtype, columns, expected in [('float', 96, 2560), ('cutlass::bfloat16_t', 96, 0), ('cutlass::bfloat16_t', 16, 7680)]:
    hits = [r for r in rows if r['kind'] == 'kernel' and ('DefaultGemmWithVisitor' in r['name'] or 'GateUpDirect::WeightPermutation' in r['name'])
            and not any(x in r['name'] for x in ('AttentionResidual::', 'DownResidual::'))
            and int(r['grid'].split(',')[1]) == columns
            and (('cutlass::bfloat16_t' in r['name']) == (dtype == 'cutlass::bfloat16_t'))]
    assert len(hits) == expected, (dtype, columns, len(hits))
gate_rows = [r for r in rows if r['kind'] == 'kernel' and ('DefaultGemmWithVisitor' in r['name'] or 'GateUpDirect::WeightPermutation' in r['name']) and int(r['grid'].split(',')[1]) == 96]
assert len(gate_rows) == 2560
result['kernels']['gate_up_gemm_total'] = {'count': len(gate_rows), 'seconds': sum((int(r['end_ns'])-int(r['start_ns']))/1e9 for r in gate_rows)}
result['gate_up_swiglu_convrot_seconds'] = result['kernels']['gate_up_gemm_total']['seconds'] + result['kernels']['gate_up_pack']['seconds'] + result['kernels']['post_swiglu_convrot' if args.gate_up_swiglu else 'f32_swiglu_convrot']['seconds']
result['kv_preparation_and_concats_seconds'] = sum(result['kernels'][key]['seconds'] for key in
    ('key_mean', 'key_quant', 'value_scale', 'value_quant', 'key_concat_and_initial_result_join', 'value_concat'))
(run / 'sage-split-summary.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
