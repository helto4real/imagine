#!/usr/bin/env python3
"""Preserve explicitly named isolated Stage 2 evidence and implementation hashes."""
import argparse,hashlib,json,re,shutil
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--runtime',type=Path,required=True)
p.add_argument('--finalize-selected-stage2',action='store_true',help='Require complete selected-run and current isolation evidence before recording the Stage 2 vertical-path exit')
a=p.parse_args()
project=Path(__file__).resolve().parents[2]
destination=project/'design/stage-2-qwen-image21-precision-sampling-evidence'
destination.mkdir(exist_ok=True)
runs=['qwen-bf16-encoder-20261018-03-int8','qwen-bf16-encoder-20261018-04-bf16','qwen-bf16-encoder-20261018-04-turbo',
      'qwen-cfg-er-sde-20261018-01-int8','qwen-cfg-er-sde-20261018-02-int8','qwen-cfg-er-sde-20261018-02-bf16','qwen-cfg-er-sde-20261018-02-turbo',
      'qwen-tiled-20261018-01-int8','qwen-stage2-final-20261018-portrait1600',
      'qwen-stage2-final-20261018-two-references','qwen-stage2-final-20261018-keep-vision',
      'qwen-stage2-final-20261018-t2i-int8']
records={};pending=[]
def copy(source,target):
 if source.is_file():
  target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
for name in runs:
 root=a.runtime/name;comparison=root/'comparison.json'
 if not comparison.is_file():pending.append(name);continue
 report=json.loads(comparison.read_text());target=destination/name
 for leaf in ['comparison.json','strict-comparison-status.json','rust-metadata-policy.json','python-metadata-policy.json','rust.log','python.log','comparison.log','isolation.json','executed-profile.json']:
  copy(root/leaf,target/leaf)
 for engine in ['rust','python']:
  own=a.runtime/(name+'-'+engine);own=own/'first-image'if engine=='rust'else own
  leaves=['result.json','conditioning.json','lora-text.json','lora-dit.json','memory-policy.json','memory-manager.json','component-memory.json','memory-services.json','prefix-cache.json','step-seconds.json','image.png']
  # Keep bounded raw fixtures. Large trained image tensors remain in the
  # isolated run root; comparison.json retains both complete SHA256 values,
  # finite checks and exact numerical metrics. Keep every PNG for inspection.
  leaves += [key+'-f32.bin'for key in report['metrics']if key!='png_rgba8'
             and (own/(key+'-f32.bin')).is_file()
             and (own/(key+'-f32.bin')).stat().st_size<=8*1024**2]
  for leaf in leaves:copy(own/leaf,target/engine/leaf)
 for engine in ['rust','python']:copy(a.runtime/(name+'-'+engine+'-policy.json'),target/(engine+'-metadata-policy.json'))
 records[name]=dict(numerical_exact=all(v.get('exact_bytes',v.get('exact'))and v.get('all_finite',True)for v in report['metrics'].values()),
  component_states_exact=report.get('component_memory',{}).get('all_observed_fields_exact'),
  timings_seconds=report['timings_seconds'],diagnostic_placement_override=report['diagnostic_memory_plan'])
 if records[name]['component_states_exact']is False:
  proofs=[target/(engine+'-metadata-policy.json')for engine in ['rust','python']]
  records[name]['same_input_original_memory_policy_exact']=all(path.is_file()and json.loads(path.read_text())['all_exact']for path in proofs)
logs=['qwen-precision-sampling-20261018-final-check.log','qwen-stage2-20261018-unfiltered-primitives.log','qwen-context-oracle.log','qwen-context-test.log','qwen-context-clippy.log','qwen-final-lib-check.log','qwen-final-production-build.log','qwen-final-production-linkage.json','qwen-final-edge-oracles.log','qwen-dit-precision-test.log','qwen-beta-native-build.log','qwen-beta-diagnostic.log','qwen-beta-test-final.log','qwen-beta-test-roundtrip.log','qwen-components-guarded-meta.log','qwen-inference-mode-vendor-verify.json','qwen-stage2-final-check.log','qwen-final-isolation-check.log','qwen-denoiser-clippy.log']
for leaf in logs:copy(a.runtime/leaf,destination/'checks'/leaf)
for path in a.runtime.glob('qwen_image21_original_*-isolation.json'):
 copy(path,destination/'isolation'/path.name)
source_inputs={}
for name in ['src','native','tests','tools','profiles','fixtures','resources','vendor']:
 for path in sorted((project/'inference'/name).rglob('*')):
  # Generated interpreter caches are not source inputs or portable fixtures.
  if '__pycache__'in path.parts or path.suffix in {'.pyc','.pyo'}:continue
  if path.is_file():source_inputs[str(path.relative_to(project))]=sha(path)
for name in ['Cargo.toml','Cargo.lock']:
 path=project/'inference'/name;source_inputs[str(path.relative_to(project))]=sha(path)
files={str(path.relative_to(destination)):dict(sha256=sha(path),bytes=path.stat().st_size)for path in sorted(destination.rglob('*'))if path.is_file()}
checkpoint=dict(schema=1,baseline_commit='ac8594c',status='continuation_in_progress',stage2_complete=False,
 reference_commit='ba994879b88d0ebd9950b1e5c1e923d928f216ba',production_runtime_read=False,new_parity_exceptions=[],
 runs=records,pending_runs=pending,source_inputs=source_inputs,evidence=files)
if a.finalize_selected_stage2:
 assert not pending,('Selected comparisons still pending',pending)
 assert all(r['numerical_exact']and not r['diagnostic_placement_override']for r in records.values())
 for name,record in records.items():
  if record['component_states_exact']is False:
   for engine in ['rust','python']:
    proof=json.loads((destination/name/(engine+'-metadata-policy.json')).read_text())
    assert proof['all_exact']and not proof['safety']['live_memory_override'],(name,engine)
 focused=(a.runtime/'qwen-stage2-final-check.log').read_text()
 assert '46 passed; 0 failed'in focused and 'FAILED'not in focused
 assert '191 passed; 0 failed'in(a.runtime/'qwen-stage2-20261018-unfiltered-primitives.log').read_text()
 assert '2 passed; 0 failed'in(a.runtime/'qwen-final-lib-check.log').read_text()
 builds=[json.loads(line)for line in (a.runtime/'qwen-image21-test-build.json').read_text().splitlines()]
 binary=next(Path(x['executable'])for x in builds if x.get('reason')=='compiler-artifact'and x['target']['name']=='primitives'and x.get('executable'))
 binary_sha=sha(binary)
 isolation={}
 expected_filters=set(re.findall(r'\b\w+::qwen_image21_original_\w+', (project/'inference/tools/check_qwen_image21.sh').read_text()))
 assert len(expected_filters)==46
 # Additional historical reload/edge proofs stay preserved but do not claim
 # validation against the current executable. Check each declared current case.
 for expected_filter in sorted(expected_filters):
  path=destination/'isolation'/(expected_filter.split('::')[-1]+'-isolation.json')
  result=json.loads(path.read_text())
  assert result['test_filter']==expected_filter
  assert result['result']=='passed'and not result['interpreter_available']and not result['comfyui_installation_available']
  assert result['binary_sha256']==binary_sha,('Stale isolation executable',path.name)
  isolation[result['test_filter']]=sha(path)
 assert len(isolation)==46,('Incomplete isolated suite',len(isolation))
 reference_locks={
  'reference-lock.json':'4765e81dfe84ff32ee8352ba84c68e89f578876061fdbcd439a3a7ccac7674bb',
  'source-manifest.json':'3e233cb602129a5e6bcb51aecd8e4215448551e176e8286df04fbf47ecd781c9',
  'source-acquisition-lock.json':'b1148f2b24bb2fa66f69478cc1ec7960acfe4579f4aae18c63e88815ba220e77',
  'native-linkage-lock.json':'37737a4525ec3db12f6314d2dd0e69a8fb4c5374d8ad04797ade0447e110bfa6'}
 for name,digest in reference_locks.items():
  assert sha(project/'design/reference-local'/name)==digest,('Frozen reference changed',name)
 checkpoint.update(status='selected_vertical_path_complete_partial_overall',stage2_complete=True,
  complete_comfyui_parity=False,completion_scope='Master design section 15: selected Qwen Image 2.1 vertical paths only; general contracts, complete reference/model matrix and product release remain later-stage requirements.',
  focused_tests_passed=46,python_free_individual_tests_passed=46,
  current_test_binary_sha256=binary_sha,isolation_records=isolation,
  report='stage-2-qwen-image21-ready.md',reference_locks=reference_locks,
  pre_context_unfiltered_primitive_tests_passed=191,library_unit_tests_passed=2)
(project/'design/stage-2-qwen-image21-precision-sampling-checkpoint.json').write_text(json.dumps(checkpoint,indent=2)+'\n')
print(f'Preserved {len(files)} evidence files, {len(records)} comparisons and {len(source_inputs)} implementation/fixture hashes')
