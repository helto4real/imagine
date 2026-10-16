"""Build timer-only diagnostic objects; never edit or replace the candidate."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / '.imagine/lora-merge'
SOURCE = BASE / 'source'
BUILD = BASE / 'build'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--loader-waits',action='store_true')
args=parser.parse_args()
OUT = BASE / ('activation-waits-diagnostic' if args.loader_waits else 'activation-diagnostic')
OUT.mkdir(exist_ok=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
manifest = json.loads((BASE / 'manifest.json').read_text())
assert sha(ROOT / manifest['binary_path']) == manifest['binary_sha256']
assert sha(ROOT / 'native/lora-merge/source-lock.json') == manifest['source_lock_sha256']
header = OUT / 'phase-timer.hpp'
header.write_text('''#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fcntl.h>
#include <unistd.h>
static unsigned long long phase_clock() {
    timespec t{}; clock_gettime(CLOCK_MONOTONIC, &t);
    return (unsigned long long)t.tv_sec * 1000000000ULL + t.tv_nsec;
}
struct ActivationPhaseTimer {
    const char* label; unsigned long long start;
    explicit ActivationPhaseTimer(const char* value): label(value), start(phase_clock()) {}
    ~ActivationPhaseTimer() {
        const auto end = phase_clock();
        static int fd = []() {
            const char* path = std::getenv("IMAGINE_LORA_PHASE_TRACE");
            return path ? open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600) : -1;
        }();
        if (fd < 0) return;
        char record[160];
        const int size = std::snprintf(record, sizeof(record), "%s\\t%llu\\t%llu\\n", label, start, end);
        if (size > 0 && size < int(sizeof(record))) (void)write(fd, record, size);
    }
};
#define ACT_TIME(label, expression) ([&]() -> decltype(auto) { ActivationPhaseTimer timer(label); return (expression); }())
''')

def wrap(s, expression, label, count=1):
    assert s.count(expression) == count, (label, s.count(expression))
    return s.replace(expression, f'ACT_TIME("{label}", {expression})')

def scope(s, signature, label):
    assert s.count(signature) == 1, label
    return s.replace(signature, signature + f'\n    ActivationPhaseTimer activation_timer("{label}");')

texts = {}
p = 'src/model_manager.cpp';s = (SOURCE/p).read_text()
s = scope(s,'bool ModelManager::apply_merged_int8_loras(const std::vector<TensorState*>& states) {','merge.total')
s = scope(s,'bool ModelManager::load_tensors(const std::vector<TensorState*>& states) {','weights.load')
s = scope(s,'bool ModelManager::prepare_lora_sources(std::vector<LoraSpec>& loras) {','lora.catalog')
a=s.index('bool ModelManager::apply_merged_int8_loras(');prefix=s[:a];s=s[a:]
for expression,label in [
 ('adapter->init_params(n_threads_, filter)','merge.adapter_init'),
 ('adapter->preprocess_lora_tensors(names)','merge.adapter_mapping'),
 ('adapter->load_merge_operand(up->second, item.up)','merge.operand_up'),
 ('adapter->load_merge_operand(down->second, item.down)','merge.operand_down'),
 ('ensure_compute_backend_capacity(request, protected_tensors, {}, protected_tensors)','merge.capacity'),
 ('ggml_backend_synchronize(backend)','merge.backend_sync'),
 ('model_loader_.load_tensors(load_scale, enable_mmap_, &selected, false)','merge.original_scales'),
 ('sd_lora_int8_merge_cuda(width, rows, group, crc ^ 0xffffffffu, int(patches.size()), patches.data(), static_cast<int8_t*>(state->tensor->data), scales.data(), device_quantized)','merge.cuda_total'),
 ('ggml_backend_tensor_set(scale, scales.data(), 0, scales.size() * sizeof(float))','merge.scale_store')]:
 s=wrap(s,expression,label)
texts[p]=prefix+s
p='src/pipeline/diffusion_engine.cpp';s=(SOURCE/p).read_text()
s=scope(s,'bool StableDiffusionGGML::apply_loras(const sd_lora_t* loras, uint32_t lora_count) {','engine.apply_loras')
s=scope(s,'SDCondition StableDiffusionGGML::get_learned_condition(const ConditionerParams& params) {','engine.conditioning')
texts[p]=s
p='src/model_loader.cpp';s=(SOURCE/p).read_text()
s=scope(s,'''bool ModelLoader::load_tensors(on_new_tensor_cb_t on_new_tensor_cb,
                               bool enable_mmap,
                               const std::set<std::string>* target_tensor_names,
                               bool log_progress) {''','loader.total')
a=s.index('bool ModelLoader::load_tensors(on_new_tensor_cb_t');b=s.index('\nbool ModelLoader::',a+5)
part=s[a:b]
part=wrap(part,'validate_sources(target_tensor_names)','loader.validate',2)
part=wrap(part,'process_model_files(enable_mmap, false)','loader.process_files')
if args.loader_waits:
    part=wrap(part,'std::this_thread::sleep_for(std::chrono::milliseconds(total_tensors_to_process <= 4 ? 10 : 200))','loader.poll_sleep')
    part=wrap(part,'w.join()','loader.join')
    marker='workers.emplace_back([&, file_path, is_zip]() {'
    assert part.count(marker)==1
    part=part.replace(marker,marker+'\n                ActivationPhaseTimer worker_timer("loader.worker");')
s=s[:a]+part+s[b:];texts[p]=s
p='src/model_loader_files.cpp';s=(SOURCE/p).read_text()
s=scope(s,'ModelLoader ModelLoader::file_reader(FileId id, SDVersion version) const {','loader.file_reader')
texts[p]=s
commands=json.loads((BUILD/'compile_commands.json').read_text())
provenance={}
objects=[]
jobs=[]
for rel,s in texts.items():
 original=SOURCE/rel;copy=OUT/Path(rel).name;obj=copy.with_suffix('.o')
 copy.write_text('#include "phase-timer.hpp"\n'+s)
 provenance[rel]={'original_sha256':sha(original),'diagnostic_sha256':sha(copy)}
 c=next(x for x in commands if x['file']==str(original))
 cmd=shlex.split(c['command']);cmd[cmd.index('-o')+1]=str(obj);cmd[cmd.index('-c')+1]=str(copy)
 cmd.insert(1,'-I'+str(original.parent));jobs.append(cmd);objects.append(str(obj))
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 for result in pool.map(lambda cmd:subprocess.run(cmd,cwd=BUILD,check=True),jobs):pass
link=subprocess.check_output(['ninja','-t','commands','bin/sd-server'],cwd=BUILD,text=True).splitlines()[-1]
assert link.startswith(': && ') and link.endswith(' && :')
cmd=[x for x in shlex.split(link[5:-5]) if not x.startswith('-Wl,--dependency-file=')]
cmd[1:1]=objects
binary=OUT/'sd-server';cmd[cmd.index('-o')+1]=str(binary)
subprocess.run(cmd,cwd=BUILD,check=True)
assert sha(ROOT/manifest['binary_path'])==manifest['binary_sha256']
result={**manifest,'binary_path':str(binary.relative_to(ROOT)),'binary_sha256':sha(binary),
 'parent_binary_sha256':manifest['binary_sha256'],'diagnostic_only':True,
 'timer_header_sha256':sha(header),'diagnostic_sources':provenance}
(OUT/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
print('Built isolated timing-only diagnostic; production candidate unchanged.')
