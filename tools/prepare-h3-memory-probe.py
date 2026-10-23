#!/usr/bin/env python3
"""Generate bounded developer probes; never modify the frozen worker or activate a product.

The original Stage 4 phase runner is retained byte-for-byte. These explicit
derivatives add two geometries, budget selection and metrics to the same phases.
Only newly generated synthetic test artifacts may be supplied to these probes.
"""
import argparse
import hashlib
import json
from pathlib import Path


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def prepare(source):
    path = source / 'inference/src/bin/h3-stage4.rs'
    native = path.read_text()
    native = replace_once(native, 'fn sha(path: &Path)', '''use imagine_inference::h3_product::protocol;
#[path = "../h3_product/memory_budget.rs"]
mod memory_budget;
fn geometry() -> protocol::Shape {
    match std::env::var("H3_MEMORY_GEOMETRY").unwrap().as_str() {
        "short" => protocol::Shape::default(),
        "large" => protocol::Shape { width: 768, height: 512, frames: 124, fps: 24 },
        _ => panic!("Unregistered synthetic geometry"),
    }
}
fn candidate() -> bool {
    match std::env::var("H3_MEMORY_VARIANT").unwrap().as_str() {
        "candidate" => true, "control" => false,
        _ => panic!("Unregistered budget variant"),
    }
}
fn capture() -> bool { std::env::var("H3_MEMORY_CAPTURE").as_deref() == Ok("1") }
fn monotonic_ns() -> u64 {
    let mut t = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    assert_eq!(unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut t) }, 0);
    t.tv_sec as u64 * 1_000_000_000 + t.tv_nsec as u64
}
fn host_metrics() -> Value {
    let status = fs::read_to_string("/proc/self/status").unwrap();
    let mut out = serde_json::Map::new();
    for line in status.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        if matches!(fields[0], "VmRSS:" | "VmHWM:" | "VmLck:" | "VmPin:" | "RssAnon:" | "RssFile:") {
            out.insert(fields[0].trim_end_matches(':').into(), json!(fields[1].parse::<u64>().unwrap() * 1024));
        }
    }
    Value::Object(out)
}
fn mark(output: &Path, name: &str, wait_ns: u64) -> std::result::Result<(), Error> {
    use std::io::Write;
    let row = json!({"name":name,"ns":monotonic_ns(),"sync_wait_ns":wait_ns,"host":host_metrics()});
    let mut f=fs::OpenOptions::new().create(true).append(true).open(output.join("metrics.jsonl"))
        .map_err(|e| Error::Native(e.to_string()))?;
    writeln!(f,"{row}").map_err(|e| Error::Native(e.to_string()))
}
fn sha(path: &Path)''')
    native = replace_once(native, '    let t = t.f_to_device(Device::Cpu)?.f_contiguous()?;', '''    let t = t.f_to_device(Device::Cpu)?.f_contiguous()?;''')
    native = replace_once(native, '    fs::write(root.join(format!("{name}.bin")), &data)?;', '''    if capture() || matches!(name, "context" | "tags" | "latent" | "video-latent" | "audio-latent") {
        fs::write(root.join(format!("{name}.bin")), &data)?;
    }''')
    native = replace_once(native, '        if name.starts_with("h3.sampling.") {', '''        if name.starts_with("h3.sampling.") {
            if name.ends_with(".x") {
                let start=monotonic_ns();
                if std::env::var_os("QWEN_ACTIVITY_TRACE").is_some() { self.native.synchronize_device(0)?; }
                mark(&self.output,name,monotonic_ns()-start)?;
            }
            if !capture() { return Ok(()); }''')
    native = replace_once(native, '    json!({"model_size":owner.model_size()', '''    let resident: u64=owner.modules.iter().filter(|m|owner.placement.resident.contains(&m.name)).map(|m|m.module_bytes).sum();
    json!({"resident_module_bytes":resident,"host":host_metrics(),"model_size":owner.model_size()''')
    start = native.index('            let requested = match name {')
    end = native.index('            manager.load(', start)
    native = native[:start] + '''            let shape=geometry();
            let requested = match name {
                "video" => (9.5 * 30. * shape.height as f64 * shape.width as f64 + 270_000_000.) * 2. * 1.03 * (shape.frames as f64 / 39.).max(1.),
                "reference" => (7. * 256. * 384. + 110_000_000.) * 2. * 1.03,
                "audio" => 42_000_000. * 4. * 1.03 * (shape.audio_samples() as f64 / 52_000.).max(1.),
                "dit" => memory_budget::dit(shape).0,
                _ => 0.,
            };
            let minimum = (name == "dit" && candidate()).then(||memory_budget::dit(shape).1);
            report["budget"] = json!({"requested":requested,"minimum":minimum,"geometry":shape});
''' + native[end:]
    native = replace_once(native, '                requested,\n                None,', '                requested,\n                minimum,')
    native = replace_once(native, '            let inference = Instant::now();', '''            if std::env::var_os("H3_MEMORY_OOM").is_some() {
                let _ = Tensor::f_empty([64_i64 * 1024 * 1024 * 1024], (Kind::Uint8, dev))?;
                return Err("Expected controlled allocation failure did not occur".into());
            }
            let inference = Instant::now();''')
    native = replace_once(native, 'Tensor::f_zeros([1, 32, 12, 16, 24], (Kind::Float, dev))?', 'Tensor::f_zeros([1, 32, shape.video_latent_t(), (shape.height / 16) as i64, (shape.width / 16) as i64], (Kind::Float, dev))?')
    native = replace_once(native, '                        let result = unsafe {\n                            h3::sampling::sample_euler_cfg1(', '''                        rt.native.synchronize_device(0)?;
                        mark(&output,"sampling.begin",0)?;
                        let sampling_start=Instant::now();
                        let result = unsafe {
                            h3::sampling::sample_euler_cfg1(''')
    native = replace_once(native, '                        for (i, value) in result.unbind().borrow().iter().enumerate() {', '''                        rt.native.synchronize_device(0)?;
                        report["sampling_seconds"]=json!(sampling_start.elapsed().as_secs_f64());
                        mark(&output,"sampling.end",0)?;
                        for (i, value) in result.unbind().borrow().iter().enumerate() {''')
    native = native.replace('[1, 3, 39, 256, 384]', '[1, 3, shape.frames as i64, shape.height as i64, shape.width as i64]')
    native = native.replace('[1, 2, 52000]', '[1, 2, shape.native_audio_samples() as i64]')
    native = replace_once(native, '    report["cuda_after_release"]', '    report["host_after_release"] = host_metrics();\n    report["cuda_after_release"]')
    native = replace_once(native, '    let mut args = std::env::args_os().skip(1);', '''    let _ = (geometry(), candidate());
    let mut args = std::env::args_os().skip(1);''')
    target = source / 'inference/src/bin/h3-memory-probe.rs'
    target.write_text(native)

    original_path = source / 'inference/tools/reference_h3_trained.py'
    original = original_path.read_text()
    original = replace_once(original, 'import os,time,gc\n', '''import os,time,gc
# The profiler is already loaded in this process. Compiler subprocesses must
# not preload it and overwrite the parent's trace. Keep the phase sync flag.
if os.environ.get('QWEN_ACTIVITY_TRACE'):os.environ.pop('LD_PRELOAD',None)
''')
    original = replace_once(original, "output=a.own_root/phase;", '''variant=os.environ['H3_MEMORY_VARIANT'];assert variant in ('control','candidate')
geometry=os.environ['H3_MEMORY_GEOMETRY'];assert geometry in ('short','large')
width,height,num_frames,vt,at=(384,256,39,12,65)if geometry=='short'else(768,512,124,37,207)
capture=os.environ.get('H3_MEMORY_CAPTURE')=='1'
def host_metrics():
 out={}
 for line in Path('/proc/self/status').read_text().splitlines():
  p=line.split()
  if p[0]in ('VmRSS:','VmHWM:','VmLck:','VmPin:','RssAnon:','RssFile:'):out[p[0][:-1]]=int(p[1])*1024
 return out
def mark(name,wait=0):
 with(output/'metrics.jsonl').open('a')as f:f.write(json.dumps(dict(name=name,ns=time.monotonic_ns(),sync_wait_ns=wait,host=host_metrics()))+'\\n')
output=a.own_root/phase;''')
    original = replace_once(original, "(output/(name+'.bin')).write_bytes(data);", "\n if capture or name in ('context','tags','latent','video-latent','audio-latent'):(output/(name+'.bin')).write_bytes(data)\n ")
    original = replace_once(original, "return dict(model_size=patcher.model_size()", "return dict(host=host_metrics(),resident_module_bytes=sum(comfy.model_management.module_size(m)for _,_,_,m,_ in patcher._load_list()if all(v.device.type=='cuda'for v in m.parameters(recurse=False))),model_size=patcher.model_size()")
    original = original.replace('24*12*16*24+32*2*65', '24*vt*(height//16)*(width//16)+32*2*at')
    original = original.replace("'video':(9.5*30*256*384+270_000_000)*2*1.03", "'video':(9.5*30*height*width+270_000_000)*2*1.03*max(1,num_frames/39)")
    original = original.replace("'audio':42_000_000*4*1.03", "'audio':42_000_000*4*1.03*max(1,at*800/52000)")
    original = replace_once(original, 'comfy.model_management.load_models_gpu([patcher],memory_required=requested)', '''minimum=None
  if role=='dit'and variant=='candidate':
   requested,minimum=comfy.sampler_helpers.estimate_memory(patcher,[1,1,24*vt*(height//16)*(width//16)+32*2*at],{})
  report['budget']=dict(requested=requested,minimum=minimum)
  comfy.model_management.load_models_gpu([patcher],memory_required=requested,minimum_memory_required=minimum)''')
    original = original.replace('(1,32,12,16,24)', '(1,32,vt,height//16,width//16)')
    original = original.replace('[1,3,39,256,384]', '[1,3,num_frames,height,width]').replace('[1,2,52000]', '[1,2,at*800]')
    original = replace_once(original, '   def callback(i,denoised,x,total):', '''   def callback(i,denoised,x,total):
    begin=time.monotonic_ns()
    if os.environ.get('QWEN_ACTIVITY_TRACE'):torch.cuda.synchronize()
    mark(f'h3.sampling.{i}.x',time.monotonic_ns()-begin)
    if not capture:return''')
    original = replace_once(original, '   result=guider.inner_sample(', "   torch.cuda.synchronize();mark('sampling.begin');sampling_start=time.monotonic()\n   result=guider.inner_sample(")
    original = replace_once(original, '   video,audio=comfy.utils.unpack_latents(result,shapes);', "   torch.cuda.synchronize();report['sampling_seconds']=time.monotonic()-sampling_start;mark('sampling.end')\n   video,audio=comfy.utils.unpack_latents(result,shapes);")
    original = replace_once(original, "report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()", "report['peak_allocated_bytes']=torch.cuda.max_memory_allocated();report['peak_reserved_bytes']=torch.cuda.max_memory_reserved()")
    original = replace_once(original, "report['seconds']=time.monotonic()-begin", "report['host_after_release']=host_metrics();report['seconds']=time.monotonic()-begin")
    oracle = source / 'inference/tools/reference_h3_memory.py'
    compile(original,str(oracle),'exec')
    oracle.write_text(original)
    return {str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [path,original_path,target,oracle]}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]/'comfyui/h3-memory-20261023'
    assert args.source.resolve()==root/'source'
    (root/'probe-sources.json').write_text(json.dumps(prepare(args.source),indent=2)+'\n')
