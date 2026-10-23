#!/usr/bin/env python3
"""Focused actual-source Reference RGB/audio preprocessing parity, CPU only, own fixtures."""
import ast,json,math,os,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
OWN=Path('/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e')
def functions(path,names,scope,remove_import=False):
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    if remove_import:
        for n in nodes:n.body=[v for v in n.body if not isinstance(v,ast.ImportFrom)]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
def main():
    case=Path(sys.argv[1]);assert case.parent==OWN and case.resolve()==case
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(OWN)]))['filesystems'][0]
    assert mount['source']=='/dev/mapper/veracrypt1' and not os.statvfs(OWN).f_flag&os.ST_RDONLY
    case.mkdir();torch.set_num_threads(2)
    scope={'torch':torch,'np':np,'math':math,'Image':Image,'F':F}
    functions(Path('/home/thhel/git/ComfyUI/comfy/utils.py'),{'lanczos','common_upscale'},scope)
    functions(Path('/home/thhel/git/ComfyUI/comfy/audio.py'),{'resample'},scope)
    config=ast.parse(Path('/home/thhel/git/comfyui-helto-director/shared/h3/config.py').read_text())
    scope['H3_REFERENCE_QUALITY_PIXELS']=next(ast.literal_eval(n.value)for n in config.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='H3_REFERENCE_QUALITY_PIXELS'for t in n.targets))
    functions(Path('/home/thhel/git/comfyui-helto-director/shared/h3/reference_quality.py'),{'reference_image_geometry','_resize','prepare_reference_image'},scope,True)
    runtime=json.loads(Path('/run/media/veracrypt1/imagine/.codex-h3-res20-compare-20261020-6f0b84c2/full-fp16-20261020-5e4a923b/candidate-runtime.json').read_text())
    pillow=runtime['h3']['environment']['IMAGINE_PILLOW_LIBRARY']
    probe=ROOT/'comfyui/h3-video-night-20261021/target-native/debug/examples/h3_reference_cpu'
    rows=[]
    for ordinal,(w,h,quality) in enumerate([(64,64,'Standard'),(416,224,'Standard'),(321,177,'Standard'),(1281,1025,'Standard'),(1537,1025,'High')]):
        folder=case/f'image-{ordinal}';folder.mkdir()
        rgb=np.fromfunction(lambda y,x,c:(x*13+y*7+c*61)%256,(h,w,3),dtype=int).astype(np.uint8);(folder/'input.rgb').write_bytes(rgb.tobytes())
        expected,geometry=scope['prepare_reference_image'](torch.from_numpy(rgb.astype(np.float32)/255.).unsqueeze(0),quality)
        (folder/'python.f32').write_bytes(expected.numpy().tobytes())
        data={'shape':{'width':416,'height':224,'frames':124,'fps':24},'references':{'quality':quality,'images':[{'width':w,'height':h}],'audios':[]},'image_files':[str(folder/'input.rgb')],'audio_files':[],'image_outputs':[str(folder/'rust.f32')],'audio_outputs':[],'pillow':pillow}
        (folder/'input.json').write_text(json.dumps(data));run=subprocess.run([str(probe),str(folder/'input.json')],capture_output=True,text=True)
        (folder/'stderr.txt').write_text(run.stderr);assert run.returncode==0,run.stderr
        actual=np.fromfile(folder/'rust.f32',np.float32).reshape(expected.shape);diff=actual.astype(np.float64)-expected.numpy().astype(np.float64)
        rows.append({'kind':'image','source':[w,h],'quality':quality,'shape':list(actual.shape),'geometry':geometry,'byteidentical':actual.tobytes()==expected.numpy().tobytes(),'mae':float(np.abs(diff).mean()),'rmse':float(np.square(diff).mean()**.5),'max':float(np.abs(diff).max())})
    for ordinal,length in enumerate([503,4410,16317,44100]):
        folder=case/f'audio-{ordinal}';folder.mkdir()
        t=np.arange(length,dtype=np.float32)/44100.;pcm=np.stack([.2*np.sin(2*np.pi*233*t),.17*np.cos(2*np.pi*377*t)],axis=-1).astype(np.float32);(folder/'input.pcm').write_bytes(pcm.tobytes())
        expected=scope['resample'](torch.from_numpy(pcm.T.copy()).unsqueeze(0),44100,32000);(folder/'python.f32').write_bytes(expected.numpy().tobytes())
        data={'shape':{'width':416,'height':224,'frames':124,'fps':24},'references':{'images':[],'audios':[{'sample_rate':44100,'samples':length}]},'image_files':[],'audio_files':[str(folder/'input.pcm')],'image_outputs':[],'audio_outputs':[str(folder/'rust.f32')],'pillow':pillow}
        (folder/'input.json').write_text(json.dumps(data));run=subprocess.run([str(probe),str(folder/'input.json')],capture_output=True,text=True)
        (folder/'stderr.txt').write_text(run.stderr);assert run.returncode==0,run.stderr
        actual=np.fromfile(folder/'rust.f32',np.float32).reshape(expected.shape);diff=actual.astype(np.float64)-expected.numpy().astype(np.float64)
        rows.append({'kind':'audio','samples_in':length,'shape':list(actual.shape),'byteidentical':actual.tobytes()==expected.numpy().tobytes(),'mae':float(np.abs(diff).mean()),'rmse':float(np.square(diff).mean()**.5),'max':float(np.abs(diff).max())})
    report={'passed':all(r['byteidentical']for r in rows),'rows':rows,'cpu_only':True,'scope':'Actual Director quality geometry/replicate padding + actual Comfy Lanczos and default sinc_interp_hann. No text permutations or model inference.'}
    (case/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));raise SystemExit(not report['passed'])
if __name__=='__main__':main()
