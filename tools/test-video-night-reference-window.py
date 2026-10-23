#!/usr/bin/env python3
"""Actual Director reference-tail attention CPU oracle, no GPU/Exp/observer."""
import ast,dataclasses,json,math,os,subprocess,sys,types
from pathlib import Path
from typing import Any,Callable
import torch
ROOT=Path(__file__).resolve().parents[1];OWN=Path('/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e')
def main():
 out=Path(sys.argv[1]);assert out.parent==OWN and out.resolve()==out
 mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(OWN)]))['filesystems'][0];assert mount['source']=='/dev/mapper/veracrypt1'and not os.statvfs(OWN).f_flag&os.ST_RDONLY
 out.mkdir();torch.set_num_threads(1);assert not torch.cuda.is_initialized()
 class Nested:
  is_nested=True
  def __init__(self,tensors):self.tensors=tuple(tensors)
 comfy=types.ModuleType('comfy');nested=types.ModuleType('comfy.nested_tensor');nested.NestedTensor=Nested;sys.modules['comfy']=comfy;sys.modules['comfy.nested_tensor']=nested
 module=types.ModuleType('imagine_reference_tail_oracle');sys.modules[module.__name__]=module;scope=module.__dict__;scope.update(torch=torch,Any=Any,Callable=Callable,dataclass=dataclasses.dataclass,H3_VIDEO_FRAMES_PER_LATENT_STEP=(1,4,4,4,4))
 source=Path('/home/thhel/git/comfyui-helto-director/shared/h3/runtime/attention_window.py')
 names={'_AttentionWindowProfile','_build_reference_tail_attention_profile','_conditioning_entry','_packed_keyframe_rows','_h3_av_streams','_reference_frame_rows','_reference_audio_rows','_positive_int','_video_step_attention_weights','_make_windowed_attention_override','_complement_ranges'}
 nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in names];assert len(nodes)==len(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),scope)
 inputs=[];expected=[]
 for w,h,frames,vt,at in [(32,32,124,37,207),(64,32,39,12,65),(64,64,22,7,37)]:
  # Audio latent length follows the actual shape's sampling formula (set below).
  at=round(frames*(5./3.))
  for paired in [False,True]:
   for guide in [False,True]:
    refs=[{'kind':'image','latent_h':2,'latent_w':4},{'kind':'audio','ref_audio_t':3},{'kind':'video_audio'if paired else'video','latent_t':7,'latent_h':2,'latent_w':4,'ref_audio_t':4 if paired else 0},{'kind':'video_audio','latent_t':2,'latent_h':4,'latent_w':2,'ref_audio_t':3}]
    keyframes=[{'latent':torch.zeros((1,24,1,h//16,w//16))}]if guide else[]
    conditioning=[[torch.zeros((1,3,2)),{'minimax_refs':refs,'minimax_keyframes':keyframes}]]
    latent={'samples':Nested((torch.zeros((1,24,vt,h//16,w//16)),torch.zeros((1,32,2,at))))}
    profile,debug=scope['_build_reference_tail_attention_profile'](conditioning=conditioning,latent=latent,boundary_video_reference_index=0,full_visible_frames=16,fade_visible_frames=16)
    n=profile.minimum_sequence_length;q=torch.arange(n*2,dtype=torch.float32).reshape(1,1,n,2)/1000.;k=q.flip((2,));v=q.sin()
    def backend(q,k,v,heads,**kwargs):return ((q@k.transpose(-2,-1)/2.**.5).softmax(-1)@v).transpose(1,2).contiguous().reshape(1,-1,2)
    run=scope['_make_windowed_attention_override'](profile,backend_resolver=lambda *a,**kw:backend)
    expected.append(run(q,k,v,1,skip_reshape=True).flatten())
    inputs.append({'shape':{'width':w,'height':h,'frames':frames,'fps':24},'refs':refs,'text':3,'keyframe_rows':(w//32)*(h//32)if guide else 0,'n':n,'paired':paired,'guide':guide})
 (out/'input.json').write_text(json.dumps(inputs))
 native=ROOT/'comfyui/h3-video-night-20261021/target-native/debug/examples/h3_reference_window_cpu'
 result=subprocess.run([str(native)],input=json.dumps(inputs),capture_output=True,text=True);(out/'probe-stderr.txt').write_text(result.stderr);assert result.returncode==0,result.stderr
 actual=json.loads(result.stdout);(out/'rust-output.json').write_text(result.stdout);rows=[]
 for source,a,b in zip(inputs,actual,expected):
  d=torch.tensor(a)-b;rows.append({'shape':source['shape'],'paired':source['paired'],'guide':source['guide'],'mae':d.abs().mean().item(),'max':d.abs().max().item(),'byteidentical':torch.equal(torch.tensor(a),b)})
 report={'passed':len(rows)==12 and all(r['max']<=1e-6 for r in rows),'cases':rows,'cpu_only':True,'runtime_binding':False,'scope':'Director actual reference-tail profile and windowed attention; first video/paired audio limited 16+16, pictures/authored video unaffected. No Exp diagnostics/model/GPU/observer.'};(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));return not report['passed']
if __name__=='__main__':raise SystemExit(main())
