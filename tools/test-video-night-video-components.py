#!/usr/bin/env python3
"""Actual-source video decode/selection/canvas/Qwen/VAE-input CPU comparison."""
import ast,json,math,os,runpy,subprocess,sys,types
from pathlib import Path
from typing import Any,NamedTuple
import av,numpy as np,torch
from PIL import Image
import torch.nn.functional as F
C=runpy.run_path(str(Path(__file__).with_name('test-video-night-reference-components.py')))
OWN=C['OWN'];ROOT=C['ROOT'];functions=C['functions']
DIRECTOR=Path('/home/thhel/git/comfyui-helto-director/shared')

def compare(a,b):
    a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape,(a.shape,b.shape)
    d=a.astype(np.float64)-b.astype(np.float64)
    return {'byteidentical':a.tobytes()==b.tobytes(),'shape':list(a.shape),'mae':float(np.abs(d).mean()),'rmse':float(np.square(d).mean()**.5),'max':float(np.abs(d).max())}

def main():
    case=Path(sys.argv[1]);assert case.parent==OWN and case.resolve()==case
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(OWN)]))['filesystems'][0];assert mount['source']=='/dev/mapper/veracrypt1'and not os.statvfs(OWN).f_flag&os.ST_RDONLY
    case.mkdir();torch.set_num_threads(2)
    scope={'torch':torch,'np':np,'math':math,'Image':Image,'F':F,'av':av,'Any':Any,'NamedTuple':NamedTuple,'resolve_media_path':lambda p:Path(p),'TARGET_SAMPLE_RATE':44100}
    media=DIRECTOR/'media.py'
    classes=[n for n in ast.parse(media.read_text()).body if isinstance(n,ast.ClassDef)and n.name in ['DecodedVideo','SelectedVideoFrames']]
    exec(compile(ast.Module(body=classes,type_ignores=[]),str(media),'exec'),scope)
    functions(media,{'decode_video_frames','trim_video_source_frames','_stream_fps','_safe_float'},scope)
    functions(DIRECTOR/'audio.py',{'decode_audio_file','_resampled_tensors','trim_audio'},scope)
    functions(DIRECTOR/'h3/runtime/media.py',{'resample_video_frames','select_h3_boundary_tail_frames'},scope)
    functions(Path('/home/thhel/git/ComfyUI/comfy/utils.py'),{'lanczos','common_upscale'},scope)
    node=Path('/home/thhel/git/ComfyUI/comfy_extras/nodes_minimax_h3.py')
    constants=[n for n in ast.parse(node.read_text()).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id in ['CANVAS_MULTIPLE','BASE_SHORT_EDGE','MAX_PIXELS','FPS']for t in n.targets)]
    exec(compile(ast.Module(body=constants,type_ignores=[]),str(node),'exec'),scope)
    scope['comfy']=types.SimpleNamespace(utils=types.SimpleNamespace(common_upscale=scope['common_upscale']))
    functions(node,{'adapt_canvas','_resize'},scope)
    runtime=json.loads((OWN/'reference-e2e-06/standard/runtime.json').read_text());pillow=runtime['h3']['environment']['IMAGINE_PILLOW_LIBRARY']
    root_probe=ROOT/'comfyui/h3-video-night-20261021/target/debug/examples/reference_video_probe'
    native_probe=ROOT/'comfyui/h3-video-night-20261021/target-native/debug/examples/h3_reference_cpu'
    rows=[]
    source=OWN/'reference-e2e-06/output/2026-10-21/video_0001.mp4'
    fixture=case/'source-30fps.mp4'
    subprocess.run(['ffmpeg','-v','error','-nostdin','-threads','1','-filter_threads','1','-i',str(source),'-t','1.333333','-vf','scale=160:96,fps=30','-c:v','libx264','-pix_fmt','yuv420p','-crf','19','-c:a','aac','-ar','44100','-ac','2',str(fixture)],check=True)
    for i,path in enumerate([source,fixture]):
        folder=case/f'decode-{i}';folder.mkdir();entry={'source_in':.1,'source_out':1.1}
        data={'source':str(path),'entry':entry,'with_audio':True,'rgb':str(folder/'rust.rgb'),'pcm':str(folder/'rust.pcm')};(folder/'input.json').write_text(json.dumps(data))
        run=subprocess.run([str(root_probe),str(folder/'input.json')],capture_output=True,text=True);(folder/'stderr.txt').write_text(run.stderr);assert run.returncode==0,run.stderr
        descriptor=json.loads(run.stdout);decoded,fps,_=scope['decode_video_frames'](str(path));frames,metadata=scope['trim_video_source_frames'](decoded,fps,entry)
        rgb=(frames*255.).to(torch.uint8).numpy();audio=scope['decode_audio_file'](str(path));audio=scope['trim_audio'](audio,.1,1.1,1.).T.contiguous().numpy()
        rows.append({'case':f'decode-{i}','rgb':compare(np.fromfile(folder/'rust.rgb',np.uint8).reshape(rgb.shape),rgb),'pcm':compare(np.fromfile(folder/'rust.pcm',np.float32).reshape(audio.shape),audio),'fps_equal':descriptor['source_fps']==fps,'source_range':metadata})
    specs=[(160,96,40,30.,None),(161,97,28,25.,None),(320,192,36,24.,22),(1280,720,7,24.,None),(72,48,3,2.,None),(224,128,40,30000/1001,None)]
    for i,(w,h,n,fps,tail)in enumerate(specs):
        folder=case/f'prepare-{i}';folder.mkdir();rgb=np.fromfunction(lambda t,y,x,c:(t*11+y*7+x*13+c*61)%256,(n,h,w,3),dtype=int).astype(np.uint8);(folder/'input.rgb').write_bytes(rgb.tobytes())
        data={'shape':{'width':416,'height':224,'frames':124,'fps':24},'references':{'videos':[{'width':w,'height':h,'frames':n,'source_fps':fps,'tail_frames':tail,'audio':None}]},'image_files':[],'audio_files':[],'image_outputs':[],'audio_outputs':[],'video_files':[str(folder/'input.rgb')],'video_outputs':[str(folder/'rust.f32')],'video_qwen_outputs':[str(folder/'qwen.f32')],'video_vae_outputs':[str(folder/'vae.f32')],'pillow':pillow};(folder/'input.json').write_text(json.dumps(data))
        run=subprocess.run([str(native_probe),str(folder/'input.json')],capture_output=True,text=True);(folder/'stderr.txt').write_text(run.stderr);assert run.returncode==0,run.stderr
        actual=json.loads(run.stdout)['videos'][0]
        frames=scope['resample_video_frames'](torch.from_numpy(rgb).to(torch.float32)/255.,fps,24.)
        if tail:frames,_=scope['select_h3_boundary_tail_frames'](frames,requested_tail_frames=tail,target_frame_count=124)
        cw,ch=scope['adapt_canvas'](w,h)
        if w*h<cw*ch:cw=max(32,round(w/32)*32);ch=max(32,round(h/32)*32)
        frames=scope['_resize'](frames,cw,ch,'disabled')[:124];count=len(frames)
        assert count>=5
        while count%17!=5:count-=1
        frames=frames[:count];qwen=frames[list(range(0,count,12))]
        # VAE.encode's actual NHWC->BCTHW order, then process_input and chosen dtype.
        vae=(frames.movedim(-1,1).movedim(1,0).unsqueeze(0)*2.-1.).to(torch.float16)
        row={'case':f'prepare-{i}','frames':compare(np.fromfile(folder/'rust.f32',np.float32).reshape(frames.shape),frames.numpy()),'qwen':compare(np.fromfile(folder/'qwen.f32',np.float32).reshape(qwen.shape),qwen.numpy()),'vae':compare(np.fromfile(folder/'vae.f32',np.float32).reshape(vae.shape),vae.float().numpy()),'vae_stride_equal':actual['vae_stride']==list(vae.stride()),'vae_dtype':actual['vae_dtype'],'vae_dtype_equal':actual['vae_dtype']=='Half','timestamps_equal':actual['timestamps']==[j/2. for j in range(len(qwen))]};rows.append(row)
    report={'passed':all(all(v['byteidentical']if isinstance(v,dict)and'byteidentical'in v else v for k,v in row.items()if k in ['rgb','pcm','frames','qwen','vae','fps_equal','vae_stride_equal','timestamps_equal','vae_dtype_equal'])for row in rows),'cases':rows,'cpu_only':True,'scope':'Actual Director PyAV decode/trim/resample/tail, actual Comfy canvas/Lanczos and exact VAE encode layout/dtype boundary; no GPU/model inference.'}
    (case/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));return not report['passed']

if __name__=='__main__':raise SystemExit(main())
