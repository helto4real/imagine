#!/usr/bin/env python3
"""Explicit own UI→API→reserved H3 worker acceptance; never read runtime logs."""
import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
C=ROOT/'comfyui/ui-groups-20261021'
OWN=Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')
PREVIOUS=Path('/home/thhel/tmp/.codex-h3-res20-compare-20261020-6f0b84c2')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('case');parser.add_argument('--package',type=Path,required=True);parser.add_argument('--worker-package',type=Path,required=True);parser.add_argument('--seed',type=int,default=42);parser.add_argument('--variant',choices=['reference_default','reference_shifted','i2v_shifted']);parser.add_argument('--project-fps',default='24');parser.add_argument('--upscale',type=float,default=1.);parser.add_argument('--lora',action='store_true');args=parser.parse_args();assert args.case.isalnum() and 0<=args.seed<=2**64-1 and math.isfinite(args.upscale) and 1<=args.upscale<=4
    from fractions import Fraction
    project_fps=Fraction(args.project_fps);assert project_fps>0
    project_rate={'num':project_fps.numerator,'den':project_fps.denominator}
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(OWN)]))['filesystems'][0]
    assert(mount['target'],mount['source'],mount['fstype'])==('/home','/dev/mapper/cryptroot[/@home]','btrfs')and not os.statvfs(OWN).f_flag&os.ST_RDONLY
    package=args.package.resolve();native=args.worker_package.resolve()
    assert package.is_relative_to(C/'packages')and native.is_relative_to(C/'worker-packages')
    case=OWN/args.case;case.mkdir()
    for name in ['project','state','output','tmp','cache','config','share','models','evidence','components']:(case/name).mkdir()
    env=dict(os.environ,TMPDIR=str(case/'tmp'),XDG_CACHE_HOME=str(case/'cache'),XDG_CONFIG_HOME=str(case/'config'),XDG_DATA_HOME=str(case/'share'),QWEN_MODEL_ROOT=str(case/'models'),CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',IMAGINE_COORDINATOR_SOCKET=str(ROOT/'.imagine/gpu.sock'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    for key,name in [('TORCH_HOME','torch'),('HF_HOME','hf'),('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda')]:
        path=case/'cache'/name;path.mkdir();env[key]=str(path)
    runtime=json.loads((PREVIOUS/'full-fp16-20261020-5e4a923b/candidate-runtime.json').read_text());h3=runtime['h3']
    import re
    protocol=int(re.search(r'pub const VERSION: u32 = (\d+)',(C/'source/shared/h3_protocol.rs').read_text())[1])
    h3['files'].pop(h3['binary']);h3['binary']=str(native/'imagine-h3-worker');digest=hashlib.sha256(Path(h3['binary']).read_bytes()).hexdigest();h3['files'][h3['binary']]=digest;h3['identity'].update(protocol=protocol,worker_sha256=digest)
    assert Path(h3['dependency_map']).is_relative_to(PREVIOUS)
    (case/'h3-dependencies.json').write_bytes(Path(h3['dependency_map']).read_bytes());h3['dependency_map']=str(case/'h3-dependencies.json');h3['environment'].pop('IMAGINE_REFERENCE_PROBE_OUTPUT',None)
    (case/'runtime.json').write_text(json.dumps(runtime,indent=2))
    # Only the explicit four model inputs and two already-known diffusion files.
    model_root=Path(h3['model_root'])
    known={'reference':'diffusion_models/minimax_h3/minimax_h3_ref2va_pruned_int8_convrot.safetensors','i2v':'diffusion_models/minimax_h3/minimaxH3.int8.safetensors','text_encoder':'text_encoders/minimax_h3/qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors','video_vae':'vae/minimax_h3/minimax_h3_video_vae_int8_convrot.safetensors','audio_vae':'vae/minimax_h3/minimax_h3_audio_vae_fp32.safetensors'}
    aliases={}
    if args.upscale>1:known['upscaler']='latent_upscale_models/h3_upscaler_lms_v0.1_fp32.safetensors'
    for key,path in known.items():
        target=model_root/path;assert target.is_file();alias=case/'components'/(key+' compatible alternative.weights');alias.symlink_to(target);aliases[key]=str(alias)
    own_lora=None
    if args.lora:
        own_lora=str(case/'components/own rank-one compatible.weights')
        subprocess.run(['/home/thhel/git/ComfyUI/.venv/bin/python','-c',"import sys,torch;from safetensors.torch import save_file;torch.set_num_threads(1);save_file({'audio_patch_proj.lora_up.weight':torch.linspace(-0.05,0.05,5376).reshape(5376,1),'audio_patch_proj.lora_down.weight':torch.linspace(-0.02,0.02,32).reshape(1,32)},sys.argv[1])",own_lora],env={**env,'CUDA_VISIBLE_DEVICES':''},check=True)
    # Reuse only our own previously verified destination contract, never user configuration.
    destination=json.loads((OWN/'ui50/evidence/endpoint.json').read_text())['destination'];destination['root']=str(case/'output');(case/'state/video-output.json').write_text(json.dumps(destination))
    with socket.socket()as port:port.bind(('127.0.0.1',0));address=port.getsockname();api='http://127.0.0.1:'+str(address[1])
    result={'case':str(case),'package':str(package),'worker_package':str(native),'runtime_logs_read':False,'production_changed':False,'gpu_coordination':'normal shared scheduler Ticket; no standalone fallback','variants':[],'passed':False}
    acceptance={'frames':39,'fps':24,'duration':39/24,'duration_tolerance':.002,'container':'mp4','video':'h264','pixel_format':'yuv420p','audio':'aac','sample_rate':32000,'channels':2,'audio_rms_min':1e-6,'worker_reaped_after_each':True,'interpretation':'Functional component/shift acceptance only; not Python numerical/algorithmic parity.'}
    (case/'evidence/acceptance-before-run.json').write_text(json.dumps(acceptance,indent=2))
    server=ui=None;session='';active=None
    def call(method,path,value=None,raw=None,mime='application/json',key=None):
        headers={'Content-Type':mime}
        if session:headers['X-Imagine-Session']=session
        if key:headers['Idempotency-Key']=key
        data=raw if raw is not None else json.dumps(value).encode()if value is not None else None
        with urllib.request.urlopen(urllib.request.Request(api+path,data=data,headers=headers,method=method),timeout=30)as response:return json.load(response)
    try:
        with(case/'evidence/server.log').open('x')as output:server=subprocess.Popen([str(package/'imagine-server'),'--project',str(case/'project'),'--data',str(case/'state'),'--profiles',str(C/'source/profiles'),'--bind',address[0]+':'+str(address[1]),'--backend','rust','--rust-runtime',str(case/'runtime.json')],env=env,stdout=output,stderr=subprocess.STDOUT)
        result.update(server_pid=server.pid,api=api);deadline=time.monotonic()+30
        while True:
            assert server.poll()is None,'Own server exited before ready; runtime log was not read.'
            try:session=call('GET','/v1/session')['id'];break
            except urllib.error.URLError:assert time.monotonic()<deadline;time.sleep(.1)
        assert call('GET','/v1/video-destination')['config']['root']==str(case/'output')
        for name,variant,shifts in [('reference_default','reference',(12.,3.)),('reference_shifted','reference',(9.5,2.25)),('i2v_shifted','i2v',(9.5,2.25))]:
            if args.variant and args.variant!=name:continue
            stage={'steps':20,'sampler':'res_multistep','scheduler':'simple','denoise':1.,'sigmas':[]}
            files={'model':aliases[variant],**{key:aliases[key]for key in ['text_encoder','video_vae','audio_vae']}}
            settings={'stage_one':stage,'stage_two':None,'base_steps_to_run':4,'components':files,'shift_video':shifts[0],'shift_audio':shifts[1]}
            model={'adapter':'h3','model':'minimax-h3','variant':variant,'profile':'h3-fl2va-i2va-v1'if variant=='i2v'else'h3-av-noref-v2','profile_revision':'2'}
            if args.upscale>1:
                settings['stage_two']={'sampling':{'steps':4,'sampler':'euler','scheduler':'simple','denoise':1.,'sigmas':[]},'model':model,'reference_model_file':aliases['reference'],'i2v_model_file':aliases['i2v'],'handoff':'re_noise_video_prediction','handoff_upscale_factor':args.upscale,'trained_latent_upscale_model':aliases['upscaler']}
            document=call('POST','/v1/video/projects',{'schema_version':1,'default_model':model,'delivery':{'width':416,'height':224,'fps':project_rate}});pid=document['project']['id'];bindings=[]
            if variant=='i2v':
                image=call('POST','/v1/video/assets',raw=(PREVIOUS/'own-portrait.png').read_bytes(),mime='image/png');document=call('POST',f'/v1/video/projects/{pid}/assets',{'schema_version':1,'expected_project_revision':document['project']['project_revision'],'asset_id':image['id']});asset=next(iter(document['project']['assets']));bindings=[{'id':'own_first_frame','asset':asset,'role':'first_frame','enabled':True}]
            prompt='An adult woman smiles and waves by the seashore. Gentle waves and a clear bell are audible.'
            scene={'id':'own_components_scene','start':{'num':0,'den':1},'duration':{'num':13,'den':8},'description':prompt,'model':'project_default','settings':[{'model':model,'schema_version':1,'values':settings}],'sections':[{'id':'own_components_section','start':{'num':0,'den':1},'duration':{'num':13,'den':8},'prompt':prompt}],'bindings':bindings,'continuity':None,'accepted_take':None}
            document=call('POST',f'/v1/video/projects/{pid}/edits',{'schema_version':1,'expected_project_revision':document['project']['project_revision'],'commands':([{'operation':'set_creative','creative':{'references':[],'sections':{},'h3_loras':{'version':1,'loras':[{'enabled':True,'name':own_lora,'strength_model':1.,'strength_clip':1.}], 'ui':{'show_strengths':'single','match':''}}}}]if args.lora else[])+[{'operation':'set_generation_defaults','generation':{'canvas':None,'settings':[],'seed':args.seed,'random_seed':False,'control_after_generate':'fixed'}},{'operation':'add_scene','scene':scene}]})
            folder=case/name;folder.mkdir();location=call('POST','/v1/video/locations',{'schema_version':1,'path':str(folder)});document=call('POST',f'/v1/video/projects/{pid}/save',{'schema_version':1,'expected_project_revision':document['project']['project_revision'],'location_id':location['location_id']})
            prefs=case/(name+'-ui')/'.imagine';prefs.mkdir(parents=True);(prefs/'ui-settings.json').write_text(json.dumps({'privacy_mode':False,'output_folder':str(case/'output')}))
            qa=case/(name+'-qa.json');qa.write_text(json.dumps({'destination':{**destination,'root':str(case)},'actions':[{'action':'wait_idle'},{'action':'open','folder':str(folder)},{'action':'wait_idle'},{'action':'click_control','name':'section-own_components_section'},{'action':'click_control','name':'timeline-generate'},{'action':'wait_idle'},{'action':'wait_time','milliseconds':300},{'action':'screenshot','file':name+'-submitted.png'},{'action':'quit'}]}))
            # A separate UI sends Generate. It never loads the user's settings or recent paths.
            with(case/'evidence'/(name+'-ui.log')).open('x')as output:ui=subprocess.Popen([str(package/'imagine-ui-qa'),'--project',str(prefs.parent),'--api',api,'--title','Imagine own component GPU acceptance','--qa-script',str(qa)],env=env,stdout=output,stderr=subprocess.STDOUT)
            assert ui.wait(timeout=150)==0
            qa_result=json.loads((case/'native-qa-result.json').read_text());(case/'evidence'/(name+'-qa-result.json')).write_text(json.dumps(qa_result,indent=2))
            ids=qa_result.get('job_ids',[]);active=ids[0]if len(ids)==1 else None
            assert qa_result['passed'],qa_result.get('error')
            assert len(ids)==1, 'Own UI must submit exactly one new job.'
            started=time.monotonic();observations=[];last=None
            while True:
                job=call('GET','/v1/jobs/'+active);stamp=(job['status'],job.get('phase'),(job.get('progress')or{}).get('completed'))
                if stamp!=last:
                    item={'status':stamp[0],'phase':stamp[1],'step':stamp[2],'seconds':time.monotonic()-started};observations.append(item);print(json.dumps({'variant':name,**item}),flush=True);last=stamp
                (case/'evidence/current.json').write_text(json.dumps({'variant':name,'job_id':active,'server_pid':server.pid,'observations':observations},indent=2))
                if job['status']in ['succeeded','failed','cancelled','interrupted']:break
                assert time.monotonic()-started<1800,'Own GPU wait/execution timeout.';time.sleep(.5)
            assert job['status']=='succeeded',job.get('error')
            assert call('GET','/v1/runtime')['engine']['pid']is None
            output=Path(job['output']['saved_path']);assert output.resolve().is_relative_to(case/'output')
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(output)],env=env));video=next(x for x in probe['streams']if x['codec_type']=='video');audio=next(x for x in probe['streams']if x['codec_type']=='audio')
            assert video['codec_name']=='h264'and video['pix_fmt']=='yuv420p'and int(video['nb_read_frames'])==39 and video['r_frame_rate']=='24/1'
            expected_canvas=[round(416*args.upscale/32)*32,round(224*args.upscale/32)*32]if args.upscale>1 else[416,224]
            assert [video['width'],video['height']]==expected_canvas
            if args.upscale>1:
                assert any(x['phase']=='upscaling_latents'for x in observations)
                assert max(x['step']or 0 for x in observations)==8
            assert video['color_space']=='bt709'and video['color_range']=='tv'
            assert audio['codec_name']=='aac'and int(audio['sample_rate'])==32000 and audio['channels']==2
            assert abs(float(probe['format']['duration'])-39/24)<=.002
            pcm=array.array('f',subprocess.check_output(['ffmpeg','-v','error','-i',str(output),'-map','0:a:0','-ar','32000','-ac','2','-f','f32le','pipe:1'],env=env));rms=math.sqrt(sum(x*x for x in pcm)/len(pcm));assert all(math.isfinite(x)for x in pcm)and rms>acceptance['audio_rms_min']
            document=call('GET',f'/v1/video/projects/{pid}');take=document['takes'][-1];assert take['provenance']['settings']['values']['components']==files and take['provenance']['settings']['values']['shift_video']==shifts[0]and take['provenance']['settings']['values']['shift_audio']==shifts[1]
            assert take['provenance']['seed']==args.seed and document['project']['generation']['seed']==args.seed and document['project']['delivery']['fps']==project_rate
            if args.lora:
                assert take['provenance']['resolved_loras']['loras'][0]['name']==own_lora
                assert take['provenance']['resolved_loras']['loras'][0]['strength_model']==1.
            result['variants'].append({'name':name,'seed':args.seed,'project_fps':project_rate,'job_id':active,'settings':settings,'output':str(output),'probe':probe,'audio_samples_per_channel':len(pcm)//2,'audio_rms':rms,'audio_peak':max(abs(x)for x in pcm),'worker_reaped':True,'observations':observations,'seconds':time.monotonic()-started});active=None
            (case/'evidence/results-in-progress.json').write_text(json.dumps(result,indent=2))
        result['passed']=True
    except Exception as error:result['error']=repr(error)
    finally:
        if active and session:
            try:
                call('POST','/v1/jobs/'+active+'/cancel',{})
                deadline=time.monotonic()+90
                while call('GET','/v1/jobs/'+active)['status']not in ['succeeded','failed','cancelled','interrupted']and time.monotonic()<deadline:time.sleep(.2)
                result['engine_after_cleanup']=call('GET','/v1/runtime')['engine']
            except Exception as error:result['cleanup_error']=repr(error)
        for name,process in [('ui',ui),('server',server)]:
            if process:
                if process.poll()is None:process.terminate()
                result[name+'_exit']=process.wait(timeout=90)
        result['own_processes_reaped']=all(p is None or p.poll()is not None for p in [ui,server]);(case/'evidence/result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result.get(k)for k in ['passed','error','cleanup_error','own_processes_reaped']}),flush=True)
    return not(result['passed']and result['own_processes_reaped'])

if __name__=='__main__':raise SystemExit(main())
