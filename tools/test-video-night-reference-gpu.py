#!/usr/bin/env python3
"""Own end-to-end cast image/voice GPU acceptance through the real coordinated queue."""
import argparse,copy,hashlib,json,math,os,re,shutil,struct,subprocess,time,traceback,urllib.request,urllib.error
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
OWN=Path('/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e')
PREVIOUS=Path('/run/media/veracrypt1/imagine/.codex-h3-res20-compare-20261020-6f0b84c2')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',action='store_true',required=True);parser.add_argument('--case',required=True);parser.add_argument('--timeline-audio',action='store_true');parser.add_argument('--reference-video',action='store_true');parser.add_argument('--ui',action='store_true');mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--standard-only',action='store_true');mode.add_argument('--observer-only',action='store_true');options=parser.parse_args()
    assert not (options.timeline_audio or options.reference_video) or options.standard_only, 'The observer approval covers only its original CP24 case.'
    assert not options.observer_only or options.case == 'reference-e2e-04', 'No new observer case is authorized.'
    assert not options.ui or (options.timeline_audio or options.reference_video) and options.standard_only
    assert re.fullmatch(r'reference-e2e-[0-9]{2}',options.case)
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(OWN)]))['filesystems'][0]
    assert (mount['source'],mount['fstype'])==('/dev/mapper/veracrypt1','ext3') and not os.statvfs(OWN).f_flag&os.ST_RDONLY
    case=OWN/options.case;case.mkdir()
    for name in ['evidence','fixtures','output','saved-project','artifacts','traces']:(case/name).mkdir()
    build_log='evidence/checkpoint26-product-build-01.log'if options.reference_video else 'evidence/checkpoint25-product-build-02.log'if options.timeline_audio else ('evidence/checkpoint24-product-build-02.log'if options.observer_only else 'evidence/checkpoint24-product-build-01.log')
    package=Path((OWN/build_log).read_text().splitlines()[-1])
    native=Path((OWN/('evidence/checkpoint26-native-build-04.log'if options.reference_video else 'evidence/checkpoint24-native-build-03.log')).read_text().splitlines()[-1])
    with Image.open(PREVIOUS/'own-portrait.png')as image: image.convert('RGB').resize((416,224),Image.Resampling.LANCZOS).save(case/'fixtures/reference.png')
    t=np.arange(16317,dtype=np.float32)/44100.;pcm=np.stack([.2*np.sin(2*np.pi*233*t),.17*np.cos(2*np.pi*377*t)],axis=-1).astype('<f4').tobytes()
    wave=b'RIFF'+struct.pack('<I',36+len(pcm))+b'WAVEfmt '+struct.pack('<IHHIIHH',16,3,2,44100,352800,8,32)+b'data'+struct.pack('<I',len(pcm))+pcm
    (case/'fixtures/reference.wav').write_bytes(wave);(case/'fixtures/reference.pcm').write_bytes(pcm)
    if options.reference_video:
      source=OWN/'reference-e2e-06/output/2026-10-21/video_0001.mp4'
      subprocess.run(['ffmpeg','-v','error','-nostdin','-threads','1','-filter_threads','1','-i',str(source),'-t','1.333333','-vf','scale=160:96,fps=30','-c:v','libx264','-pix_fmt','yuv420p','-crf','19','-c:a','aac','-ar','44100','-ac','2',str(case/'fixtures/reference.mp4')],check=True)
    result={'case' :str(case),'variants':[],'completed':False,'gpu_coordination':'real shared scheduler Ticket; no standalone GPU path','acceptance':{'container':'mp4','frames':124,'fps':24,'duration':31/6,'audio_rms_min':.0001,'payload_counts':[1,1,2,1,1]},'package':str(package),'native':str(native)}
    server=ui=None
    try:
      variants=[('observer',ROOT/'comfyui/h3-video-night-20261021/target-native/debug/examples/h3_reference_worker_probe')]if options.observer_only else [('standard',native/'imagine-h3-worker')]
      for variant,binary in variants:
        folder=case/variant;folder.mkdir()
        for name in ['project','state','tmp','cache','config','share','models']:(folder/name).mkdir()
        trace=case/'traces'/variant;trace.mkdir()
        runtime=json.loads((PREVIOUS/'full-fp16-20261020-5e4a923b/candidate-runtime.json').read_text());h3=runtime['h3'];h3['files'].pop(h3['binary']);h3['binary']=str(binary);digest=hashlib.sha256(binary.read_bytes()).hexdigest();h3['files'][str(binary)]=digest;h3['identity'].update(protocol=8 if options.reference_video else 7,worker_sha256=digest)
        dep=Path(h3['dependency_map']);assert dep.is_relative_to(PREVIOUS);(folder/'h3-dependencies.json').write_bytes(dep.read_bytes());h3['dependency_map']=str(folder/'h3-dependencies.json')
        if options.observer_only:h3['environment']['IMAGINE_REFERENCE_PROBE_OUTPUT']=str(trace)
        (folder/'runtime.json').write_text(json.dumps(runtime,indent=2))
        destination=json.loads((OWN/'q22/evidence/endpoint.json').read_text())['destination'];destination['root']=str(case/'output');(folder/'state/video-output.json').write_text(json.dumps(destination))
        env=dict(os.environ,TMPDIR=str(folder/'tmp'),XDG_CACHE_HOME=str(folder/'cache'),XDG_CONFIG_HOME=str(folder/'config'),XDG_DATA_HOME=str(folder/'share'),QWEN_MODEL_ROOT=str(folder/'models'),CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',IMAGINE_COORDINATOR_SOCKET=str(ROOT/'.imagine/gpu.sock'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        env.pop('IMAGINE_REFERENCE_PROBE_OUTPUT',None)
        for key,name in [('TORCH_HOME','torch'),('HF_HOME','hf'),('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda')]:env[key]=str(folder/'cache'/name)
        log=case/'evidence'/f'{variant}-server.log'
        with log.open('x')as f:server=subprocess.Popen([str(package/'imagine-server'),'--project',str(folder/'project'),'--data',str(folder/'state'),'--profiles',str(ROOT/'profiles'),'--bind','127.0.0.1:0','--backend','rust','--rust-runtime',str(folder/'runtime.json')],env=env,stdout=f,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+30;api=None
        while api is None:
          assert server.poll()is None,'own server exited';found=re.search(r'Imagine listening on (http://127[.]0[.]0[.]1:[0-9]+)',log.read_text())
          if found:api=found[1];break
          assert time.monotonic()<deadline;time.sleep(.05)
        session=''
        def request(method,route,value=None,raw=None,mime='application/json',key=None):
          headers={'Content-Type':mime}
          if session:headers['X-Imagine-Session']=session
          if key:headers['Idempotency-Key']=key
          req=urllib.request.Request(api+route,data=raw if raw is not None else json.dumps(value).encode()if value is not None else None,headers=headers,method=method)
          try:
            with urllib.request.urlopen(req,timeout=30)as response:return json.load(response)
          except urllib.error.HTTPError as error:raise RuntimeError(f'own {route}: {error.code}: {error.read().decode()}')from error
        session=request('GET','/v1/session')['id'];location=request('POST','/v1/video/locations',{'schema_version':1,'path':str(case/'saved-project')})
        if not (case/'saved-project/project.json').exists():
          model={'adapter':'h3','model':'minimax-h3','variant':'reference','profile':'h3-av-noref-v2','profile_revision':'2'}
          document=request('POST','/v1/video/projects',{'schema_version':1,'default_model':model,'delivery':{'width':416,'height':224,'fps':{'num':24,'den':1}}});project=document['project']['id'];assets=[]
          for filename,mime in ([('reference.mp4','video/mp4')]if options.reference_video else [('reference.png','image/png'),('reference.wav','audio/wav')]):
            upload=request('POST','/v1/video/assets',raw=(case/'fixtures'/filename).read_bytes(),mime=mime)
            before=set(document['project']['assets']);document=request('POST',f'/v1/video/projects/{project}/assets',{'schema_version':1,'expected_project_revision':document['project']['project_revision'],'asset_id':upload['id']});assets.append(next(iter(set(document['project']['assets'])-before)))
          settings={'ref_image_size':'Standard','stage_one':{'steps':20,'sampler':'res_multistep','scheduler':'simple','denoise':1.,'sigmas':[]},'stage_two':None,'base_steps_to_run':4}
          scene={'id':'own_reference_scene','start':{'num':0,'den':1},'duration':{'num':31,'den':6},'description':'The adult woman smiles gently and waves in a quiet room.','model':'project_default','settings':[{'model':model,'schema_version':1,'values':settings}],'sections':[{'id':'own_reference_section','start':{'num':0,'den':1},'duration':{'num':31,'den':6},'prompt':'@keeper says [English] "Hello!" and waves gently. Audio: quiet room ambience.'}],'bindings':[],'continuity':None,'accepted_take':None}
          reference={'id':'own_keeper','name':'Own fictional keeper','alias':'keeper','kind':'character','description':'the adult woman in the reference picture','tags':[],'enabled':True,'strength':1.,'retention':'','images':[{'id':'own_reference_image','asset':assets[0],'enabled':True,'preview':False,'contribution':'facial identity and appearance'}],'voice':{'language':'English','description':'clear speech','assets':[]if options.reference_video else[assets[1]]},'origin':None}
          creative={'references':[reference],'sections':{}}
          if options.reference_video:
            reference['images']=[]
            scene['sections'][0]['prompt']='The adult woman smiles gently and waves. Audio: quiet room ambience.'
            creative['sections']={'own_reference_section':{'images':[],'videos':[{'id':'own-video-binding','asset':assets[0],'role':'video_reference','enabled':False,'video':{'role':'video_reference','custom_role':'','source_in':{'num':1,'den':10},'source_out':{'num':11,'den':10},'soundtrack_usage':'ignore'}}],'dialogue_intent':None}}
          commands=[{'operation' :'add_scene','scene':scene},{'operation':'set_creative','creative':creative}]
          if options.timeline_audio:
            reference['images']=[];reference['voice']['assets']=[]
            creative['sections']={'own_reference_section':{'images':[{'image_id':'own-section-picture','asset':assets[0],'role':'visual_reference'}],'dialogue_intent':None}}
            clip={'id':'own-timeline-reference','asset':assets[1],'start':{'num':1,'den':8},'duration':{'num':1,'den':4},'source_in':{'num':1,'den':20},'source_out':{'num':3,'den':10},'volume':35.,'normalization':{},'fade_in':.1,'fade_out':.1,'enabled':True,'locked':False,'name':'Own trimmed reference','lane':0,'usage':{'generation_reference':True,'final_mix':False},'reference':{'purpose':'voice_timbre','target_reference_id':'own_keeper','relationship':'reference','description':'Own timeline voice timbre','strength':1.}}
            commands.append({'operation':'set_audio_tracks','tracks':[{'id':'own-reference-track','clips':[clip]}]})
          document=request('POST',f'/v1/video/projects/{project}/edits',{'schema_version':1,'expected_project_revision':document['project']['project_revision'],'commands':commands})
          document=request('POST',f'/v1/video/projects/{project}/save',{'schema_version':1,'expected_project_revision':document['project']['project_revision'],'location_id':location['location_id']})
        else:document=request('POST','/v1/video/projects/open',{'schema_version':1,'location_id':location['location_id']});project=document['project']['id']
        compiled=request('GET',f'/v1/video/projects/{project}/scenes/own_reference_scene/compiled');assert compiled['runtime_issues']==[]
        if not options.reference_video:assert len(compiled['reference_manifest']['images'])==1 and len(compiled['reference_manifest']['audios'])==1
        if options.timeline_audio:
          image=compiled['reference_manifest']['images'][0];audio=compiled['reference_manifest']['audios'][0]
          assert image['roles']==['section_picture']and image['section_bindings'][0]['resolved_role']=='influence'
          assert audio['role']=='standalone_timeline_audio_reference'and audio['target_subject_label']=='<Subject 1>'and audio['source_in']==.05 and audio['source_out']==.3
          result['timeline_audio_manifest_verified']=True;result['section_picture_manifest_verified']=True
        (case/'evidence'/f'{variant}-compiled.json').write_text(json.dumps(compiled,indent=2))
        started=time.monotonic()
        if options.ui:
          ui_case=case/'ui-start';ui_case.mkdir();prefs=ui_case/'project/.imagine';prefs.mkdir(parents=True)
          (prefs/'ui-settings.json').write_text(json.dumps({'output_folder':str(case/'output'),'privacy_mode':False}))
          destination=json.loads((folder/'state/video-output.json').read_text());destination['root']=str(ui_case)
          actions=[{'action':'wait_idle'},{'action':'open','folder':str(case/'saved-project')},{'action':'wait_idle'},{'action':'click_control','name':'section-own_reference_section'},{'action':'click_control','name':'h3-timeline-audio-references'},{'action':'wait_idle'},{'action':'click_control','name':'h3-timeline-audio-references'},{'action':'wait_idle'},{'action':'click_control','name':'compiled-open'},{'action':'wait_idle'},{'action':'expect','compiled_contains':'Own timeline voice timbre','compiled_issues':0},{'action':'screenshot','file':'section-and-timeline-review.png'},{'action':'click_control','name':'compiled-close'},{'action':'click_control','name':'timeline-generate'},{'action':'wait_status','status':'running','count':1},{'action':'screenshot','file':'actual-gpu-submitted.png'},{'action':'quit'}]
          if options.reference_video:
            actions=[{'action':'wait_idle'},{'action':'open','folder':str(case/'saved-project')},{'action':'wait_idle'},{'action':'click_control','name':'section-own_reference_section'},{'action':'click_control','name':'media-use-'+assets[0]},{'action':'wait_idle'},{'action':'click_control','name':'video-soundtrack-'+assets[0]},{'action':'click_control','name':'video-soundtrack-'+assets[0]+'-both'},{'action':'wait_idle'},{'action':'click_control','name':'compiled-open'},{'action':'wait_idle'},{'action':'expect','compiled_contains':'<Video 1>','compiled_issues':0},{'action':'screenshot','file':'video-paired-audio-review.png'},{'action':'click_control','name':'compiled-close'},{'action':'click_control','name':'timeline-generate'},{'action':'wait_status','status':'running','count':1},{'action':'screenshot','file':'actual-video-audio-gpu-submitted.png'},{'action':'quit'}]
          plan=ui_case/'qa-plan.json';plan.write_text(json.dumps({'destination':destination,'actions':actions},indent=2))
          ui_env=dict(env,CUDA_VISIBLE_DEVICES='',LIBGL_ALWAYS_SOFTWARE='1')
          with(case/'evidence/ui-start.log').open('x')as f:ui=subprocess.Popen([str(package/'imagine-ui'),'--project',str(ui_case/'project'),'--api',api,'--title','Imagine Video Night - own Section + timeline audio GPU QA','--qa-script',str(plan)],env=ui_env,stdout=f,stderr=subprocess.STDOUT)
          result.update(ui_pid=ui.pid,server_pid=server.pid,api=api)
          (case/'evidence/ui-running.json').write_text(json.dumps(result,indent=2))
          print(json.dumps({'own_ui_pid':ui.pid,'own_server_pid':server.pid,'api':api}),flush=True)
          deadline=time.monotonic()+130;job_id=None
          while job_id is None:
            status=request('GET','/v1/runtime');job_id=status['engine'].get('job_id')
            if job_id:break
            assert ui.poll()is None and time.monotonic()<deadline,'UI did not submit own job';time.sleep(.05)
          result['ui_pid']=ui.pid;result['submission']='actual native UI Generate';job={'id':job_id}
        else:job=request('POST','/v1/jobs',{'kind':'video_scene','schema_version':1,'project_id':project,'expected_project_revision':document['project']['project_revision'],'scene_id':'own_reference_scene','seed':42},key=options.case+'-'+variant);job_id=job['id']
        last=None;observations=[]
        (case/'evidence/running.json').write_text(json.dumps({'server_pid':server.pid,'job_id':job_id,'api':api,'variant':variant}))
        while True:
          job=request('GET',f'/v1/jobs/{job_id}');state=(job['status'],job.get('phase'),(job.get('progress')or{}).get('completed'))
          if state!=last:row={'variant':variant,'status':state[0],'phase':state[1],'step':state[2],'seconds':time.monotonic()-started};observations.append(row);print(json.dumps(row),flush=True);last=state
          if job['status']in ['succeeded','failed','cancelled']:break
          assert time.monotonic()-started<900,'own job timeout';time.sleep(.25)
        (case/'evidence'/f'{variant}-job.json').write_text(json.dumps(job,indent=2));assert job['status']=='succeeded',job.get('error')
        status=request('GET','/v1/runtime');assert status['engine']['pid']is None
        if options.ui:
          result['ui_exit']=ui.wait(timeout=30);result['ui_qa']=json.loads((ui_case/'native-qa-result.json').read_text());assert result['ui_qa']['passed'];ui=None
          saved=request('GET',f'/v1/video/projects/{project}')
          result['ui_setting_persisted']=(saved['project']['creative']['sections']['own_reference_section']['videos'][0]['video']['soundtrack_usage']=='both' if options.reference_video else saved['project']['scenes'][0]['settings'][0]['values']['audio_reference_mode']=='Timeline Audio References');assert result['ui_setting_persisted']
        if options.reference_video:
          compiled=request('GET',f'/v1/video/projects/{project}/scenes/own_reference_scene/compiled')
          manifest=compiled['reference_manifest'];assert compiled['runtime_issues']==[]
          assert len(manifest['videos'])==1 and len(manifest['audios'])==1 and len(manifest['images'])==0
          assert manifest['audios'][0]['paired_video_key']==manifest['videos'][0]['native_key']
          assert manifest['presentation_order']==[{'kind':'audio','label':'<Audio 1>'},{'kind':'video','label':'<Video 1>'}]
          result['paired_video_manifest_verified']=True
          (case/'evidence/standard-compiled-after-ui.json').write_text(json.dumps(compiled,indent=2))
        output=Path(job['output']['saved_path']);assert output.resolve().is_relative_to(case/'output')
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(output)]))
        v=next(s for s in probe['streams']if s['codec_type']=='video');a=next(s for s in probe['streams']if s['codec_type']=='audio')
        assert v['codec_name']=='h264' and v['pix_fmt']=='yuv420p' and int(v['nb_read_frames'])==124 and v['r_frame_rate']=='24/1'
        assert a['codec_name']=='aac' and int(a['sample_rate'])==32000 and a['channels']==2 and abs(float(probe['format']['duration'])-31/6)<.002
        decoded=np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-i',str(output),'-map','0:a:0','-f','f32le','pipe:1']),np.float32);rms=float(np.sqrt(np.mean(decoded.astype(np.float64)**2)));assert rms>.0001
        video_bytes=subprocess.check_output(['ffmpeg','-v','error','-i',str(output),'-map','0:v:0','-pix_fmt','rgb24','-f','rawvideo','pipe:1'])
        row={'name':variant,'output':str(output),'seconds':time.monotonic()-started,'audio_rms':rms,'audio_peak':float(np.max(np.abs(decoded))),'video_sha256':hashlib.sha256(video_bytes).hexdigest(),'pcm_sha256':hashlib.sha256(decoded.tobytes()).hexdigest(),'worker_reaped':True,'probe':probe,'observations':observations};result['variants'].append(row)
        server.terminate();row['server_exit']=server.wait(timeout=60);row['server_reaped']=True;server=None
        (case/'evidence/result-in-progress.json').write_text(json.dumps(result,indent=2));print(json.dumps({'variant_passed':variant,'rms':rms}),flush=True)
      if options.observer_only:
        names=['reference_image_0_prepared','reference_image_0_latent','reference_audio_0_resampled','reference_audio_0_latent','reference_payload_counts','output_video_latent','output_audio_latent']
        tensors={name:json.loads((trace/f'{name}.json').read_text())for name in names}
        assert all(v['finite']and v['samples']>0 for v in tensors.values())
        counts=np.fromfile(trace/'reference_payload_counts.f32',np.float32).tolist();assert counts==result['acceptance']['payload_counts']
        standard=json.loads((OWN/'reference-e2e-03/evidence/result.json').read_text())['variants'][0]
        result['standard_output_byteidentical']={key:standard[key]==result['variants'][0][key]for key in ['video_sha256','pcm_sha256']};assert all(result['standard_output_byteidentical'].values())
        result['observed_tensors']=tensors;result['observed_payload_counts']=counts
      result['observer_status']='explicitly declared in candidate H3 runtime contract; user approved'if options.observer_only else 'not requested'
      result['raw_gpu_reference_tensor_observation']=options.observer_only
      result['completed']=len(result['variants'])==1
    except BaseException as error:result['error']=str(error)or repr(error);result['traceback']=traceback.format_exc();print(json.dumps({'own_test_error':result['error'],'traceback':result['traceback']}),flush=True)
    finally:
      if ui and ui.poll()is None:ui.terminate();ui.wait(timeout=30)
      if server and server.poll()is None:server.terminate();server.wait(timeout=60)
      result['own_servers_reaped']=server is None or server.poll()is not None;(case/'evidence/result.json').write_text(json.dumps(result,indent=2));print(json.dumps({'completed':result['completed'],'variants':len(result['variants']),'own_servers_reaped':result['own_servers_reaped']}),flush=True)
    return 0 if result['completed']and result['own_servers_reaped']else 1
if __name__=='__main__':raise SystemExit(main())
