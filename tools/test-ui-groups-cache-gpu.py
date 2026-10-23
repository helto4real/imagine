#!/usr/bin/env python3
"""Own UI Generate → real coordinated worker → raw RGB8/PCM and MP4 cache equivalence."""
import argparse,array,hashlib,json,math,os,re,socket,struct,subprocess,time,urllib.request,urllib.error
from pathlib import Path
from ui_groups_paths import relocated
ROOT=Path(__file__).resolve().parents[1]
C=ROOT/'comfyui/ui-groups-20261021'
OWN=Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')
PREVIOUS=Path('/home/thhel/tmp/.codex-h3-res20-compare-20261020-6f0b84c2')
def main():
    p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('--package',type=Path,required=True);p.add_argument('--worker-package',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');p.add_argument('--reference-mode',choices=['image_audio','video','all'],default='image_audio');p.add_argument('--image-fixture',choices=['portrait','synthetic'],default='portrait');p.add_argument('--preview-compare',action='store_true');p.add_argument('--steps',type=int,default=20);a=p.parse_args();assert a.case.isalnum() and a.steps>0
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(OWN)]))['filesystems'][0]
    assert (mount['target'],mount['source'],mount['fstype'])==('/home','/dev/mapper/cryptroot[/@home]','btrfs') and not os.statvfs(OWN).f_flag&os.ST_RDONLY
    package=a.package.resolve();native=a.worker_package.resolve();assert package.is_relative_to(C/'packages') and native.is_relative_to(C/'worker-packages')
    case=OWN/a.case;case.mkdir()
    for name in ['project','state','output','tmp','cache','config','share','models','evidence','fixtures','saved-project']:(case/name).mkdir()
    env=dict(os.environ,TMPDIR=str(case/'tmp'),XDG_CACHE_HOME=str(case/'cache'),XDG_CONFIG_HOME=str(case/'config'),XDG_DATA_HOME=str(case/'share'),QWEN_MODEL_ROOT=str(case/'models'),CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',IMAGINE_COORDINATOR_SOCKET=str(ROOT/'.imagine/gpu.sock'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    for key,name in [('TORCH_HOME','torch'),('HF_HOME','hf'),('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda')]:
        path=case/'cache'/name;path.mkdir();env[key]=str(path)
    def rebase(value):
        if isinstance(value,dict):return {str(relocated(k)) if k.startswith('/run/media/veracrypt1/imagine/') else k:rebase(v)for k,v in value.items()}
        if isinstance(value,list):return [rebase(v)for v in value]
        return str(relocated(value))if isinstance(value,str)and value.startswith('/run/media/veracrypt1/imagine/')else value
    runtime=rebase(json.loads((PREVIOUS/'full-fp16-20261020-5e4a923b/candidate-runtime.json').read_text()));h3=runtime['h3'];h3['files'].pop(h3['binary']);h3['binary']=str(native/'imagine-h3-worker');digest=hashlib.sha256(Path(h3['binary']).read_bytes()).hexdigest();h3['files'][h3['binary']]=digest
    protocol=int(re.search(r'pub const VERSION: u32 = (\d+)',(C/'source/shared/h3_protocol.rs').read_text())[1]);h3['identity'].update(protocol=protocol,worker_sha256=digest)
    dep=Path(h3['dependency_map']);assert dep.is_relative_to(PREVIOUS);(case/'h3-dependencies.json').write_text(json.dumps(rebase(json.loads(dep.read_text()))));h3['dependency_map']=str(case/'h3-dependencies.json');h3['environment'].pop('IMAGINE_REFERENCE_PROBE_OUTPUT',None);(case/'runtime.json').write_text(json.dumps(runtime,indent=2))
    destination=json.loads((OWN/'qtransitioncp30n/evidence/endpoint.json').read_text())['destination'];destination['root']=str(case/'output');(case/'state/video-output.json').write_text(json.dumps(destination))
    # Only previous own generated portrait and new synthetic tone/video, all directly encrypted.
    (case/'fixtures/portrait.png').write_bytes((PREVIOUS/'own-portrait.png').read_bytes())
    if a.image_fixture=='synthetic':
        # A separate, explicitly smaller input workload; never replace the failed large case.
        subprocess.run(['/usr/bin/ffmpeg','-v','error','-nostdin','-y','-f','lavfi','-i','testsrc2=size=416x224:rate=1','-frames:v','1',str(case/'fixtures/portrait.png')],env=env,check=True)
    pcm=b''.join(struct.pack('<ff',.2*math.sin(i*.033),.17*math.cos(i*.047)) for i in range(16317))
    wav=b'RIFF'+struct.pack('<I',36+len(pcm))+b'WAVEfmt '+struct.pack('<IHHIIHH',16,3,2,44100,352800,8,32)+b'data'+struct.pack('<I',len(pcm))+pcm;(case/'fixtures/tone.wav').write_bytes(wav)
    subprocess.run(['/usr/bin/ffmpeg','-v','error','-nostdin','-f','lavfi','-i','testsrc2=size=160x96:rate=24:duration=1','-an','-c:v','libx264','-pix_fmt','yuv420p','-crf','19',str(case/'fixtures/reference.mp4')],env=env,check=True)
    availability=subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],env=env,text=True).strip()
    if a.preview_compare and int(availability.split(',')[0])>0:
        print(json.dumps({'blocked':'GPU currently active; no worker/job started','aggregate':availability}),flush=True);return 2
    expected_counts={'image_audio':[1,0,1],'video':[0,1,0],'all':[1,1,1]}[a.reference_mode];encodings=sum(expected_counts)
    result={'case':str(case),'package':str(package),'worker_package':str(native),'worker_sha256':digest,'protocol':protocol,'reference_mode':a.reference_mode,'gpu_availability_aggregate':availability,'gpu_coordination':'normal shared scheduler Ticket; no standalone path','runtime_logs_read':False,'production_changed':False,'variants':[],'passed':False}
    acceptance={'frames':39,'fps':24,'duration':39/24,'shape':[416,224],'steps':20,'sampler':'res_multistep','scheduler':'simple','shifts':[12,3],'seed':42,'references':['own portrait','synthetic video without audio','synthetic stereo tone'],'raw_rgb':'HWC RGB8 wire before export','raw_audio':'stereo float32 wire before export','equivalence':'byte-identical raw RGB8/PCM and decoded MP4 video/audio','cache':'3 misses/entries then 3 hits with unchanged entries then off/zero retained bytes','scope':'Rust cache miss/hit/off; not Python full pipeline or a quality benchmark'}
    acceptance.update(reference_mode=a.reference_mode,image_fixture=a.image_fixture,reference_counts=expected_counts,cache=f'{encodings} misses/entries then {encodings} hits with unchanged entries then off/zero retained bytes')
    acceptance['steps']=a.steps
    if a.preview_compare:acceptance.update(cache='disabled in both variants',scope='Rust preview off/on result regression; not Python parity or quality acceptance',preview={'interval_seconds':.05,'max_resolution':128,'frames':4})
    (case/'evidence/acceptance-before-run.json').write_text(json.dumps(acceptance,indent=2))
    server=ui=None;session='';api='';active=None
    def call(method,path,value=None,raw=None,mime='application/json'):
        headers={'Content-Type':mime}
        if session:headers['X-Imagine-Session']=session
        data=raw if raw is not None else json.dumps(value).encode() if value is not None else None
        with urllib.request.urlopen(urllib.request.Request(api+path,data=data,headers=headers,method=method),timeout=30) as r:return json.load(r)
    try:
        with (case/'evidence/server.log').open('x') as out:server=subprocess.Popen([str(C/'target/debug/examples/h3_cache_acceptance_server'),str(case)],env=env,stdout=out,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+30
        while not (case/'evidence/endpoint.json').exists():
            assert server.poll() is None and time.monotonic()<deadline,'Own server startup failed; runtime log not read';time.sleep(.05)
        api=json.loads((case/'evidence/endpoint.json').read_text())['api'];session=call('GET','/v1/session')['id'];result.update(server_pid=server.pid,api=api)
        model={'adapter':'h3','model':'minimax-h3','variant':'reference','profile':'h3-av-noref-v2','profile_revision':'2'}
        doc=call('POST','/v1/video/projects',{'schema_version':1,'default_model':model,'delivery':{'width':416,'height':224,'fps':{'num':24,'den':1}}});pid=doc['project']['id'];assets=[]
        for name,mime in [('portrait.png','image/png'),('tone.wav','audio/wav'),('reference.mp4','video/mp4')]:
            upload=call('POST','/v1/video/assets',raw=(case/'fixtures'/name).read_bytes(),mime=mime);old=set(doc['project']['assets']);doc=call('POST',f'/v1/video/projects/{pid}/assets',{'schema_version':1,'expected_project_revision':doc['project']['project_revision'],'asset_id':upload['id']});assets.append((set(doc['project']['assets'])-old).pop())
        settings={'ref_image_size':'Standard','stage_one':{'steps':a.steps,'sampler':'res_multistep','scheduler':'simple','denoise':1.,'sigmas':[]},'stage_two':None}
        duration={'num':13,'den':8};scene={'id':'own_cache_scene','start':{'num':0,'den':1},'duration':duration,'description':'The adult keeper smiles and waves gently in a quiet room.','model':'project_default','settings':[{'model':model,'schema_version':1,'values':settings}],'sections':[{'id':'own_cache_section','start':{'num':0,'den':1},'duration':duration,'prompt':'@keeper says [English] "Hello!" and waves gently. Audio: a clear soft bell and quiet ambience.'}],'bindings':[],'continuity':None,'accepted_take':None}
        creative={'references':[{'id':'own_keeper','name':'Own fictional keeper','alias':'keeper','kind':'character','description':'the adult woman in the reference picture','tags':[],'enabled':True,'strength':1.,'retention':'','images':[{'id':'own_image','asset':assets[0],'enabled':True,'preview':False,'contribution':'identity and appearance'}],'voice':{'language':'English','description':'clear tone','assets':[assets[1]]},'origin':None}],'sections':{'own_cache_section':{'images':[],'videos':[{'id':'own_video','asset':assets[2],'role':'video_reference','enabled':True,'video':{'role':'video_reference','custom_role':'','source_in':{'num':0,'den':1},'source_out':{'num':1,'den':1},'soundtrack_usage':'ignore'}}],'dialogue_intent':None}}}
        if a.reference_mode=='image_audio':creative['sections']={}
        elif a.reference_mode=='video':
            creative['references']=[];scene['sections'][0]['prompt']='The colored pattern moves gently. Audio: quiet ambience.'
        doc=call('POST',f'/v1/video/projects/{pid}/edits',{'schema_version':1,'expected_project_revision':doc['project']['project_revision'],'commands':[{'operation':'set_generation_defaults','generation':{'canvas':None,'settings':[],'seed':42,'random_seed':False,'control_after_generate':'fixed'}},{'operation':'add_scene','scene':scene},{'operation':'set_creative','creative':creative}]})
        location=call('POST','/v1/video/locations',{'schema_version':1,'path':str(case/'saved-project')});doc=call('POST',f'/v1/video/projects/{pid}/save',{'schema_version':1,'expected_project_revision':doc['project']['project_revision'],'location_id':location['location_id']})
        result['stage']='compile'
        compiled=call('GET',f'/v1/video/projects/{pid}/scenes/own_cache_scene/compiled');(case/'evidence/compiled.json').write_text(json.dumps(compiled,indent=2));assert compiled['runtime_issues']==[],compiled['runtime_issues'];manifest=compiled['reference_manifest'];counts=[len(manifest[k]) for k in ['images','videos','audios']];assert counts==expected_counts,{'reference_counts':counts}
        if a.prepare_only:
            result.update(passed=True,prepared_only=True,reference_counts=counts);return 0
        call('POST','/v1/video/reference-cache',{'schema_version':1,'limit_bytes':268435456,'enabled':True});call('POST','/v1/video/reference-cache/clear',{})
        for variant in (['preview_off','preview_on']if a.preview_compare else ['miss','hit','off']):
            result['stage']=variant
            raw_before=set((case/'raw').iterdir())
            if variant=='off' or a.preview_compare:call('POST','/v1/video/reference-cache',{'schema_version':1,'limit_bytes':268435456,'enabled':False})
            if a.preview_compare:call('POST','/v1/runtime/sampling-preview',{'enabled':variant=='preview_on','interval_seconds':.05,'max_resolution':128,'frames':4})
            before=call('GET','/v1/video/reference-cache');folder=case/(variant+'-ui');prefs=folder/'.imagine';prefs.mkdir(parents=True);(prefs/'ui-settings.json').write_text(json.dumps({'privacy_mode':False,'output_folder':str(case/'output')}))
            actions=[{'action':'wait_idle'},{'action':'open','folder':str(case/'saved-project')},{'action':'wait_idle'},{'action':'click_control','name':'section-own_cache_section'},{'action':'click_control','name':'timeline-generate'},{'action':'wait_idle'}]
            if variant=='preview_on':actions += [{'action':'wait_sampling_preview','minimum_sequence':1},{'action':'expect_sampling_preview','label':'gpu-visible','has_clip':True,'has_texture':True}]
            actions += [{'action':'wait_time','milliseconds':300},{'action':'screenshot','file':variant+'-submitted.png'},{'action':'quit'}];plan=case/(variant+'-qa.json');plan.write_text(json.dumps({'destination':{**destination,'root':str(case)},'actions':actions},indent=2))
            started=time.monotonic()
            with (case/'evidence'/(variant+'-ui.log')).open('x') as out:ui=subprocess.Popen([str(package/'imagine-ui-qa'),'--project',str(folder),'--api',api,'--title','Imagine own real H3 cache acceptance','--qa-script',str(plan),'--qa-window-size','1100','760'],env={**env,'CUDA_VISIBLE_DEVICES':'','LIBGL_ALWAYS_SOFTWARE':'1'},stdout=out,stderr=subprocess.STDOUT)
            assert ui.wait(timeout=150)==0;qa=json.loads((case/'native-qa-result.json').read_text());(case/'evidence'/(variant+'-qa-result.json')).write_text(json.dumps(qa,indent=2));assert qa['passed'],qa['error'];assert len(qa['job_ids'])==1;active=qa['job_ids'][0];observations=[];last=None;previews=[];preview_sequence=0
            while True:
                job=call('GET','/v1/jobs/'+active);stamp=(job['status'],job.get('phase'),(job.get('progress') or {}).get('completed'))
                if stamp!=last:
                    row={'variant':variant,'job_id':active,'status':stamp[0],'phase':stamp[1],'step':stamp[2],'seconds':time.monotonic()-started};observations.append(row);print(json.dumps(row),flush=True);last=stamp
                (case/'evidence/current.json').write_text(json.dumps({'server_pid':server.pid,'api':api,'variant':variant,'job_id':active,'observations':observations},indent=2))
                if a.preview_compare and job['status']=='running':
                    req=urllib.request.Request(api+f'/v1/jobs/{active}/sampling-preview?after={preview_sequence}',headers={'X-Imagine-Session':session})
                    with urllib.request.urlopen(req,timeout=30)as response:
                        if response.status!=204:
                            payload=response.read();n=struct.unpack('>I',payload[:4])[0];metadata=json.loads(payload[4:4+n]);rgb_preview=payload[4+n:]
                            assert len(rgb_preview)==metadata['width']*metadata['height']*metadata['frames']*3 and metadata['source_duration_seconds']==39/24
                            previews.append({'metadata':metadata,'rgb_sha256':hashlib.sha256(rgb_preview).hexdigest(),'rgb_bytes':len(rgb_preview),'pixels_persisted':False});preview_sequence=metadata['sequence']
                if job['status'] in ['succeeded','failed','cancelled','interrupted']:break
                assert time.monotonic()-started<1800,'Own execution/queue deadline';time.sleep(.5)
            result['last_terminal']={'job_id':active,'status':job['status'],'error':job.get('error'),'observations':observations,'cache':call('GET','/v1/video/reference-cache'),'engine':call('GET','/v1/runtime')['engine']}
            assert result['last_terminal']['engine']['pid'] is None
            active=None
            assert job['status']=='succeeded',job.get('error')
            completed=job['id']
            (case/'evidence'/(variant+'-job.json')).write_text(json.dumps(job,indent=2))
            after=call('GET','/v1/video/reference-cache');delta={k:after[k]-before[k] for k in ['hits','misses']}
            if variant=='miss':assert delta=={'hits':0,'misses':encodings} and after['entries']==encodings
            elif variant=='hit':assert delta=={'hits':encodings,'misses':0} and after['entries']==encodings
            else:assert delta=={'hits':0,'misses':encodings} and after['entries']==0 and after['used_bytes']==0
            created=set((case/'raw').iterdir())-raw_before;assert len(created)==1,{'new_raw_folders':list(map(str,created))}
            raw=created.pop();contract=json.loads((raw/'contract.json').read_text());assert contract['frames']==39 and contract['pcm_bytes']==52000*8
            output=Path(job['output']['saved_path']);assert output.resolve().is_relative_to(case/'output')
            probe=json.loads(subprocess.check_output(['/usr/bin/ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(output)],env=env));video=next(x for x in probe['streams'] if x['codec_type']=='video');audio=next(x for x in probe['streams'] if x['codec_type']=='audio');assert video['codec_name']=='h264' and video['pix_fmt']=='yuv420p' and int(video['nb_read_frames'])==39 and video['r_frame_rate']=='24/1' and video['color_space']=='bt709' and video['color_range']=='tv';assert audio['codec_name']=='aac' and int(audio['sample_rate'])==32000 and audio['channels']==2;assert abs(float(probe['format']['duration'])-39/24)<=.002
            rgb=subprocess.check_output(['/usr/bin/ffmpeg','-v','error','-i',str(output),'-map','0:v:0','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],env=env);decoded_pcm=subprocess.check_output(['/usr/bin/ffmpeg','-v','error','-i',str(output),'-map','0:a:0','-f','f32le','pipe:1'],env=env)
            floats=array.array('f',(raw/'audio.f32le').read_bytes());rms=math.sqrt(sum(v*v for v in floats)/len(floats));assert all(math.isfinite(v) for v in floats) and (a.preview_compare or rms>1e-6)
            if a.preview_compare:
                assert bool(previews)==(variant=='preview_on'),previews
                (case/'evidence'/(variant+'-previews.json')).write_text(json.dumps({'previews':previews,'ui_sampling_state':qa['privacy_captures'].get('sampling-preview-gpu-visible'),'pixels_persisted':False},indent=2))
            result['variants'].append({'name':variant,'job_id':completed,'raw':str(raw),'output':str(output),'cache_before':before,'cache_after':after,'cache_delta':delta,'raw_rgb_sha256':hashlib.sha256((raw/'frames.rgb').read_bytes()).hexdigest(),'raw_pcm_sha256':hashlib.sha256((raw/'audio.f32le').read_bytes()).hexdigest(),'decoded_rgb_sha256':hashlib.sha256(rgb).hexdigest(),'decoded_pcm_sha256':hashlib.sha256(decoded_pcm).hexdigest(),'pcm_rms':rms,'pcm_peak':max(abs(v) for v in floats),'probe':probe,'seconds':time.monotonic()-started,'observations':observations,'worker_reaped':True});(case/'evidence/results-in-progress.json').write_text(json.dumps(result,indent=2))
        control=result['variants'][0]
        for variant in result['variants'][1:]:
            for key in ['raw_rgb_sha256','raw_pcm_sha256','decoded_rgb_sha256','decoded_pcm_sha256']:assert variant[key]==control[key],(variant['name'],key)
        result['passed']=True
    except Exception as e:result['error']=repr(e)
    finally:
        if active and session:
            try:
                call('POST','/v1/jobs/'+active+'/cancel',{});deadline=time.monotonic()+90
                while call('GET','/v1/jobs/'+active)['status'] not in ['succeeded','failed','cancelled','interrupted'] and time.monotonic()<deadline:time.sleep(.2)
                result['engine_after_cleanup']=call('GET','/v1/runtime')['engine']
            except Exception as e:result['cleanup_error']=repr(e)
        for name,process in [('ui',ui),('server',server)]:
            if process:
                if process.poll() is None:process.send_signal(__import__('signal').SIGINT if name=='server' else __import__('signal').SIGTERM)
                result[name+'_exit']=process.wait(timeout=90)
        result['own_processes_reaped']=all(p is None or p.poll() is not None for p in [ui,server]);(case/'evidence/result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result.get(k) for k in ['passed','error','cleanup_error','own_processes_reaped']}),flush=True)
    return not result['passed']
if __name__=='__main__':raise SystemExit(main())
