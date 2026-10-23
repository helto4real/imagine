#!/usr/bin/env python3
"""Own native project/scene settings through CPU request preparation and direct MP4 export."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
OWN=Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')
BIN=ROOT/'comfyui/ui-groups-20261021/target/debug'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('case');parser.add_argument('--package',type=Path);parser.add_argument('--review',action='store_true');parser.add_argument('--compatible-export',action='store_true');parser.add_argument('--incompatible-export',action='store_true');parser.add_argument('--mode-draft',action='store_true');parser.add_argument('--components',action='store_true');parser.add_argument('--window-size',type=float,nargs=2);parser.add_argument('--floating',action='store_true');parser.add_argument('--seed',type=int,default=9876543210123);parser.add_argument('--seed-control',choices=['fixed','increment','decrement','randomize']);parser.add_argument('--repeat',action='store_true');parser.add_argument('--stage-toggle',action='store_true');parser.add_argument('--runs',action='store_true');parser.add_argument('--scene-kinds',action='store_true');parser.add_argument('--fps',type=float,default=24);parser.add_argument('--quality',default='Draft');parser.add_argument('--reference-quality',default='High');parser.add_argument('--upscale',type=float,default=1.);parser.add_argument('--loras',action='store_true');parser.add_argument('--lora-catalog',action='store_true');args=parser.parse_args();assert args.case.isalnum() and not(args.compatible_export and args.incompatible_export) and not(args.repeat and args.seed_control=='randomize')
    args.loras |= args.lora_catalog
    assert 0<=args.seed<=2**64-1
    import math
    assert math.isfinite(args.fps) and args.fps>0 and math.isfinite(args.upscale) and 1<=args.upscale<=4
    golden=json.loads((ROOT/'comfyui/ui-groups-20261021/source/tests/fixtures/h3-director-geometry.json').read_text())
    dims=next(c['shape'] for c in golden['cases'] if c['selection']['quality']==args.quality and c['selection']['ratio']=='4:3' and c['selection']['portrait'])
    width,height=dims['width'],dims['height']
    next_seed=lambda seed: min(seed+1,2**50) if args.seed_control=='increment' else min(max(0,seed-1),2**50) if args.seed_control=='decrement' else None if args.seed_control=='randomize' else seed
    queued_seeds=[args.seed]+([next_seed(args.seed)] if args.repeat else [])
    stored_seed=next_seed(queued_seeds[-1])
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(OWN)]))['filesystems'][0]
    assert (mount['target'],mount['source'],mount['fstype'])==('/home','/dev/mapper/cryptroot[/@home]','btrfs') and not os.statvfs(OWN).f_flag&os.ST_RDONLY
    case=OWN/args.case;case.mkdir()
    for name in ['evidence','tmp','cache','config','share','models']:(case/name).mkdir()
    env=dict(os.environ,TMPDIR=str(case/'tmp'),CUDA_VISIBLE_DEVICES='',LIBGL_ALWAYS_SOFTWARE='1',XDG_CACHE_HOME=str(case/'cache'),XDG_CONFIG_HOME=str(case/'config'),XDG_DATA_HOME=str(case/'share'),QWEN_MODEL_ROOT=str(case/'models'),PYTHONDONTWRITEBYTECODE='1')
    package=args.package.resolve() if args.package else None
    if package:assert package.is_relative_to(ROOT/'comfyui/ui-groups-20261021/packages')
    server_binary=package/'timeline_fixture_server' if package else BIN/'examples/timeline_fixture_server'
    ui_binary=package/'imagine-ui-qa' if package else BIN/'imagine-ui'
    server=ui=None;result={'case':str(case),'gpu':False,'runtime_logs_read':False,'production_changed':False}
    try:
        with(case/'evidence/server.log').open('x')as output:server=subprocess.Popen([str(server_binary),str(case),'--cast','--with-audio']+(['--lora-catalog']if args.lora_catalog else []),env=env,stdout=output,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+30
        while not(case/'evidence/endpoint.json').exists():
            assert server.poll() is None and time.monotonic()<deadline
            time.sleep(.05)
        endpoint=json.loads((case/'evidence/endpoint.json').read_text());api=endpoint['api'];assert endpoint['backend'].startswith('synthetic CPU')
        def call(path,body=None):
            req=urllib.request.Request(api+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=20)as response:return json.load(response)
        session=call('/v1/session')['id']
        def rpc(path,body=None):
            req=urllib.request.Request(api+path,data=json.dumps(body).encode()if body is not None else None,headers={'Content-Type':'application/json','X-Imagine-Session':session})
            with urllib.request.urlopen(req,timeout=20)as response:return json.load(response)
        project_location=rpc('/v1/video/locations',{'schema_version':1,'path':endpoint['project']})['location_id']
        document=rpc('/v1/video/projects/open',{'schema_version':1,'location_id':project_location})
        prefs=case/'ui/.imagine';prefs.mkdir(parents=True,exist_ok=True);(prefs/'ui-settings.json').write_text(json.dumps({'privacy_mode':False,'output_folder':str(case/'output')}))
        click=lambda name:{'action':'click_control','name':name};idle={'action':'wait_idle'}
        key=lambda name,**mods:{'action':'key','key':name,**mods}
        text=lambda value:{'action':'text','text':value}
        field=lambda name,value:[click(name),key('A',ctrl=True),text(value)if value else key('Delete')]
        scroll=lambda name:{'action':'scroll_to_control','scroll':'generation-scroll','name':name}
        expect=lambda path,value:{'action':'expect_project','pointer':path,'value':value}
        actions=[idle,{'action':'open','folder':endpoint['project']},idle,
            click('project-menu'),click('project-new'),idle,expect('/generation/canvas/quality','Standard'),expect('/generation/seed',0),expect('/generation/control_after_generate','randomize'),expect('/delivery/width',832),expect('/delivery/height',480),
            {'action':'open','folder':endpoint['project']},idle,click('section-section_qa_1'),click('project-menu'),click('project-generation'),{'action':'wait_time','milliseconds':300},{'action':'screenshot','file':'generation-dialog-open.png'},
            scroll('project-quality'),click('project-quality'),click('project-quality-'+args.quality),click('project-ratio'),click('project-ratio-4:3'),click('project-portrait'),
            scroll('project-seed'),*field('project-seed',str(args.seed)),scroll('reference-quality'),click('reference-quality'),click('reference-quality-'+args.reference_quality),
            scroll('base-sampler'),*field('base-sampler','euler'),scroll('base-sigmas'),*field('base-sigmas','1, 0.7, 0.4'),scroll('sampling-two-stages'),click('sampling-two-stages'),
            scroll('stage2-sigmas'),*field('stage2-sigmas','0.4, 0.2, 0'),scroll('generation-apply'),{'action':'wait_time','milliseconds':300},{'action':'screenshot','file':'project-two-stage-settings.png'},click('generation-apply'),idle,
            expect('/delivery/width',width),expect('/delivery/height',height),expect('/generation/seed',args.seed),
            {'action':'scroll_to_control','scroll':'inspector-scroll','name':'scene-sampling'},click('scene-sampling'),{'action':'scroll_to_control','scroll':'inspector-scroll','name':'scene-inherit-settings'},click('scene-inherit-settings'),idle,expect('/scenes/0/settings',[]),
            click('timeline-generate'),{'action':'wait_status','status':'succeeded','count':1},idle,{'action':'wait_frames','frames':10},
            {'action':'save'},idle,{'action':'screenshot','file':'project-generated-cpu-take.png'},
            click('project-menu'),click('project-generation'),scroll('project-seed'),*field('project-seed','123'),scroll('generation-cancel'),click('generation-cancel'),expect('/generation/seed',stored_seed),
            {'action':'quit'}]

        if args.upscale>1:
            at=next(i for i,a in enumerate(actions) if a.get('file')=='project-two-stage-settings.png')-1
            actions[at:at]=[scroll('handoff-renoise'),click('handoff-renoise'),scroll('stage2-upscale-factor'),*field('stage2-upscale-factor',str(args.upscale)),scroll('stage2-upscale-file'),*field('stage2-upscale-file','/own/compatible upscaler.weights'),{'action':'wait_time','milliseconds':300},{'action':'screenshot','file':'trained-upscale-settings.png'},scroll('generation-apply')]
        if args.loras:
            global_lora=case/'own-global compatible.weights';scene_lora=case/'own-scene compatible.weights'
            if args.lora_catalog:
                catalog_root=case/'models/loras';catalog_root.mkdir()
                global_lora=catalog_root/'own-global compatible.weights'
                (catalog_root/'Other 🙂.weights').write_bytes(b'Own unknown-format fixture, not a model')
            global_name=global_lora.name if args.lora_catalog else str(global_lora)
            header=json.dumps({'__metadata__':{'ss_output_name':'Own synthetic LoRA fixture','ss_base_model_version':'synthetic','ss_tag_frequency':json.dumps({'own':{'own fixture trigger':3}})},'weight':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}},separators=(',',':')).encode()
            header+=b' '*((-len(header))%8)
            for path in [global_lora,scene_lora]:path.write_bytes(len(header).to_bytes(8,'little')+header+b'\0'*4)
            at=next(i for i,a in enumerate(actions) if a.get('file')=='project-two-stage-settings.png')-1
            actions[at:at]=[scroll('project-loras'),click('project-loras'),click('lora-add'),*field('lora-0-file','Own unselected draft'if args.lora_catalog else global_name),*field('lora-0-strength','0.75'),{'action':'wait_time','milliseconds':300},{'action':'screenshot','file':'project-loras.png'},click('lora-0-info'),{'action':'wait_time','milliseconds':300},{'action':'screenshot','file':'lora-info.png'},click('lora-info-close'),click('lora-apply'),idle,scroll('generation-apply')]
            if args.lora_catalog:
                at=next(i for i,a in enumerate(actions)if a.get('file')=='project-loras.png')
                actions[at:at]=[*field('lora-filter','(?i)own'),{'action':'wait_time','milliseconds':500},{'action':'screenshot','file':'lora-invalid-js-filter.png'},*field('lora-filter','^own-global'),{'action':'wait_time','milliseconds':500},click('lora-refresh'),{'action':'wait_time','milliseconds':500},click('lora-0-catalog'),{'action':'wait_time','milliseconds':300},{'action':'screenshot','file':'lora-catalog-filtered.png'},click('lora-0-choice-'+global_name),*field('lora-filter','')]
            at=next(i for i,a in enumerate(actions) if a.get('name')=='timeline-generate')
            actions[at:at]=[{'action':'scroll_to_control','scroll':'inspector-scroll','name':'scene-loras'},click('scene-loras'),click('lora-override'),click('lora-merge'),click('lora-merge-Add To Global'),click('lora-add'),*field('lora-0-file',str(scene_lora)),*field('lora-0-strength','-1.25'),{'action':'wait_time','milliseconds':300},{'action':'screenshot','file':'scene-loras.png'},click('lora-apply'),idle,key('Z',ctrl=True),idle,expect('/creative',{**document['project']['creative'],'h3_loras':{'version':1,'loras':[{'enabled':True,'name':global_name,'strength_model':.75,'strength_clip':.75}],'ui':{'show_strengths':'single','match':''}}}),{'action':'screenshot','file':'scene-loras-undone.png'},key('Z',ctrl=True,shift=True),idle,expect('/creative/scene_loras/scene_qa_1/merge_mode','Add To Global')]
        if args.seed_control:
            at=next(i for i,a in enumerate(actions) if a.get('name')=='reference-quality')
            actions[at:at]=[scroll('project-seed-control'),click('project-seed-control'),click('project-seed-control-'+args.seed_control)]
        if args.loras:
            actions=[step for i,action in enumerate(actions) for step in ([action,{'action':'wait_time','milliseconds':250},{'action':'screenshot','file':f'lora-after-add-{i}.png'}] if action.get('name')=='lora-add' else [action])]
        at=next(i for i,a in enumerate(actions) if a.get('name')=='project-seed')
        actions[at:at]=[scroll('project-fps'),*field('project-fps',str(args.fps))]
        if args.scene_kinds:
            video=case/'Own source clip.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=blue:s=160x96:r=25:d=2','-f','lavfi','-i','sine=frequency=330:sample_rate=32000:duration=2','-c:v','libx264','-pix_fmt','yuv420p','-crf','19','-c:a','aac','-ac','2',str(video)],env=env,check=True)
            inspector=lambda name:{'action':'scroll_to_control','scroll':'inspector-scroll','name':name}
            choose=lambda kind:[inspector('scene-kind'),click('scene-kind'),click('scene-kind-'+kind),idle,expect('/creative/scene_kinds/scene_qa_1',kind)]
            extra=choose('imported')+[inspector('scene-clip-pick'),click('scene-clip-pick'),{'action':'import_picker','paths':[str(video)]},idle,{'action':'expect','imported_clips':1},key('Z',ctrl=True),idle,{'action':'expect','imported_clips':0},key('Z',ctrl=True,shift=True),idle,{'action':'expect','imported_clips':1},click('timeline-generate'),idle,{'action':'expect_error_contains','text':'Imported'}]
            for kind in ['generated','imported','extended','edited','placeholder']:
                extra += choose(kind)+[{'action':'expect','imported_clips':1}]
            extra += [{'action':'screenshot','file':'scene-types-retain-clip.png'}]+choose('generated')
            at=next(i for i,a in enumerate(actions) if a.get('name')=='scene-sampling')
            actions[at:at]=extra
        if args.repeat:
            at=next(i for i,a in enumerate(actions) if a.get('action')=='save')
            actions[at:at]=[idle,expect('/generation/seed',next_seed(args.seed)),click('timeline-generate'),{'action':'wait_status','status':'succeeded','count':2},idle,expect('/generation/seed',stored_seed)]
        if args.stage_toggle:
            at=next(i for i,a in enumerate(actions) if a.get('file')=='project-two-stage-settings.png')-2
            # Persist an inactive preset with unfinished raw text, reopen it,
            # then correct only the raw text before the real CPU job.
            actions[at:at]=[scroll('stage2-sigmas'),*field('stage2-sigmas','0.4, unfinished'),key('Tab'),scroll('sampling-two-stages'),{'action':'screenshot','file':'before-disable-stage.png'},click('sampling-two-stages'),scroll('generation-apply'),click('generation-apply'),idle,click('project-menu'),click('project-generation'),scroll('sampling-two-stages'),click('sampling-two-stages'),scroll('stage2-sigmas'),{'action':'wait_time','milliseconds':300},{'action':'screenshot','file':'inactive-stage-restored.png'},*field('stage2-sigmas','0.4, 0.2, 0')]

        if args.mode_draft:
            at=next(i for i,a in enumerate(actions) if a.get('file')=='generation-dialog-open.png')+1
            actions[at:at]=[scroll('base-sigmas'),*field('base-sigmas','1, unfinished'),scroll('project-model'),click('project-model'),click('project-model-i2v'),click('project-model'),click('project-model-reference'),scroll('generation-apply'),click('generation-apply'),idle,{'action':'wait_time','milliseconds':300},{'action':'screenshot','file':'invalid-sigmas-retained.png'},scroll('generation-cancel'),click('generation-cancel'),click('project-menu'),click('project-generation')]
        if args.compatible_export:
            compatible=case/'compatible';compatible.mkdir()
            actions[-1:-1]=[click('project-menu'),click('project-export-compatible'),{'action':'wait_time','milliseconds':300},{'action':'screenshot','file':'compatible-export-explanation.png'},{'action':'export_compatible','folder':str(compatible)},idle,{'action':'wait_time','milliseconds':300},{'action':'screenshot','file':'compatible-export-complete.png'},click('compatible-close'),expect('/id',document['project']['id'])]
        if args.incompatible_export:
            assert args.seed>2**53-1
            incompatible=case/'older-copy';incompatible.mkdir()
            actions[-1:-1]=[click('project-menu'),click('project-export-compatible'),{'action':'export_compatible','folder':str(incompatible)},idle,{'action':'expect_error_contains','text':'older project format'},expect('/id',document['project']['id']),expect('/generation/seed',stored_seed),{'action':'screenshot','file':'older-copy-exact-seed-error.png'}]
        at=next(i for i,a in enumerate(actions) if a.get('file')=='project-generated-cpu-take.png')+1
        actions[at:at]=[{'action':'hover_control','name':'run-menu'},{'action':'wait_time','milliseconds':700},{'action':'screenshot','file':'generation-run-tooltip.png'},click('run-menu'),{'action':'wait_time','milliseconds':300},{'action':'screenshot','file':'generation-run-menu-settled.png'},key('Escape'),{'action':'open','folder':str(case/'saved-project')},idle,expect('/generation/seed',stored_seed),click('section-section_qa_1')]
        if args.runs:
            at=next(i for i,a in enumerate(actions) if a.get('file')=='project-generated-cpu-take.png')+1
            extra=[]
            for i,mode in enumerate(['missing','same','fresh','dependents'],1):
                extra += [idle,click('run-menu')]
                if i==1:extra += [{'action':'wait_time','milliseconds':300},{'action':'screenshot','file':'generation-run-options.png'}]
                extra += [click('run-'+mode),{'action':'wait_run_status','status':'completed','count':i},idle]
            actions[at:at]=extra+[expect('/generation/seed',stored_seed),{'action':'save'},idle]
        if args.components:
            at=next(i for i,a in enumerate(actions) if a.get('name')=='base-sampler')
            own_fields=[('component-model','diffusion_models/own compatible β.weights'),('component-text','/own/compatible text.encoder'),('component-video-vae','vae/own compatible video.model'),('component-audio-vae','/own/audio-compatible.bin'),('component-video-shift','9.5'),('component-audio-shift','2.25')]
            extra=[scroll('component-files'),click('component-files')]
            for name,value in own_fields:extra += [scroll(name),*field(name,value)]
            extra += [{'action':'wait_time','milliseconds':300},{'action':'screenshot','file':'free-component-files-shifts.png'},scroll('component-model'),{'action':'wait_time','milliseconds':300},{'action':'screenshot','file':'free-component-files-top.png'},{'action':'hover_control','name':'component-model-browse'},{'action':'wait_time','milliseconds':700},{'action':'screenshot','file':'component-file-tooltip.png'},scroll('component-files'),click('component-files')]
            actions[at:at]=extra
            at=next(i for i,a in enumerate(actions) if a.get('name')=='stage2-sigmas')
            actions[at:at]=[scroll('stage2-reference-file'),*field('stage2-reference-file','/own/stage2-reference-compatible.bin'),scroll('stage2-i2v-file'),*field('stage2-i2v-file','/own/stage2-i2v-compatible.bin')]
        if args.review:
            at=next(i for i,a in enumerate(actions) if a.get('name')=='timeline-generate')
            assert not args.scene_kinds and not args.loras
            reviewed='Own reviewed lighthouse scene, changed after opening review.'
            actions[at:at+1]=[click('compiled-open'),idle,{'action':'expect','compiled_seed':args.seed,'compiled_stale':False},{'action':'screenshot','file':'review-exact-seed.png'},
                {'action':'scroll_to_control','scroll':'inspector-scroll','name':'scene-description'},*field('scene-description',reviewed),key('Tab'),{'action':'expect','compiled_stale':True},{'action':'click_control','name':'compiled-approve','expected_enabled':False},{'action':'wait_time','milliseconds':300},{'action':'expect','takes':len(document['takes'])},
                {'action':'screenshot','file':'review-stale-disabled.png'},click('compiled-refresh'),idle,{'action':'expect','compiled_seed':args.seed,'compiled_stale':False,'compiled_contains':reviewed},{'action':'screenshot','file':'review-refreshed.png'},click('compiled-approve')]
        # Let native smooth scrolling settle before typing into its moving fields.
        if args.window_size:
            actions[1:1]=[{'action':'resize','width':args.window_size[0],'height':args.window_size[1]},{'action':'wait_frames','frames':12}]
        if args.floating:
            actions[0:0]=[{'action':'wait_time','milliseconds':2500}]
        actions=[item for action in actions for item in ([action,{'action':'wait_time','milliseconds':250}] if action['action']=='scroll_to_control' else [action])]
        actions=[item for index,action in enumerate(actions) for item in ([action,{'action':'screenshot','file':f'stage-toggle-scroll-{index}.png'}] if action.get('action')=='scroll_to_control' and action.get('name')=='sampling-two-stages' else [action])]
        actions=[a for a in actions if not(a.get('action')=='expect_project' and a.get('pointer')=='/generation/seed' and a.get('value') is None)]
        plan=case/'qa-plan.json';plan.write_text(json.dumps({'destination':endpoint['destination'],'actions':actions},indent=2))
        command=[str(ui_binary),'--project',str(case/'ui'),'--api',api,'--title','Imagine UI Groups own Generation settings QA','--qa-script',str(plan)]
        if args.window_size:command+=['--qa-window-size',*[str(n) for n in args.window_size]]
        with(case/'evidence/ui.log').open('x')as output:ui=subprocess.Popen(command,env=env,stdout=output,stderr=subprocess.STDOUT)
        result.update(server_pid=server.pid,ui_pid=ui.pid,api=api);print(json.dumps(result),flush=True)
        if args.floating:
            assert os.environ.get('XDG_CURRENT_DESKTOP')=='Hyprland'
            time.sleep(1)
            assert ui.poll()is None
            # Target only the exact child we just created; no user windows/config are queried.
            operation=subprocess.run(['hyprctl','dispatch','setfloating','pid:'+str(ui.pid)],text=True,capture_output=True)
            result['own_window_operation']={'pid':ui.pid,'dispatcher':'setfloating','exit':operation.returncode,'stdout':operation.stdout,'stderr':operation.stderr}
            assert operation.returncode==0 and operation.stdout.strip()=='ok', 'Own test window could not be made floating.'
        result['ui_exit']=ui.wait(timeout=150);result['qa']=json.loads((case/'native-qa-result.json').read_text());assert result['qa']['passed']
        p=json.loads((case/'saved-project/project.json').read_text())['project']
        if args.seed_control=='randomize':
            stored_seed=p['generation']['seed'];assert isinstance(stored_seed,int) and 0<=stored_seed<2**50
        assert p['generation']['seed']==stored_seed and p['delivery']['width']==width and p['delivery']['height']==height
        assert p['scenes'][0]['settings']==[] and p['generation']['canvas']=={'ratio':'4:3','portrait':True,'quality':args.quality}
        captures=[json.loads(p.read_text()) for p in sorted((case/'evidence').glob('generated-input-*.json'))];assert len(captures)==len(queued_seeds)+(10 if args.runs else 0)
        for seed in queued_seeds:assert seed in [c['seed'] for c in captures]
        capture=next(c for c in captures if c['seed']==args.seed);assert capture['shape']['width']==(round(width*args.upscale/32)*32 if args.upscale>1 else width) and capture['shape']['height']==(round(height*args.upscale/32)*32 if args.upscale>1 else height) and capture['shape']['fps']==24
        if args.review:
            assert reviewed in capture['prompt'] and p['scenes'][0]['description']==reviewed
            result['reviewed_plan']={'displayed_seed':args.seed,'worker_seed':capture['seed'],'stale_click_did_not_submit':True,'refreshed_prompt_reached_backend':True,'direct_generate_after_review':args.repeat}
        assert capture['reference_quality']==args.reference_quality
        assert abs(p['delivery']['fps']['num']/p['delivery']['fps']['den']-args.fps)<1e-6
        assert capture['sampling']['first']['sampler']=='euler' and all(abs(a-b)<1e-7 for a,b in zip(capture['sampling']['first']['sigmas'],[1.0,0.7,0.4],strict=True))
        assert all(abs(a-b)<1e-7 for a,b in zip(capture['sampling']['second']['sampling']['sigmas'],[0.4,0.2,0.0],strict=True))
        if args.loras:
            assert capture['options']['loras']==[{'name':global_name,'strength':.75},{'name':str(scene_lora),'strength':-1.25}]
            assert p['creative']['scene_loras']['scene_qa_1']['merge_mode']=='Add To Global'
        if args.upscale>1:
            assert capture['sampling']['second']['upscale']=={'factor':args.upscale,'base_width':width,'base_height':height,'model':'/own/compatible upscaler.weights'}
        if args.components:
            assert capture['options']=={'components':{'model':own_fields[0][1],'text_encoder':own_fields[1][1],'video_vae':own_fields[2][1],'audio_vae':own_fields[3][1]},'shift_video':9.5,'shift_audio':2.25}
            assert capture['sampling']['second']['model_file']=='/own/stage2-reference-compatible.bin'
        if args.compatible_export:
            legacy=json.loads((compatible/'project.json').read_text())
            assert 'generation' not in legacy['project'] and legacy['project']['id']!=p['id']
            assert legacy['takes']==json.loads((case/'saved-project/project.json').read_text())['takes']
            active=next(x for x in p['generation']['settings'] if x['model']==p['default_model'])
            assert active in legacy['project']['scenes'][0]['settings']
            probe=ROOT/'comfyui/ui-groups-20261021/legacy-reader/target/debug/imagine-ui-legacy-reader-probe'
            original_read=json.loads(subprocess.check_output([str(probe),str(case/'saved-project')],env=env))
            compatible_read=json.loads(subprocess.check_output([str(probe),str(compatible)],env=env))
            assert not original_read['ok'] and original_read['error_code']=='invalid_manifest'
            assert compatible_read['ok'] and compatible_read['takes']==len(legacy['takes'])
            result['legacy_reader']={'original':original_read,'compatible':compatible_read}
        if args.incompatible_export:
            assert not list(incompatible.iterdir())
            assert json.loads((case/'saved-project/project.json').read_text())['takes'][-1]['provenance']['seed']==args.seed
            result['older_copy_error_preserved_original']=True
        if args.runs:
            runs=rpc('/v1/video/runs')['runs'];by_mode={r['mode']:r for r in runs}
            assert set(by_mode)=={'missing_only','all_same_seeds','all_fresh_seeds','selected_with_dependents'}
            for mode,r in by_mode.items():assert r['status']=='completed' and len(r['targets'])==(1 if mode=='selected_with_dependents' else 3)
            seeds=lambda mode:[t['seed'] for t in by_mode[mode]['targets']]
            assert seeds('missing_only')==seeds('all_same_seeds') and not set(seeds('all_fresh_seeds'))&set(seeds('missing_only'))
            assert seeds('selected_with_dependents')==seeds('all_fresh_seeds')[:1]
            result['runs']=[{'mode':mode,'status':r['status'],'targets':len(r['targets']),'seeds':seeds(mode)} for mode,r in by_mode.items()]
        result['frozen_input']=capture
        result['seed_sequence']={'queued':queued_seeds,'stored':stored_seed,'control':args.seed_control}
        if args.compatible_export and args.seed>2**53-1:raise AssertionError('Use ordinary full-u64 tests independently from the explicit older-reader export test.')
        result['passed']=True
    except Exception as error:result['error']=repr(error)
    finally:
        if args.runs and server and server.poll() is None:
            try:result['own_run_states']=rpc('/v1/video/runs')['runs']
            except Exception as error:result['own_run_state_error']=repr(error)
        for name,process in [('ui',ui),('server',server)]:
            if process:
                if process.poll() is None:process.terminate()
                result[name+'_exit']=process.wait(timeout=30)
        result['own_processes_reaped']=all(p is None or p.poll() is not None for p in [ui,server]);(case/'evidence/result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items()if k not in ['qa','frozen_input']}|{'qa_passed':result.get('qa',{}).get('passed'),'actions':result.get('qa',{}).get('actions_completed')}),flush=True)
    return not(result.get('passed')and result['own_processes_reaped'])

if __name__=='__main__':raise SystemExit(main())
