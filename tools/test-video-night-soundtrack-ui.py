#!/usr/bin/env python3
"""Own native Final mix/Both acceptance; CPU montage, no GPU or runtime-log reads."""
import argparse,json,os,re,shutil,socket,subprocess,time,urllib.request
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OWN=Path('/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e')
def main():
 p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('usage',choices=['both','final_mix']);a=p.parse_args();assert re.fullmatch('q[0-9]+',a.case)
 mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(OWN)]))['filesystems'][0];assert mount['source']=='/dev/mapper/veracrypt1'and not os.statvfs(OWN).f_flag&os.ST_RDONLY
 case=OWN/a.case;case.mkdir()
 for name in ['evidence','state','server-project','output','cache','config','share','models','tmp']:(case/name).mkdir()
 shutil.copytree(OWN/'reference-e2e-07/saved-project',case/'saved-project')
 package=Path((OWN/'evidence/checkpoint26-product-build-02.log').read_text().splitlines()[-1])
 runtime=json.loads((OWN/'reference-e2e-07/standard/runtime.json').read_text());assert 'IMAGINE_REFERENCE_PROBE_OUTPUT'not in runtime['h3']['environment'];(case/'runtime.json').write_text(json.dumps(runtime))
 destination=json.loads((OWN/'reference-e2e-07/standard/state/video-output.json').read_text());destination['root']=str(case/'output');(case/'state/video-output.json').write_text(json.dumps(destination))
 with socket.socket()as available:available.bind(('127.0.0.1',0));port=available.getsockname()[1]
 api=f'http://127.0.0.1:{port}'
 env=dict(os.environ,TMPDIR=str(case/'tmp'),CUDA_VISIBLE_DEVICES='',LIBGL_ALWAYS_SOFTWARE='1',XDG_CACHE_HOME=str(case/'cache'),XDG_CONFIG_HOME=str(case/'config'),XDG_DATA_HOME=str(case/'share'),QWEN_MODEL_ROOT=str(case/'models'),PYTHONDONTWRITEBYTECODE='1')
 server=ui=None;result={'case':str(case),'usage':a.usage,'gpu':False,'api':api,'package':str(package),'runtime_logs_read':False}
 try:
  with(case/'evidence/server.log').open('x')as f:server=subprocess.Popen([str(package/'imagine-server'),'--project',str(case/'server-project'),'--data',str(case/'state'),'--profiles',str(ROOT/'config/image/profiles'),'--bind',f'127.0.0.1:{port}','--backend','rust','--rust-runtime',str(case/'runtime.json')],env=env,stdout=f,stderr=subprocess.STDOUT)
  deadline=time.monotonic()+30
  while True:
   assert server.poll()is None and time.monotonic()<deadline
   try:
    with urllib.request.urlopen(api+'/v1/session',timeout=1)as response:session=json.load(response)['id'];break
   except OSError:time.sleep(.05)
  prefs=case/'ui/.imagine';prefs.mkdir(parents=True);(prefs/'ui-settings.json').write_text(json.dumps({'output_folder':str(case/'output'),'privacy_mode':False}))
  document=json.loads((case/'saved-project/project.json').read_text());asset=next(k for k,v in document['project']['assets'].items()if v['kind']=='video')
  click=lambda name:{'action':'click_control','name':name}
  actions=[{'action':'wait_idle'},{'action':'open','folder':str(case/'saved-project')},{'action':'wait_idle'},click('section-own_reference_section'),click('video-soundtrack-'+asset),click('video-soundtrack-'+asset+'-'+a.usage),{'action':'wait_idle'},click('take-0'),{'action':'wait_idle'},click('take-accept'),{'action':'wait_idle'},{'action':'expect','accepted_takes':1},click('assembly-open'),click('assembly-audio-timeline'),click('assembly-normalize'),click('assembly-freeze'),{'action':'wait_idle'},{'action':'expect','assembly_frames':124,'assembly_error':False},{'action':'scroll_pixels','x':450,'y':350,'delta_y':-1200},{'action':'wait_frames','frames':5},{'action':'screenshot','file':'soundtrack-frozen.png'},click('assembly-render'),{'action':'wait_status','status':'succeeded','count':1},{'action':'wait_idle'},{'action':'expect','deliveries':1},{'action':'scroll_pixels','x':450,'y':350,'delta_y':-1200},{'action':'wait_frames','frames':5},click('assembly-delivery-0'),{'action':'wait_idle'},{'action':'expect','preview':True},click('timeline-play'),{'action':'wait_time','milliseconds':500},{'action':'expect','playing':True,'audio_running':True},{'action':'screenshot','file':'actual-soundtrack-delivery-playing.png'},click('timeline-play'),{'action':'expect','playing':False,'audio_running':False},{'action':'save'},{'action':'wait_idle'},{'action':'quit'}]
  plan=case/'qa-plan.json';plan.write_text(json.dumps({'destination':dict(destination,root=str(case)),'actions':actions},indent=2))
  with(case/'evidence/ui.log').open('x')as f:ui=subprocess.Popen([str(package/'imagine-ui'),'--project',str(case/'ui'),'--api',api,'--title','Imagine Video Night own soundtrack mix '+a.case,'--qa-script',str(plan)],env=env,stdout=f,stderr=subprocess.STDOUT)
  result.update(server_pid=server.pid,ui_pid=ui.pid);print(json.dumps(result),flush=True)
  result['ui_exit']=ui.wait(timeout=120);result['qa']=json.loads((case/'native-qa-result.json').read_text());assert result['qa']['passed']
  doc=json.loads((case/'saved-project/project.json').read_text());delivery=doc['deliveries'][0];result['plan']=delivery['plan'];assert result['plan']['options']['audio_mode']=='timeline'and not result['plan']['options']['normalize_mix']
  assert len(result['plan']['audio'])==1 and result['plan']['audio'][0]['source_in']=={'num':1,'den':10}
  output=next((case/'output').glob('*/*.mp4'));probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(output)]));video=next(s for s in probe['streams']if s['codec_type']=='video');audio=next(s for s in probe['streams']if s['codec_type']=='audio')
  assert video['codec_name']=='h264'and video['nb_read_frames']=='124'and audio['codec_name']=='aac'and audio['sample_rate']=='44100'and abs(float(probe['format']['duration'])-31/6)<.002
  pcm=np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-i',str(output),'-map','0:a:0','-f','f32le','pipe:1']),np.float32)
  result.update(output=str(output),probe=probe,pcm_rms=float(np.sqrt(np.mean(pcm.astype(np.float64)**2))),pcm_peak=float(np.max(abs(pcm))))
  assert result['pcm_rms']>.001;result['passed']=True
 except Exception as error:result['error']=repr(error)
 finally:
  for name,process in [('ui',ui),('server',server)]:
   if process:
    if process.poll()is None:process.terminate()
    result[name+'_exit']=process.wait(timeout=30)
  result['own_processes_reaped']=all(p is None or p.poll()is not None for p in [ui,server]);(case/'evidence/result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items()if k not in ['plan','probe']}),flush=True)
 return not(result.get('passed')and result['own_processes_reaped'])
if __name__=='__main__':raise SystemExit(main())
