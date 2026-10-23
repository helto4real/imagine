#!/usr/bin/env python3
"""Opt-in own CPU fixtures versus actual Director PyAV functions; no live modules."""
import argparse,ast,json,math,os,struct,subprocess,sys,time
from pathlib import Path

OWN=Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')
DIRECTOR=Path('/home/thhel/git/comfyui-helto-director/shared/media_cache.py')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('case');parser.add_argument('--binary',type=Path,required=True);args=parser.parse_args()
    assert args.case.isalnum() and args.binary.resolve().is_relative_to(Path('/home/thhel/ChatGPT/imagine/comfyui/ui-groups-20261021'))
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(OWN)]))['filesystems'][0]
    assert (mount['target'],mount['source'],mount['fstype'])==('/home','/dev/mapper/cryptroot[/@home]','btrfs') and not os.statvfs(OWN).f_flag&os.ST_RDONLY
    # Import only the already installed test dependency once access is approved.
    import av
    from typing import Any
    module=ast.parse(DIRECTOR.read_text())
    functions=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name in ['_decode_audio_waveform','_bin_peaks']]
    assert len(functions)==2
    namespace={'av':av,'math':math,'Path':Path,'Any':Any}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(DIRECTOR),'exec'),namespace)
    case=OWN/args.case;case.mkdir();(case/'tmp').mkdir()
    env=dict(os.environ,TMPDIR=str(case/'tmp'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
    samples=9973;rate=8000;fixtures=[]
    for name,bits,format_code,mime in [('float',32,3,'audio/wav'),('signed',16,1,'audio/wav'),('unsigned',8,1,'audio/wav')]:
        pcm=bytearray()
        for i in range(samples):
            gain=(.2+.8*i/samples)
            for channel in range(2):
                v=math.sin(i*(.17+.013*channel))*gain
                if format_code==3:pcm.extend(struct.pack('<f',v*.02))
                elif bits==16:pcm.extend(struct.pack('<h',-32768 if i%211==0 else round(v*24000)))
                else:pcm.extend(struct.pack('B',min(255,max(0,round(128+v*127)))))
        width=2*bits//8;header=b'RIFF'+struct.pack('<I',36+len(pcm))+b'WAVEfmt '+struct.pack('<IHHIIHH',16,format_code,2,rate,rate*width,width,bits)+b'data'+struct.pack('<I',len(pcm))
        path=case/(name+'.wav');path.write_bytes(header+pcm);fixtures.append((path,mime))
    for source,name,codec,mime in [(fixtures[0][0],'planar-aac.m4a','aac','video/mp4'),(fixtures[1][0],'signed.flac','flac','audio/flac'),(fixtures[1][0],'signed.mp3','libmp3lame','audio/mpeg'),(fixtures[0][0],'planar-opus.ogg','libopus','audio/ogg')]:
        path=case/name
        subprocess.run(['/usr/bin/ffmpeg','-nostdin','-v','error','-i',str(source),'-map','0:a:0','-c:a',codec,str(path)],env=env,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        fixtures.append((path,mime))
    requests=[[str(path),mime,count]for path,mime in fixtures for count in [16,96,512]]
    started=time.monotonic()
    process=subprocess.run([str(args.binary)],input=json.dumps(requests),env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,check=True,timeout=120)
    actual=json.loads(process.stdout);rows=[]
    for result in actual:
        assert 'error' not in result,result
        expected=namespace['_decode_audio_waveform'](Path(result['path']),result['count']);observed=result['waveform'];a=expected['peaks'];b=observed['peaks'];assert len(a)==len(b)
        differences=[abs(x-y)for x,y in zip(a,b)]
        row={'file':Path(result['path']).name,'count':result['count'],'max':max(differences,default=0),'mae':sum(differences)/len(differences),'identical_peaks':a==b,'sample_rate_equal':expected['sample_rate']==observed['sample_rate'],'channels_equal':expected['channels']==observed['channels'],'duration_error':abs(expected['duration_seconds']-observed['duration_seconds'])}
        rows.append(row)
    report={'source':'Actual Director _decode_audio_waveform/_bin_peaks isolated AST, existing PyAV; native waveform::decode','cases':rows,'seconds':time.monotonic()-started,'gpu':False,'runtime_logs_read':False,'passed':all(r['max']<=.000100000001 and r['sample_rate_equal'] and r['channels_equal'] and r['duration_error']<1e-5 for r in rows),'criterion':'One 4-decimal peak quantum maximum; exact rates/channels; duration within10microseconds. Report exact identity separately.'}
    (case/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));return not report['passed']

if __name__=='__main__':raise SystemExit(main())
