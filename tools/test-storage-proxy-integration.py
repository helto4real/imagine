#!/usr/bin/env python3
"""Packaged proxy to source-owned private API; CPU, no live user data or log reads."""
import argparse
import hashlib
import http.server
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
C=ROOT/'comfyui/ui-groups-20261021'
OWN=Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')

class Backend(http.server.BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_GET(self):
        value={'version':'own-cpu-fixture'} if self.path=='/api/version' else {'models':[]}
        raw=json.dumps(value).encode();self.send_response(200)
        self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)))
        self.end_headers();self.wfile.write(raw)
    def do_POST(self):
        self.rfile.read(int(self.headers.get('Content-Length','0')))
        self.do_GET()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package',type=Path);parser.add_argument('case')
    args=parser.parse_args();package=args.package.resolve()
    assert package.parent==ROOT/'comfyui/products'
    assert args.case.isascii() and args.case.isalnum() and args.case.startswith('sp44')
    case=OWN/args.case;case.mkdir()
    for name in ['home','models','r','input','unused-cache','technical']:(case/name).mkdir()
    env={k:v for k,v in os.environ.items() if not k.startswith(('ST_PROXY_','ST_STACK_'))}
    env.update(HOME=str(case/'home'),TMPDIR=str(case/'tmp'),XDG_RUNTIME_DIR=str(case/'r'),
               XDG_CACHE_HOME=str(case/'technical'),XDG_CONFIG_HOME=str(case/'home/.config'),
               CUDA_VISIBLE_DEVICES='',QWEN_MODEL_ROOT=str(case/'models'),PYTHONDONTWRITEBYTECODE='1')
    report=dict(passed=False,package=str(package),gpu=False,model_jobs=0,synthetic_CPU_jobs=0,
                runtime_logs_read=False,stdout_stderr='DEVNULL; no persistent runtime log',
                actual_production_volume=False,volume='explicit own encrypted-home fixture policy; actual volume proven separately in CP43',
                coordinator='owned synthetic fixture, no GPU lease')
    processes=[];backend=None;thread=None
    before=(ROOT/'comfyui/active-product.sh').read_bytes()
    token='own-synthetic-mcp-token'
    env['ST_PROXY_MCP_TOKEN']=token
    def request(url,body=None,headers=None):
        h={'Content-Type':'application/json',**(headers or {})}
        req=urllib.request.Request(url,data=None if body is None else json.dumps(body).encode(),headers=h)
        with urllib.request.urlopen(req,timeout=30) as response:
            raw=response.read()
            if response.headers.get_content_type()=='text/event-stream':
                return next(json.loads(line[6:]) for line in raw.decode().splitlines()
                            if line.startswith('data: ') and json.loads(line[6:]).get('id')==1)
            return json.loads(raw)
    try:
        fixture=C/'target/debug/examples/timeline_fixture_server'
        server=subprocess.Popen([str(fixture),str(case),'--private-storage'],env=env,
                                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        processes.append(server);report['server_pid']=server.pid
        endpoint=case/'evidence/endpoint.json';deadline=time.monotonic()+30
        while not endpoint.exists():
            assert server.poll() is None and time.monotonic()<deadline,'Own CPU fixture startup failed; no logs read'
            time.sleep(.05)
        api=json.loads(endpoint.read_text())['api']
        connection=request(api+'/v1/session');assert connection['encrypted']
        session={'X-Imagine-Session':connection['id']}
        backend=http.server.ThreadingHTTPServer(('127.0.0.1',0),Backend)
        thread=threading.Thread(target=backend.serve_forever,daemon=True);thread.start()
        with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        proxy=subprocess.Popen([str(package/'st-proxy-launch'),'--proxy-only','--llm-backend','ollama',
            '--llm-url',f'http://127.0.0.1:{backend.server_port}','--comfy-enabled','false',
            '--chat-port','0','--image-port','0','--mcp-port',str(port),
            '--mcp-input-dir',str(case/'input'),
            '--mcp-output-dir',str(case/'unused-cache/obsolete'),
            '--mcp-workflows-dir',str(C/'source/proxy/workflows'),'--model-profiles','',
            '--imagine-url',api,'--coordinator-socket',str(case/'r/gpu.sock'),'--poll-interval','0.05'],
            env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        processes.append(proxy);report['proxy_pid']=proxy.pid
        mcp=f'http://127.0.0.1:{port}/mcp'
        headers={'Accept':'application/json, text/event-stream','Authorization':'Bearer '+token}
        def rpc(name,arguments):
            reply=request(mcp,{'jsonrpc':'2.0','id':1,'method':'tools/call','params':{'name':name,'arguments':arguments}},headers)
            assert 'error' not in reply, 'Own MCP protocol failed'
            return reply['result']
        deadline=time.monotonic()+30
        while True:
            assert proxy.poll() is None,'Own packaged proxy startup failed; no logs read'
            try:
                result=rpc('list_profiles',{});assert not result.get('isError');break
            except (OSError,TimeoutError):
                assert time.monotonic()<deadline;time.sleep(.05)
        try:request(mcp,{'jsonrpc':'2.0','id':1,'method':'ping'}, {'Accept':'application/json, text/event-stream'})
        except urllib.error.HTTPError as error:assert error.code in (401,403)
        else:raise AssertionError('MCP token not enforced')
        report['mcp_auth_preserved']=True
        # A procedural source with EXIF6 distinguishes original path from RAM normalization.
        from PIL import Image
        original=case/'input/own-original.jpg';image=Image.new('RGB',(96,64),(90,130,170))
        exif=image.getexif();exif[274]=6;image.save(original,format='JPEG',exif=exif)
        digest=hashlib.sha256(original.read_bytes()).hexdigest();inode=original.stat().st_ino
        files_before={p for p in case.rglob('*') if p.is_file()}
        registered=rpc('register_image',{'profile':'qwen','path':str(original)})
        assert not registered.get('isError'), 'Own original registration failed'
        reference=registered['structuredContent']
        assert reference['preview_path']==str(original) and reference['width']==64 and reference['height']==96
        assert {p for p in case.rglob('*') if p.is_file()}==files_before,'Reference registration created disk copy/preview'
        report['original_path_and_RAM_EXIF_preview']=True
        generated=rpc('generate',{'job_id':'own-storage44','profile':'qwen',
            'parameters':{'prompt':'Own procedural storage test, never a user prompt','negative_prompt':'','max_side':256,
                          'sampling':{'steps':1},'seed':42},'references':[reference['image_id']]})
        if generated.get('isError'):
            error=generated.get('structuredContent',{}).get('error',{})
            report['own_error_category']=error.get('category')
            # Only stable known source codes; no raw diagnostic/log persistence.
            report['own_known_error_codes']=[code for code in ['prompt_decryption_failed','unsupported_sampling','invalid_request','image_generation_failed','coordinator_unavailable'] if code in str(error)]
        assert not generated.get('isError'), f'Own synthetic CPU request failed in RAM: {generated.get("structuredContent",{}).get("error",{})}'
        value=generated['structuredContent'];assert value['status']=='succeeded'
        report['synthetic_CPU_jobs']=1
        output=value['images'][0];saved=Path(output['path'])
        assert saved.parent==case/'private/projects/own-image-workspace/outputs/images'
        assert output['storage']=='saved' and saved.is_file()
        saved_hash=hashlib.sha256(saved.read_bytes()).hexdigest()
        acknowledgement={k:output[k] for k in ['image_id','bytes','sha256']}
        acknowledgement['saved_path']=str(saved)
        first=rpc('confirm_saved',{'job_id':'own-storage44','images':[acknowledgement]})
        assert not first.get('isError')
        assert rpc('confirm_saved',{'job_id':'own-storage44','images':[acknowledgement]})==first
        assert hashlib.sha256(saved.read_bytes()).hexdigest()==saved_hash
        assert original.stat().st_ino==inode and hashlib.sha256(original.read_bytes()).hexdigest()==digest
        assert not (case/'unused-cache/obsolete').exists()
        new_files={p for p in case.rglob('*') if p.is_file()}-files_before
        assert new_files=={saved},'Private output created unexpected temporary/cache files'
        report.update(original_unchanged=True,original_copied=False,preview_written_to_disk=False,
                      direct_private_publication=True,no_mcp_delivery_copy=True,idempotent_ack_non_destructive=True,
                      no_private_fallback_files=True,output=str(saved))
        # Broken originals fail in RAM; no private error/template dump appears.
        invalid=rpc('register_image',{'profile':'qwen','path':str(case/'input/own-missing-private-marker.jpg')})
        assert invalid.get('isError')
        assert {p for p in case.rglob('*') if p.is_file()}-files_before=={saved}
        proxy.send_signal(signal.SIGTERM);assert proxy.wait(timeout=15)==0
        assert saved.is_file() and original.is_file()
        report['outputs_survive_proxy_close']=True
        server.send_signal(signal.SIGINT);assert server.wait(timeout=15)==0
        report['passed']=True
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
        if backend:backend.shutdown();backend.server_close()
        if thread:thread.join(timeout=5)
        report['own_processes_reaped']=all(p.poll() is not None for p in processes)
        report['active_selection_unchanged']=(ROOT/'comfyui/active-product.sh').read_bytes()==before
        assert report['active_selection_unchanged']
        (case/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report))

if __name__=='__main__':main()
