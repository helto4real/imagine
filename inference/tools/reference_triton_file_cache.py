#!/usr/bin/env python3
"""Original local FileCacheManager on newly owned, synthetic filesystem state."""
from reference_paths import reference_dir
import argparse,hashlib,itertools,json,os,sys,tempfile
from pathlib import Path

def capture(workspace):
    from triton import knobs
    from triton.runtime.cache import FileCacheManager
    cases=[];workspace.mkdir(parents=True,exist_ok=True);previous=Path.cwd()
    with tempfile.TemporaryDirectory(prefix='file-cache-oracle-',dir=workspace)as owned,knobs.cache.scope():
        os.chdir(owned)
        try:
            knobs.cache.dir='cache/main';knobs.cache.dump_dir='cache/dump';knobs.cache.override_dir='cache/override'
            for i,(override,dump)in enumerate(itertools.product([False,True],repeat=2)):
                key=f'KEY_{i}';manager=FileCacheManager(key,override=override,dump=dump)
                c=dict(key=key,override=override,dump=dump,directory=manager.cache_dir,lock=manager.lock_path,before_directory=Path(manager.cache_dir).is_dir(),steps=[])
                def step(op,*args):
                    s=dict(operation=op,args=list(args))
                    try:
                        if op=='put':out=manager.put(bytes.fromhex(args[0]),args[1])
                        elif op=='put_group':out=manager.put_group(args[0],dict(args[1]))
                        elif op=='get_file':out=manager.get_file(args[0])
                        elif op=='get_group':
                            out=manager.get_group(args[0]);out=None if out is None else [[[ord(c)for c in k],{'text':[ord(c)for c in v]}if isinstance(v,str)else{'value':v}]for k,v in out.items()]
                        elif op=='write_group':Path(manager._make_path('__grp__'+args[0])).write_bytes(bytes.fromhex(args[1]));out=None
                        elif op=='put_group_points':
                            group={''.join(map(chr,k)):''.join(map(chr,v))for k,v in args[1]};out=manager.put_group(args[0],group);s['bytes']=Path(out).read_bytes().hex()
                        elif op=='prepare_nonutf8':
                            path=manager._make_path('byte-'+chr(0xdcff))
                            with open(os.fsencode(path),'wb')as f:f.write(b'own byte filename')
                            out=None
                        elif op=='prepare_override':
                            Path(manager.cache_dir).mkdir(parents=True);Path(manager._make_path('kernel.bin')).write_bytes(b'own override bytes');out=None
                        elif op=='mkdir':Path(manager._make_path(args[0])).mkdir();out=None
                        elif op=='unlink':Path(args[0]).unlink();out=None
                        else:raise AssertionError(op)
                        s['output']=out
                    except Exception as e:s.update(error=type(e).__name__,message=str(e))
                    c['steps'].append(s)
                step('get_file','kernel.bin');step('get_group','kernel.json')
                if override and not dump:
                    step('put','616263','kernel.bin');step('put_group','kernel.json',[['kernel',manager._make_path('kernel.bin')]])
                    step('prepare_override')
                else:
                    for name,payload in [('kernel.bin',bytes(range(256))),('Svenska åäö🙂.txt','own text\n\t'.encode()),('empty',b'')]:step('put',payload.hex(),name)
                    group=[['z',manager._make_path('kernel.bin')],['å🙂',manager._make_path('Svenska åäö🙂.txt')],['missing',manager._make_path('missing')],['directory','.']]
                    step('put_group','kernel.json',group);c['group_bytes']=Path(manager._make_path('__grp__kernel.json')).read_bytes().hex()
                    step('get_group','kernel.json');step('unlink',manager._make_path('kernel.bin'));step('get_group','kernel.json')
                step('prepare_nonutf8')
                if not override or dump:
                    points=[[[0xd800],[46]],[[0xdcff],[ord(c)for c in manager._make_path('byte-'+chr(0xdcff))]],[[0x1f642],[46]]]
                    step('put_group_points','unicode.json',points);step('get_group','unicode.json')
                step('mkdir','directory');step('get_file','directory')
                invalid=[b'{',b'\xff',b'null',b'[]',b'true',b'7',b'7.5',b'"text"',b'{}',b'{"child_paths":null}',b'{"child_paths":[]}',b'{"child_paths":true}',b'{"child_paths":7}',b'{"child_paths":7.5}',b'{"child_paths":"text"}',b'{"child_paths":{"x":null}}',b'{"child_paths":{"x":1.5}}',b'{"child_paths":{"x":[]}}',b'{"child_paths":{"x":{}}}',b'{"child_paths":{"x":-1}}',b'{"child_paths":{"x":2147483648}}',b'{"child_paths":{"x":-2147483649}}',b'{"child_paths":{"x":9223372036854775808}}',b'{"child_paths":{"x":99999999999999999999999999999999999}}',b'NaN',b'Infinity',b'-Infinity',b'{"child_paths":{"x":NaN}}']
                invalid += [json.dumps(chr(0xd800)).encode(),json.dumps(chr(0xdcff)).encode(),json.dumps({'child_paths':{chr(0xd800):'.','bad_path':chr(0xd800),'byte_path':manager._make_path('byte-'+chr(0xdcff))}}).encode(),json.dumps({'child_paths':{chr(0x1f642):'.'}}).encode()]
                for raw in invalid:step('write_group','bad.json',raw.hex());step('get_group','bad.json')
                duplicate=b'{"child_paths":{"z":".","a":"missing","z":"missing","b":"."}}';step('write_group','bad.json',duplicate.hex());step('get_group','bad.json')
                step('mkdir','__grp__as_directory');step('get_group','as_directory')
                c['temporary_remains']=[p.name for p in Path(manager.cache_dir).iterdir()if p.name.startswith('tmp.pid_')];c['lock_exists']=Path(manager.lock_path).exists()if manager.lock_path is not None else False
                cases.append(c)
            knobs.cache.dir=''
            try:FileCacheManager('EMPTY')
            except Exception as e:cases.append(dict(empty_base=True,error=type(e).__name__,message=str(e)))
            else:raise AssertionError('Expected empty default cache rejection')
        finally:os.chdir(previous)
    return dict(triton_file_cache=cases)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    project=Path(__file__).resolve().parents[2];name='triton/runtime/cache.py';h=hashlib.sha256((a.site/name).read_bytes()).hexdigest();manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text());assert h==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name)
    sys.path.insert(0,str(a.site));report=capture(a.workspace);report['source_hashes']={name:h};a.output.write_text(json.dumps(report,indent=2)+'\n');print(f"{len(report['triton_file_cache'])} local-cache scenarios",flush=True)
