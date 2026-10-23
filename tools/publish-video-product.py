#!/usr/bin/env python3
"""Stage immutable server/UI/worker artifacts; optionally select them for the next start."""
import argparse,datetime,hashlib,json,os,re,shlex,shutil,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ACTIVE=ROOT/'comfyui/active-product.sh'
def digest(path):
 with path.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def selector(values):
 return ''.join(key+'='+shlex.quote(str(value))+'\n'for key,value in values.items())
def selected(data):
 return {key:shlex.split(value)[0]for line in data.decode().splitlines()if line and not line.startswith('#')for key,value in [line.split('=',1)]}
def copy_binary(source,destination):
 shutil.copy2(source,destination);destination.chmod(0o755)
PROXY_BINARIES=('st-proxy','st-proxy-launch','st-kobold-bootstrap','st-strata-logging')
COMFY_SCRIPTS=('worker.py','runtime.py','sampling_preview.py')
def bundle_comfy(source,destination):
 destination.mkdir()
 for name in COMFY_SCRIPTS:shutil.copy2(source/name,destination/name)
def bundle_proxy(source,destination,script):
 for name in PROXY_BINARIES:copy_binary(source/name,destination/name)
 shutil.copy2(script,destination/'st-stack.zsh')
 return destination/'st-proxy-launch'
def preserved_proxy(values,destination):
 result={}
 if values.get('IMAGINE_PRODUCT_PROXY_LAUNCH'):
  source=Path(values['IMAGINE_PRODUCT_PROXY_LAUNCH']).parent
  result['IMAGINE_PRODUCT_PROXY_LAUNCH']=bundle_proxy(source,destination,source/'st-stack.zsh')
 if 'IMAGINE_PRIVATE_STORAGE'in values:result['IMAGINE_PRIVATE_STORAGE']=values['IMAGINE_PRIVATE_STORAGE']
 return result
def h3_protocol_version(source_root):
 return int(re.search(r'pub const VERSION: u32 = ([0-9]+);',(source_root/'shared/h3_protocol.rs').read_text())[1])
def relocated_runtime(original,worker,directory):
 runtime=json.loads(json.dumps(original));h3=runtime['h3'];old=h3['binary'];h3['files'].pop(old);h3['binary']=str(worker);h3['files'][str(worker)]=digest(worker);h3['identity']['worker_sha256']=digest(worker)
 dependencies=directory/'h3-dependencies.json';dependencies.write_bytes(Path(h3['dependency_map']).read_bytes());h3['dependency_map']=str(dependencies)
 return runtime
def relocated_image_runtime(runtime,worker,directory):
 old=runtime['binary'];binary=directory/'imagine-worker';copy_binary(worker.resolve(),binary)
 runtime['files'].pop(old,None);runtime['binary']=str(binary);runtime['files'][str(binary)]=digest(binary)
 return runtime
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--server',type=Path);p.add_argument('--ui',type=Path);p.add_argument('--worker',type=Path);p.add_argument('--image-worker',type=Path);p.add_argument('--h3-template',type=Path);p.add_argument('--source-root',type=Path,default=ROOT);p.add_argument('--proxy-dir',type=Path);p.add_argument('--private-storage',action='store_true');mode=p.add_mutually_exclusive_group();mode.add_argument('--activate',type=Path);mode.add_argument('--rollback',type=Path);a=p.parse_args()
 if a.rollback:
  package=a.rollback.resolve();stage=ACTIVE.with_name('.active-product-'+uuid.uuid4().hex+'.new');stage.write_bytes((package/'rollback-active-product.sh').read_bytes());os.replace(stage,ACTIVE);print(package/'rollback');return
 if a.activate:
  package=a.activate.resolve();meta=json.loads((package/'package.json').read_text());prior=(package/'previous-active-product.sh').read_bytes()
  if ACTIVE.read_bytes()!=prior:raise ValueError('Start selection changed since staging; inspect the newer selection before publishing.')
  stage=ACTIVE.with_name('.active-product-'+uuid.uuid4().hex+'.new');stage.write_bytes((package/'active-product.sh').read_bytes());os.replace(stage,ACTIVE)
  meta['selected_for_next_start']=True;(package/'package.json').write_text(json.dumps(meta,indent=2)+'\n');print(package);return
 if not all([a.server,a.ui,a.worker]):p.error('--server, --ui and --worker are required for staging')
 if a.private_storage and not a.proxy_dir:p.error('--private-storage needs the matching --proxy-dir')
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8];package=ROOT/'comfyui/products'/stamp;package.mkdir(parents=True)
 previous=ACTIVE.read_bytes();values=selected(previous);base=json.loads(Path(values['IMAGINE_PRODUCT_RUNTIME']).read_text())
 (package/'previous-active-product.sh').write_bytes(previous)
 # Preserve running executable inodes and a self-contained rollback selection.
 rollback=package/'rollback';rollback.mkdir()
 for key,name in [('IMAGINE_PRODUCT_SERVER','imagine-server'),('IMAGINE_PRODUCT_UI','imagine-ui')]:copy_binary(Path(values[key]),rollback/name)
 copy_binary(Path(base['h3']['binary']),rollback/'imagine-h3-worker')
 old_runtime=relocated_runtime(base,rollback/'imagine-h3-worker',rollback)
 old_runtime=relocated_image_runtime(old_runtime,Path(base['binary']),rollback)
 (rollback/'runtime.json').write_text(json.dumps(old_runtime,indent=2)+'\n')
 old_values={'IMAGINE_PRODUCT_SERVER':rollback/'imagine-server','IMAGINE_PRODUCT_UI':rollback/'imagine-ui','IMAGINE_PRODUCT_RUNTIME':rollback/'runtime.json'}
 old_values.update(preserved_proxy(values,rollback))
 old_comfy=Path(values['IMAGINE_PRODUCT_SERVER']).parent/'comfy-worker'
 if old_comfy.is_dir():bundle_comfy(old_comfy,rollback/'comfy-worker')
 (package/'rollback-active-product.sh').write_text(selector(old_values))
 for source,name in [(a.server,'imagine-server'),(a.ui,'imagine-ui'),(a.worker,'imagine-h3-worker')]:copy_binary(source.resolve(),package/name)
 if a.h3_template:base['h3']=json.loads(a.h3_template.read_text())['h3']
 runtime=relocated_runtime(base,package/'imagine-h3-worker',package)
 if a.image_worker:runtime=relocated_image_runtime(runtime,a.image_worker,package)
 runtime['h3']['identity']['protocol']=h3_protocol_version(a.source_root)
 if 'profile_checkpoints'not in runtime['h3']:
  runtime['h3']['profile_checkpoints']=dict(re.findall(r'"(h3-[^\"]+)"[.]into\(\),\s*"([0-9a-f]{64})"[.]into\(\)',(a.source_root/'shared/h3_protocol.rs').read_text()))
 runtime['h3']['environment'].pop('IMAGINE_REFERENCE_PROBE_OUTPUT',None)
 if '/.codex-'in json.dumps(runtime):raise ValueError('Production deployment still refers to an isolated test destination')
 (package/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
 new_values={'IMAGINE_PRODUCT_SERVER':package/'imagine-server','IMAGINE_PRODUCT_UI':package/'imagine-ui','IMAGINE_PRODUCT_RUNTIME':package/'runtime.json'}
 if a.proxy_dir:new_values['IMAGINE_PRODUCT_PROXY_LAUNCH']=bundle_proxy(a.proxy_dir,package,a.source_root/'proxy/st-stack.zsh')
 if a.private_storage:
  new_values['IMAGINE_PRIVATE_STORAGE']='1'
  bundle_comfy(a.source_root/'workers/comfy',package/'comfy-worker')
 (package/'active-product.sh').write_text(selector(new_values))
 meta={'schema':1,'version':stamp,'selected_for_next_start':False,'server_sha256':digest(package/'imagine-server'),'ui_sha256':digest(package/'imagine-ui'),'worker_sha256':digest(package/'imagine-h3-worker'),'h3_protocol':runtime['h3']['identity']['protocol'],'sources':{'server':str(a.server.resolve()),'ui':str(a.ui.resolve()),'worker':str(a.worker.resolve())},'rollback_selection':str(package/'rollback-active-product.sh'),'runtime_environment_observer_enabled':False}
 if a.image_worker:meta.update(image_worker_sha256=digest(package/'imagine-worker'),image_protocol=1);meta['sources']['image_worker']=str(a.image_worker.resolve())
 meta['source_root']=str(a.source_root.resolve())
 meta['private_storage']=a.private_storage
 if a.proxy_dir:meta['proxy_sha256']={name:digest(package/name)for name in PROXY_BINARIES};meta['stack_sha256']=digest(package/'st-stack.zsh')
 if a.private_storage:meta['comfy_scripts_sha256']={name:digest(package/'comfy-worker'/name)for name in COMFY_SCRIPTS}
 (package/'package.json').write_text(json.dumps(meta,indent=2)+'\n')
 print(package)
if __name__=='__main__':main()
