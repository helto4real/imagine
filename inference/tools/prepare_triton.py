#!/usr/bin/env python3
"""Validate standalone compiler inputs without advancing the Stage 0 reference."""
from reference_paths import reference_dir
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--site',type=Path,required=True)
args=parser.parse_args()
project=Path(__file__).resolve().parents[2]
lock=json.loads((project/'design/stage-1-triton-toolchain-lock.json').read_text())
source=args.root/'sources/triton-candidate'
llvm=args.root/'llvm'/lock['llvm']['distribution']

def sha(path):return hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==lock['triton']['commit']
for name,expected in lock['triton']['source_file_sha256'].items():assert sha(source/name)==expected,name
manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text())
reference={x['path']:x['sha256'] for x in manifest if x['root']=='site'}
for item in lock['triton']['installed_python_source_matches']:
    name='triton/'+item['installed']
    assert sha(args.site/name)==reference[name]==item['installed_sha256'],name
    assert sha(source/item['candidate'])==item['installed_sha256'],name
assert sha(args.site/'triton/_C/libtriton.so')==reference['triton/_C/libtriton.so']==lock['triton']['installed_libtriton_sha256']
for name,item in lock['reference_cuda_codegen_inputs'].items():assert sha(args.site/name)==item['sha256'],name
assert sha(args.site/'triton-3.7.1.dist-info/RECORD')==reference['triton-3.7.1.dist-info/RECORD']
assert sha(args.root/(lock['llvm']['distribution']+'.tar.gz'))==lock['llvm']['archive_sha256']
for name,expected in lock['llvm']['file_sha256'].items():assert sha(llvm/name)==expected,name
for name,target in lock['llvm']['symlinks'].items():assert str((llvm/name).readlink())==target,name
assert lock['llvm']['commit'] in (llvm/'include/llvm/Support/VCSRevision.h').read_text()
images=json.loads((args.root/'containers/vfs-images/images.json').read_text())
image=next(x for x in images if x['id']==lock['build_image']['id'])
assert image['digest']==lock['build_image']['manifest_digest']
layer=image['layer'];assert len(layer)==64 and all(c in '0123456789abcdef' for c in layer)
image_root=args.root/'containers/vfs/dir'/layer
assert (image_root/'bin/bash').is_file()
report={'schema':1,'result':'input hashes passed; standalone codegen equivalence still separate',
        'image_root':str(image_root),'llvm_root':str(llvm),'source_root':str(source),
        'triton_source_files':len(lock['triton']['source_file_sha256']),
        'llvm_files':len(lock['llvm']['file_sha256']),'python_files_matched':115,
        'source_lock_sha256':sha(project/'design/stage-1-triton-toolchain-lock.json')}
(args.root/'runtime/triton-input-check.json').write_text(json.dumps(report,indent=2)+'\n')
print(image_root)
