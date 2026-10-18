#!/usr/bin/env python3
"""Lock approved downloaded source identities; never imports downloaded code."""
import hashlib
import json
import subprocess
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
ROOT=Path('/tmp/imagine-rust-comfy-parity-20261017/sources')
REVISIONS={
    'comfy-kitchen':'be003b7c23c5b01328657955b8bc5d3f073d868e',
    'comfy-aimdo':'08b37eec31319b7f245afdc009e34a9861565c25',
    'cutlass':'d4b4b494c3c51bf6507e7ab09fbafd1e9fa94f39',
    'flash-attention':'979702c87a8713a8e0a5e9fee122b90d2ef13be5',
    'pytorch':'cf30153c4c131c8164ee7798e5022d810682e2cb',
    'tch-rs':'4227b89b72059b0651ff83a38637693574e0a2d6',
    'funchook':'88388db3c69e16c1560fee65c6857d75f5ce6fd5',
    'distorm':'ab59d6e193948cfa5d1482fb6c7e64870e9e93b9',
}
EXTENSIONS={'.c','.cpp','.h','.hpp','.cu','.cuh','.py','.rs','.cmake','.toml','.in','.yml','.yaml','.txt','.json'}


def git(repo,*args):return subprocess.check_output(['git','-C',str(repo),*args])


def capture():
    records={}
    for name,revision in REVISIONS.items():
        repo=ROOT/name
        assert git(repo,'rev-parse','HEAD').decode().strip()==revision,name
        tree=git(repo,'ls-tree','-rz','HEAD')
        blobs={};links={};source={}
        for entry in tree.split(b'\0'):
            if not entry:continue
            metadata,path=entry.split(b'\t',1);mode,kind,oid=metadata.decode().split();rel=path.decode()
            if kind=='commit':links[rel]=oid;continue
            blobs[rel]={'mode':mode,'git_blob':oid}
            p=repo/rel
            if mode=='100644' or mode=='100755':
                if p.suffix in EXTENSIONS or p.name in {'CMakeLists.txt','Makefile','.gitmodules','LICENSE'}:
                    data=p.read_bytes()
                    assert hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()==oid,(name,rel)
                    source[rel]=hashlib.sha256(data).hexdigest()
        records[name]={'revision':revision,'tree':git(repo,'rev-parse','HEAD^{tree}').decode().strip(),
                       'git_tree_listing_sha256':hashlib.sha256(tree).hexdigest(),
                       'git_blobs':blobs,'gitlinks':links,'source_file_sha256':source,
                       'source_root':str(repo),'rebuild_status':'not built; source identity only'}
        print(name,len(source),'source file hashes;',len(links),'submodule pins',flush=True)
    # Installed headers contain documented install-time wrappers. Compare the
    # exact transformation, not a permissive textual or numerical tolerance.
    include=Path('/home/thhel/git/ComfyUI/.venv/lib/python3.12/site-packages/torch/include')
    paths=['c10/macros/Macros.h','ATen/core/TensorBase.h','ATen/ATen.h','c10/core/TensorImpl.h','torch/csrc/api/include/torch/torch.h']
    comparisons=[]
    for rel in paths:
        src=ROOT/'pytorch'/('aten/src/'+rel if rel.startswith('ATen/') else rel)
        data=src.read_text()
        wrapped=('#if !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)\n'+data+'\n#else\n'
                 '#error "This file should not be included when either TORCH_STABLE_ONLY or TORCH_TARGET_VERSION is defined."\n'
                 '#endif  // !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)\n')
        assert (include/rel).read_text()==wrapped,rel
        comparisons.append({'installed_header':rel,'exact_install_wrapper_match':True})
    obj={'repos':records,'torch_header_comparisons':comparisons,
         'torch_header_transform_evidence':['pytorch/tools/wrap_headers.py','pytorch/cmake/PostBuildSteps.cmake'],
         'policy':'Original local source/binaries remain the oracle. Downloaded sources are not rebuilt replacements. Gitlink pins enumerate unfetched nested dependencies; no claim of a recursive LibTorch source-build archive.'}
    (BASE/'source-acquisition-lock.json').write_text(json.dumps(obj,indent=2)+'\n')


if __name__=='__main__':capture()
