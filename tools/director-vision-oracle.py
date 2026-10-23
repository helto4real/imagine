#!/usr/bin/env python3
"""Opt-in actual Director normalize/encode comparison on new procedural owned fixtures."""
import ast
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import struct
import zlib
import numpy as np
from PIL import Image, ImageOps, features

ROOT=Path(__file__).resolve().parents[1]
OWN=Path('/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e')
SOURCE=Path('/home/thhel/git/comfyui-helto-director/shared/prompt_studio.py')

def main():
    own=Path(sys.argv[1]);assert own.parent==OWN and own.resolve()==own
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(OWN)]))['filesystems'][0]
    assert mount['source']=='/dev/mapper/veracrypt1' and not os.statvfs(OWN).f_flag&os.ST_RDONLY
    own.mkdir(exist_ok=False)
    tree=ast.parse(SOURCE.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'normalize_visual','encode_visual'}]
    assert len(nodes)==2
    scope={'Image':Image,'ImageOps':ImageOps,'MAX_VISUAL_SIDE':1280,'io':io,'base64':base64}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),scope)
    cases=[]
    def pattern(w,h):return np.fromfunction(lambda y,x,c:(x*13+y*7+c*61)%256,(h,w,3),dtype=int).astype(np.uint8)
    def add(name,image,fmt='PNG',**options):
        folder=own/name;folder.mkdir();path=folder/('source.'+fmt.lower());image.save(path,format=fmt,**options);cases.append((name,path,folder))
    for w,h in [(1,1),(128,80),(1280,3),(1281,1),(2048,1025),(1001,1903),(2560,3)]:add(f'rgb_{w}x{h}',Image.fromarray(pattern(w,h)))
    rgba=np.concatenate([pattern(37,29),np.fromfunction(lambda y,x:((x+y)%2)*255,(29,37),dtype=int).astype(np.uint8)[...,None]],axis=2)
    add('rgba',Image.fromarray(rgba))
    add('palette',Image.fromarray(pattern(57,31)).convert('P',palette=Image.Palette.ADAPTIVE),transparency=0)
    add('gray8',Image.fromarray(pattern(49,35)[...,0]))
    add('gray16',Image.fromarray(np.fromfunction(lambda y,x:(x*113+y*1000)%65536,(23,41),dtype=int).astype(np.uint16)))
    # Pillow strips the low byte of multichannel 16-bit PNGs; grayscale is
    # integer-mode clipping instead. Encode these fixtures without PIL conversion.
    for color,channels in [(2,3),(4,2),(6,4)]:
        folder=own/f'png16_color_{color}';folder.mkdir();path=folder/'source.png'
        w,h=43,19
        values=np.fromfunction(lambda y,x,c:(x*113+y*1000+c*5003)%65536,(h,w,channels),dtype=int).astype('>u2')
        def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
        rows=b''.join(b'\0'+row.tobytes() for row in values)
        path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,16,color,0,0,0))+chunk(b'IDAT',zlib.compress(rows))+chunk(b'IEND',b''))
        cases.append((folder.name,path,folder))
    for orientation in range(1,9):
        exif=Image.Exif();exif[274]=orientation
        add(f'jpeg_exif_{orientation}',Image.fromarray(pattern(93,65)),'JPEG',quality=94,exif=exif)
    exif=Image.Exif();exif[274]=6
    add('jpeg_large_exif',Image.fromarray(pattern(1025,2048)),'JPEG',quality=94,exif=exif)
    add('png_exif',Image.fromarray(pattern(53,27)),exif=exif)
    add('jpeg_gray',Image.fromarray(pattern(63,43)[...,0]),'JPEG',quality=94)
    add('jpeg_cmyk',Image.fromarray(pattern(47,39)).convert('CMYK'),'JPEG',quality=94)
    add('webp_lossy',Image.fromarray(pattern(107,61)),'WEBP',quality=83)
    add('webp_alpha_lossless',Image.fromarray(rgba),'WEBP',lossless=True)
    report=[]
    for name,path,folder in cases:
        with Image.open(path) as image:normalized=scope['normalize_visual'](image)
        jpeg=base64.b64decode(scope['encode_visual'](normalized)['base64']);raw=normalized.tobytes()
        (folder/'python-rgb.raw').write_bytes(raw);(folder/'python.jpg').write_bytes(jpeg)
        result=subprocess.run([str(ROOT/'comfyui/h3-video-night-20261021/target/debug/examples/vision_probe'),str(path),str(folder)],capture_output=True,text=True)
        (folder/'probe-stderr.txt').write_text(result.stderr)
        value=json.loads(result.stdout)
        record={'name':name,'source_size':list(Image.open(path).size),'target_size':list(normalized.size),'rust':value,'passed':False}
        if result.returncode==0:
            a=np.frombuffer((folder/'rust-rgb.raw').read_bytes(),np.uint8).astype(np.float64);b=np.frombuffer(raw,np.uint8).astype(np.float64)
            same=a.size==b.size;diff=a-b if same else np.array([float('inf')]);encoded=(folder/'rust.jpg').read_bytes()
            record.update(rgb_identical=same and np.array_equal(a,b),rgb_mae=float(abs(diff).mean()),rgb_rmse=float(np.sqrt(np.mean(diff*diff))),rgb_max=float(abs(diff).max()),jpeg_identical=encoded==jpeg,jpeg_rust_bytes=len(encoded),jpeg_python_bytes=len(jpeg),python_sha256=hashlib.sha256(jpeg).hexdigest(),rust_sha256=hashlib.sha256(encoded).hexdigest())
            record['passed']=record['rgb_identical'] and record['jpeg_identical'] and [value['width'],value['height']]==list(normalized.size)
        report.append(record);print(json.dumps(record),flush=True)
    result={'passed':all(r['passed'] for r in report),'cases':report,'source':str(SOURCE),'pillow':Image.__version__,'jpeg_codec':features.version_codec('jpg'),'cpu_only':True,'actual_reference_functions':['normalize_visual','encode_visual']}
    (own/'report.json').write_text(json.dumps(result,indent=2));raise SystemExit(0 if result['passed'] else 1)

if __name__=='__main__':main()
