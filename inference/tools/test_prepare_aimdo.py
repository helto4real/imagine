#!/usr/bin/env python3
"""Prevent development preparation from invalidating an active ELF mapping."""
import argparse,hashlib,mmap,subprocess,sys,tempfile
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--site',type=Path,required=True);p.add_argument('--own-root',type=Path,required=True)
a=p.parse_args();a.own_root.mkdir(parents=True,exist_ok=True)
tool=Path(__file__).with_name('prepare_aimdo.py')
with tempfile.TemporaryDirectory(dir=a.own_root,prefix='aimdo-publication-')as directory:
 output=Path(directory);target=output/'aimdo.so'
 def prepare():subprocess.run([sys.executable,'-B',str(tool),'--site',str(a.site),'--output',str(output)],check=True,stdout=subprocess.DEVNULL)
 prepare();before=target.stat();digest=hashlib.sha256(target.read_bytes()).hexdigest()
 with target.open('rb')as file,mmap.mmap(file.fileno(),0,access=mmap.ACCESS_READ)as mapping:
  prepare();after=target.stat()
  assert(before.st_ino,before.st_mtime_ns)==(after.st_ino,after.st_mtime_ns)
  assert hashlib.sha256(mapping).hexdigest()==digest
 target.write_bytes(b'own sentinel, never executed')
 with target.open('rb')as file,mmap.mmap(file.fileno(),0,access=mmap.ACCESS_READ)as mapping:
  inode=target.stat().st_ino;prepare()
  assert target.stat().st_ino!=inode
  assert mapping[:]==b'own sentinel, never executed'
  assert hashlib.sha256(target.read_bytes()).hexdigest()==digest
print('AIMDO publication: unchanged inode/mtime retained; old mapping preserved on atomic refresh.')
