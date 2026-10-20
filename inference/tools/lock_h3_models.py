#!/usr/bin/env python3
"""Full checksums of exactly the four approved H3 files; no tensor allocation."""
import argparse,hashlib,json,time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
project=Path(__file__).resolve().parents[2]
accepted=project/'design/stage-3-evidence/native-av-selected-headers-20261020.json'
previous=json.loads(accepted.read_text())
assert not a.output.exists(),'Retain earlier identity evidence; use a fresh output'
report=dict(schema=1,scope='full identity of exactly four selected H3 checkpoints',trained_weight_tensors_allocated=0,gpu_work=False,header_evidence_sha256=hashlib.sha256(accepted.read_bytes()).hexdigest(),components=[],full_checkpoint_bytes_read=0,result='running')
def write():a.output.write_text(json.dumps(report,indent=2)+'\n')
write()
for entry in previous['components']:
 path=Path(entry['path']);before=path.stat()
 assert before.st_size==entry['file_bytes']and before.st_mtime_ns==entry['mtime_ns'],path
 with path.open('rb')as file:
  n=int.from_bytes(file.read(8),'little');assert n==entry['header_bytes']and n<2**20
  header=file.read(n);assert hashlib.sha256(header).hexdigest()==entry['header_sha256']
  file.seek(0);start=time.monotonic();digest=hashlib.file_digest(file,'sha256').hexdigest();elapsed=time.monotonic()-start
 after=path.stat();assert(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),path
 report['components'].append(dict(role=entry['role'],path=str(path),file_bytes=before.st_size,mtime_ns=before.st_mtime_ns,header_sha256=entry['header_sha256'],sha256=digest,checksum_seconds=elapsed))
 report['full_checkpoint_bytes_read']+=before.st_size;write();print('SHA locked',entry['role'],before.st_size,'bytes',flush=True)
report['result']='passed';write()
