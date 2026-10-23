#!/usr/bin/env python3
"""Stop BEFORE clearing/deallocating a source set with an overcounted used field.
Own tiny metadata only. Child exits through os._exit, so the unsafe C cleanup
is not invoked. This is a diagnostic; undefined cleanup outcomes are not an
exact parity oracle.
"""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--child',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
assert sys.version_info[:3]==(3,12,13)
if a.child:
 import resource
 resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(512*1024*1024,512*1024*1024))
 status=1
 try:
  active=False;done=False
  class Key:
   def __init__(self,index):self.index=index
   def __hash__(self):return 0
   def __eq__(self,other):
    global done
    if active and not done and other.index==5:done=True;s.add(keys[6])
    return self.index==other.index
  keys=[Key(i)for i in range(7)];s=set(keys[:4]);s.discard(keys[0]);active=True;s.add(keys[5])
  order=[v.index for v in s];logical=len(s);capacity=(s.__sizeof__()-set().__sizeof__())//16 or 8
  assert logical==5 and order==[5,1,2,3] and capacity==8
  print(json.dumps(dict(logical_used=logical,active_entries=len(order),order=order,capacity=capacity,smalltable=True,clear_loop_remaining_after_all_valid_entries=logical-len(order),cleanup_executed=False,exit='os._exit; no Python object finalization')),flush=True);status=0
 finally:os._exit(status)
if a.output is None:p.error('--output required')
runs=[]
for _ in range(3):
 child=subprocess.run([sys.executable,'-S','-B',__file__,'--child'],capture_output=True,text=True,timeout=10)
 assert child.returncode==0,(child.returncode,child.stderr)
 runs.append(json.loads(child.stdout))
source=Path(__file__).resolve().parents[2]/'comfyui/stage3-core-20261020/sources/cpython/Objects/setobject.c'
a.output.write_text(json.dumps(dict(schema=1,python=sys.version,trained_model_weights=0,cuda_work=False,source=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()),runs=runs,conclusion='Source set_clear_internal and set_dealloc loop until used active entries, without a mask bound. Each observed smalltable has used=5 but only 4 active entries; the next read after its valid table is out of bounds. Cleanup was not executed.'),indent=2)+'\n');print('Verified overcount in three fresh processes; unsafe cleanup was not executed')
