#!/usr/bin/env python3
"""Run reference stack behavior expectations with Rust launcher/helper identities.
The fixture owns every service, HOME and process group; no real services run.
"""
import argparse,pathlib,shutil,sys,tempfile
p=argparse.ArgumentParser();p.add_argument('--reference',type=pathlib.Path,required=True);p.add_argument('-k',default='');a=p.parse_args()
sys.dont_write_bytecode=True;sys.path[:0]=[str(a.reference/'src'),str(a.reference)]
import pytest
ROOT=pathlib.Path(__file__).resolve().parents[2]
class Paths:
 def pytest_collection_modifyitems(self,items):
  modules={item.module for item in items}
  for module in modules:
   if not hasattr(module,'FakeStack'):continue
   module.STACK_SCRIPT=ROOT/'proxy/st-stack.zsh'
   module.REPO_ROOT=a.reference
   original=module.FakeStack.__init__
   def init(self,root,original=original):
    original(self,root)
    binary=root/'target/release';binary.mkdir(parents=True)
    shutil.copy2(self.repo/'.venv/bin/st-vram-proxy',binary/'st-proxy')
    for name in ('st-kobold-bootstrap','st-strata-logging'):(binary/name).symlink_to(ROOT/'target/debug'/name)
   module.FakeStack.__init__=init
with tempfile.TemporaryDirectory(prefix='st-proxy-stack-tests-') as tmp:
 # The only assertion adaptation is the process executable identity: the Rust
 # filter replaces `python -m st_proxy.strata_logging`. Count/cleanup expectations
 # remain unchanged, and the reference file is never modified.
 source=(a.reference/'tests/test_stack_script.py').read_text()
 assert source.count('b"st_proxy.strata_logging"')==1
 source=source.replace('b"st_proxy.strata_logging"','b"st-strata-logging"')
 test=pathlib.Path(tmp)/'test_stack_script.py';test.write_text(source)
 sys.exit(pytest.main([str(test),'-q','-p','no:cacheprovider','--basetemp',str(pathlib.Path(tmp)/'fixtures'),'-k',a.k],plugins=[Paths()]))
