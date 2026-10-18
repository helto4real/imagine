#!/usr/bin/env python3
"""Preserve exact SciPy beta bodies and the needed unchanged Boost headers."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--scipy',type=Path,required=True)
p.add_argument('--boost',type=Path,required=True)
a=p.parse_args()
project=Path(__file__).resolve().parents[2]
scipy_revision='527eb7fd7953a1de068f94bf8b322f249b9405ae'
boost_revision='5e088ffe2ed0e237b9069e3a7352865283d8f196'
def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
assert git(a.scipy,'rev-parse','HEAD')==scipy_revision
assert git(a.boost,'rev-parse','HEAD')==boost_revision
assert git(a.scipy,'rev-parse','HEAD:subprojects/boost_math/math')==boost_revision
assert not git(a.scipy,'diff','--name-only') and not git(a.boost,'diff','--name-only')
destination=project/'inference/native/scipy_beta'
destination.mkdir(exist_ok=True)
header=a.scipy/'scipy/special/boost_special_functions.h'
source=header.read_text()
def extract(marker):
 start=source.index(marker);body=source.index('{',start);depth=1;end=body+1
 while depth:
  if source[end]=='{':depth+=1
  elif source[end]=='}':depth-=1
  end+=1
 return source[start:end]+'\n'
markers=['template <class RealType>\nRealType\nboost::math::policies::user_evaluation_error',
         'template <class RealType>\nRealType\nboost::math::policies::user_overflow_error',
         'template<typename Real, typename Policy>\nstatic inline\nReal ibeta_inv_wrap',
         'template<typename Real>\nReal beta_ppf_wrap']
bodies=[extract(marker)for marker in markers]
original='// Selected bodies copied verbatim from SciPy 1.17.1. See SCIPY-LICENSE.txt.\n'+ '\n'.join(bodies)
(destination/'original.hpp').write_text(original)
shutil.copyfile(header,destination/'boost_special_functions.h.source')
shutil.copyfile(a.scipy/'scipy/special/sf_error.cc',destination/'sf_error.cc.source')
shutil.copyfile(a.scipy/'LICENSE.txt',destination/'SCIPY-LICENSE.txt')
shutil.copyfile(a.boost/'LICENSE',destination/'BOOST-LICENSE.txt')
adapter=project/'inference/native/scipy_beta.cpp'
deps=subprocess.check_output(['clang++','-std=c++20','-DBOOST_MATH_STANDALONE=1','-M',str(adapter),'-I'+str(a.boost/'include'),'-I'+str(destination.parent)],text=True)
paths=shlex.split(deps.replace('\\\n',' '))[1:]
headers=[]
for value in paths:
 path=Path(value)
 if path.is_relative_to(a.boost/'include'):
  relative=path.relative_to(a.boost/'include');target=destination/relative
  target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
  headers.append(str(relative))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
lock=dict(schema=1,scipy=dict(repository='https://github.com/scipy/scipy',revision=scipy_revision,version='1.17.1',beta_header_sha256=digest(header)),
 boost=dict(repository='https://github.com/boostorg/math',revision=boost_revision,headers=sorted(set(headers))),
 transformation='Four SciPy template bodies unchanged; Python error calls mapped to owned C ABI diagnostics; original ignored special-function statuses retained. Boost dependency headers unchanged; no Python linkage.',
 sources={str(path.relative_to(destination)):digest(path)for path in sorted(destination.rglob('*'))if path.is_file()},
 extracted_bodies_sha256=[hashlib.sha256(body.encode()).hexdigest()for body in bodies])
(project/'inference/resources/scipy-beta-source-lock.json').write_text(json.dumps(lock,indent=2)+'\n')
print(f'Preserved four verbatim SciPy bodies and {len(set(headers))} original Boost headers')
