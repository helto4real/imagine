#!/usr/bin/env python3
"""Check the preserved original beta sources without importing numerical code."""
import hashlib,json
from pathlib import Path
project=Path(__file__).resolve().parents[2]
root=project/'inference/native/scipy_beta'
lock=json.loads((project/'inference/resources/scipy-beta-source-lock.json').read_text())
for name,digest in lock['sources'].items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
assert {str(p.relative_to(root))for p in root.rglob('*')if p.is_file()}==set(lock['sources'])
source=(root/'boost_special_functions.h.source').read_text()
original=(root/'original.hpp').read_text()
markers=['template <class RealType>\nRealType\nboost::math::policies::user_evaluation_error','template <class RealType>\nRealType\nboost::math::policies::user_overflow_error','template<typename Real, typename Policy>\nstatic inline\nReal ibeta_inv_wrap','template<typename Real>\nReal beta_ppf_wrap']
bodies=[]
for marker in markers:
 start=source.index(marker);end=source.index('{',start)+1;depth=1
 while depth:
  if source[end]=='{':depth+=1
  elif source[end]=='}':depth-=1
  end+=1
 bodies.append(source[start:end]+'\n')
assert [hashlib.sha256(body.encode()).hexdigest()for body in bodies]==lock['extracted_bodies_sha256']
assert original=='// Selected bodies copied verbatim from SciPy 1.17.1. See SCIPY-LICENSE.txt.\n'+'\n'.join(bodies)
print(f'Verified four unchanged SciPy template bodies and {len(lock["boost"]["headers"])} unchanged Boost headers')
