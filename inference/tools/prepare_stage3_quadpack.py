#!/usr/bin/env python3
"""Extract unchanged numerical QUADPACK bodies, without its Python header."""
import argparse
import hashlib
import json
from pathlib import Path
import re

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
source=a.source.read_text()
def extract(name):
    start=re.search(r'(?:static void|void)\s*\n'+name+r'\(',source).start()
    body=source.index('{',start)
    # C comments contain braces; scan lexical tokens rather than counting them.
    depth=0
    for token in re.finditer(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|[{}]',source[body:],re.S):
        if token.group()=='{':depth+=1
        elif token.group()=='}':
            depth-=1
            if depth==0:return source[start:body+token.end()]
    raise AssertionError(name)
names=['dqagse','dqelg','dqk21','dqpsrt']
license=source[:source.index('/*',2)]
constants=source[source.index('// Constants'):source.index('// Exported functions')]
decls='''#include <math.h>
static void dqelg(int*,double*,double*,double*,double*,int*);
static void dqk21(double(*)(double*),const double,const double,double*,double*,double*,double*);
static void dqpsrt(const int,const int,int*,double*,const double*,int*,int*);
'''
bodies={name:extract(name) for name in names}
a.output.mkdir(parents=True,exist_ok=True)
(a.output/'original.c').write_text(license+decls+'\n'+constants+'\n\n'+'\n\n'.join(bodies.values())+'\n')
(a.output/'source-lock.json').write_text(json.dumps(dict(source_url='https://raw.githubusercontent.com/scipy/scipy/v1.17.1/scipy/integrate/__quadpack.c',version='1.17.1',source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),function_sha256={n:hashlib.sha256(b.encode()).hexdigest()for n,b in bodies.items()},changes='unchanged numerical bodies; Python header replaced with C declarations',output_sha256=hashlib.sha256((a.output/'original.c').read_bytes()).hexdigest()),indent=2)+'\n')
print('Extracted four unchanged original numerical bodies; Python code included: none')
