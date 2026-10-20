#!/usr/bin/env python3
"""Source constants for the locked restricted-pickle global lookup policy."""
import argparse,ast,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
def assignment(tree,name):
 return ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id==name for t in n.targets)))
utils=a.site/'torch/_utils.py';restricted=a.site/'torch/_weights_only_unpickler.py'
tree=ast.parse(utils.read_text());imports=assignment(tree,'IMPORT_MAPPING');names=assignment(tree,'NAME_MAPPING')
tree=ast.parse(restricted.read_text());blocked=assignment(tree,'_blocklisted_modules')
owner=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Unpickler');load=next(n for n in owner.body if isinstance(n,ast.FunctionDef)and n.name=='load');special=[]
for node in ast.walk(load):
 if isinstance(node,ast.If)and isinstance(node.test,ast.Compare)and isinstance(node.test.left,ast.Name)and node.test.left.id=='full_path'and isinstance(node.test.comparators[0],ast.List):
  paths=ast.literal_eval(node.test.comparators[0]);message=ast.literal_eval(node.body[0].exc.args[0]);special.append((paths,message))
assert len(special)==2
q=json.dumps
lines=['// Generated from locked Torch source; do not edit.',f'// _utils.py sha256: {hashlib.sha256(utils.read_bytes()).hexdigest()}',f'// _weights_only_unpickler.py sha256: {hashlib.sha256(restricted.read_bytes()).hexdigest()}','pub(super) const IMPORTS: &[(&str,&str)] = &[']
lines += [f'({q(k)}, {q(v)}),'for k,v in imports.items()];lines += ['];','pub(super) const NAMES: &[super::NameMapping] = &[']
lines += [f'(({q(k[0])},{q(k[1])}),({q(v[0])},{q(v[1])})),'for k,v in names.items()];lines += ['];','pub(super) const BLOCKED: &[&str] = &[']+[q(v)+','for v in blocked]+['];','pub(super) const SPECIAL: &[(&[&str], &str)] = &[']
lines += ['(&['+','.join(q(v)for v in paths)+'], '+q(message)+'),'for paths,message in special];lines += ['];']
lines=[item for line in lines for item in (['#[rustfmt::skip]',line]if line.startswith('pub(super) const ')else[line])]
a.output.write_text('\n'.join(lines)+'\n');print('Generated',len(imports),'module mappings,',len(names),'name mappings and two special import groups')
