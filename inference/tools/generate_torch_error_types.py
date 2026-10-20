#!/usr/bin/env python3
"""Retain the locked pure processErrorMsg type replacement table in Rust."""
import argparse,hashlib,json,re
from pathlib import Path
from reference_paths import reference_dir
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
project=Path(__file__).resolve().parents[2]
lock=json.loads((reference_dir(project)/'source-acquisition-lock.json').read_text())
expected=lock['repos']['pytorch']['source_file_sha256']['torch/csrc/Exceptions.cpp']
assert hashlib.sha256(a.source.read_bytes()).hexdigest()==expected,'Locked exception source changed'
s=a.source.read_text();start=s.index('static void processErrorMsgInplace(std::string& str) {');end=s.index('void translate_exception_to_python(',start);s=s[start:end]
table=s[s.index('std::to_array<'):s.index('// Avoid doing any work')]
pairs=re.findall(r'\{"([A-Za-z0-9.]+)", "([A-Za-z0-9.]+)"\}',table)
assert len(pairs)==32 and len({v[0]for v in pairs})==32
assert 'if (str.find("Type") == str.npos)'in s and 'c10::ReplaceAll(str, it.first, it.second);'in s
lines=['// Generated from locked torch/csrc/Exceptions.cpp; no runtime source parsing.',f'// Source SHA256: {expected}','pub(crate) const TYPE_NAMES: &[(&str, &str)] = &[']
lines.extend(f'    ({json.dumps(before)}, {json.dumps(after)}),'for before,after in pairs);lines.append('];')
a.output.write_text('\n'.join(lines)+'\n')
print('Retained',len(pairs),'original ordered error type substitutions')
