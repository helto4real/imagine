#!/usr/bin/env python3
"""Call the locked pure C++ Torch error formatter; no Python/Torch inference."""
import argparse,ctypes,hashlib,json,subprocess
from pathlib import Path
from reference_paths import reference_dir
p=argparse.ArgumentParser(description=__doc__)
for name in ['source','site','work','output']:p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();project=Path(__file__).resolve().parents[2]
lock=json.loads((reference_dir(project)/'source-acquisition-lock.json').read_text())
expected=lock['repos']['pytorch']['source_file_sha256']['torch/csrc/Exceptions.cpp'];assert hashlib.sha256(a.source.read_bytes()).hexdigest()==expected
s=a.source.read_text();start=s.index('static void processErrorMsgInplace(std::string& str) {');end=s.index('void translate_exception_to_python(',start);body=s[start:end]
a.work.mkdir(parents=True,exist_ok=True);src=a.work/'formatter.cpp';lib=a.work/'formatter.so'
src.write_text('#include <array>\n#include <string>\n#include <string_view>\n#include <cstring>\n#include <c10/util/StringUtil.h>\nnamespace {\n'+body+'}\nextern "C" char* own_format(const char* text) { return strdup(processErrorMsg(text).c_str()); }\nextern "C" void own_free(void* text) { free(text); }\n')
cmd=['clang++','-std=c++20','-shared','-fPIC','-O0','-D_GLIBCXX_USE_CXX11_ABI=1','-I'+str(a.site/'torch/include'),str(src),'-L'+str(a.site/'torch/lib'),'-Wl,-rpath,'+str(a.site/'torch/lib'),'-lc10','-o',str(lib)]
subprocess.run(cmd,check=True);original=ctypes.CDLL(str(lib));original.own_format.argtypes=[ctypes.c_char_p];original.own_format.restype=ctypes.c_void_p;original.own_free.argtypes=[ctypes.c_void_p]
# Independent source-derived corpus includes concatenated, repeated and Unicode text.
import re
names=re.findall(r'\{"([A-Za-z0-9.]+)", "[A-Za-z0-9.]+"\}',body)
messages=['','own bounded error\nsecond line','own Type keyword',' '.join(names),'åäö '+''.join(names)+' 終',*[f'own {name} {name}\n{name}Suffix'for name in names]]
cases=[]
for message in messages:
 raw=original.own_format(message.encode())
 try:output=ctypes.string_at(raw).decode()
 finally:original.own_free(raw)
 cases.append(dict(input=message,output=output))
a.output.write_text(json.dumps(dict(schema=1,source_sha256=expected,cpp_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),library_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),trained_model_weights=0,torch_python_imported=False,cases=cases),ensure_ascii=False,indent=2)+'\n')
print('Captured',len(cases),'original pure C++ formatter cases; no tensor allocations or Torch Python import')
