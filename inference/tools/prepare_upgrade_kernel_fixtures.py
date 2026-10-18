#!/usr/bin/env python3
"""Merge only the fresh Kitchen .38 captures into the preserved base fixture."""
from reference_paths import reference_dir
import argparse,gzip,hashlib,json,os,tempfile
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fixture',type=Path,required=True);a=p.parse_args()
project=Path(__file__).resolve().parents[2];reference=reference_dir(project)
lock=json.loads((reference/'reference-lock.json').read_text())
archive=project/'inference/fixtures/kitchen-038-upgrade.json.gz'
data=json.loads(gzip.decompress(archive.read_bytes()))
assert data['reference_revision']==lock['comfy_revision']and data['kitchen_version']==lock['kitchen_version']
manifest=json.loads((reference/'source-manifest.json').read_text())
for capture in data['captures'].values():
 for name,digest in capture.items():assert digest==next(x['sha256']for x in manifest if x['root']=='site'and x['path']==name),name
fixture=json.loads(a.fixture.read_text());fixture.update(data['sections'])
with tempfile.NamedTemporaryFile(dir=a.fixture.parent,prefix='.kernel-fixture-',mode='w',delete=False)as output:
 temporary=Path(output.name);json.dump(fixture,output,separators=(',',':'));output.write('\n')
try:os.replace(temporary,a.fixture)
finally:temporary.unlink(missing_ok=True)
provenance=dict(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),reference_revision=data['reference_revision'],
 sections={k:len(v)for k,v in data['sections'].items()},remaining_sections='Preserved base captures; this merge does not claim a complete fresh Kitchen registry oracle')
a.fixture.with_suffix('.kitchen038-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps(provenance))
