#!/usr/bin/env python3
"""CPython/Director lexical parity on synthetic Unicode fixtures; no runtime imports."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import unicodedata

HELPERS = runpy.run_path(str(Path(__file__).with_name('director-h3-prompt-oracle.py')))

def main():
    own = HELPERS['OWN']
    destination = Path(sys.argv[1])
    assert destination.parent == own and destination.resolve() == destination
    mount = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(own)]))['filesystems'][0]
    assert mount['source'] == '/dev/mapper/veracrypt1' and not os.statvfs(own).f_flag & os.ST_RDONLY
    destination.mkdir(exist_ok=False)
    source = HELPERS['DIRECTOR']
    scope = HELPERS['references']()
    dialogue = HELPERS['pure'](source/'timeline/dialogue.py', scope)
    common = HELPERS['pure'](source/'h3/prompt_common.py', scope)
    i2v = HELPERS['pure'](source/'h3/i2v_prompt.py', {**common, 'compile_dialogue_macros': dialogue['compile_dialogue_macros']})
    studio = {'re': HELPERS['re'], 'Any': HELPERS['Any'], 'REFERENCE_RE': HELPERS['re'].compile(r'(?<![\w@])@[A-Za-z0-9][A-Za-z0-9_.-]*')}
    import ast
    nodes = [n for n in ast.parse((source/'prompt_studio.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name in {'protect_references', 'restore_references'}]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'actual_prompt_studio', 'exec'), studio)
    cases, expected = [], []
    points = set()
    previous = False
    for point in range(0x110000):
        current = chr(point).isalnum()
        if current != previous:
            points.update([point - 1, point])
        previous = current
    # Every Python alnum transition plus combining marks and newer Unicode assignments.
    points.update([0x345, 0x5b0, 0x10d40, 0x1c89, 0x1e5d0, 0x1ccf0])
    for point in sorted(p for p in points if 0 <= p < 0x110000 and not 0xd800 <= p <= 0xdfff):
        text = chr(point) + '@keeper'
        protected, refs = studio['protect_references'](text)
        restored, _ = studio['restore_references'](protected, refs)
        cases.append({'operation':'protection', 'source':text, 'response':protected})
        expected.append({'protected':protected, 'references':refs, 'restored':restored, 'introduced':False})
    maps = {'subject':{'keeper':'<Subject 1>', 'i_keeper':'<Subject 2>'}, 'speaker':{'keeper':'S1', 'i_keeper':'S2'}, 'language':{'keeper':'Swedish'}, 'voice_profile':{'keeper':'\u001cquiet.\u001f'}}
    texts = []
    for ch in [chr(i) for i in range(0x110000) if chr(i).isspace()]:
        texts.extend([f'@keeper{ch}says{ch}[{ch}]{ch}"Hej!"', f'@keeper sings in{ch}voiceover "Hej!"'])
    for alias in ['keeper','KEEPER','İkeeper','ıkeeper','ſkeeper','Keeper']:
        for verb in ['SAYS','ſays','ſpeakſ','aſkſ','sİngs']:
            for context in ['', ' in voiceover', ' ın voiceover', ' off-ſcreen']:
                texts.append(f'@{alias} {verb}{context} "Hej!"')
    for prefix in ['İ','ı','ſ','K','é','\u0345','\u05b0']:
        texts.append(prefix+'@keeper says "Hej!"')
    for text in texts:
        kwargs = {key+'_by_alias': maps.get(key, {}) for key in ['subject','speaker','language','voice_audio','voice_profile']}
        cases.append({'operation':'dialogue', 'text':text, 'maps':maps})
        expected.append({'prompt':dialogue['compile_dialogue_macros'](text, **kwargs), 'cues':dialogue['parse_dialogue_macros'](text)[0]})
    for ch in [chr(i) for i in range(0x110000) if chr(i).isspace()] + ['\u0345','\u05b0']:
        for label in ['music','muſic','ſcore','SOUND','ſound']:
            data = {'sections':[{'item_id':'own_section','start_time':0.,'end_time':1.,'prompt':f'View{ch}{label}{ch}:{ch}wind; audio: waves'}], 'text_references':[], 'keyframes':{}, 'global_prompt':{}, 'scene_prompt':{'summary':ch+'Own'+ch+'scene'+ch}, 'effective_duration_seconds':1.}
            prompt, metadata = i2v['compile_i2v_prompt'](**data)
            cases.append({'operation':'i2v', 'input':data})
            expected.append({'prompt':prompt,'metadata':metadata})
    probe = HELPERS['ROOT']/'comfyui/h3-video-night-20261021/target/debug/examples/h3_prompt_probe'
    result = subprocess.run([str(probe)], input=json.dumps(cases), capture_output=True, text=True, check=True)
    actual = json.loads(result.stdout)
    expected = json.loads(json.dumps(expected))
    assert len(actual) == len(expected)
    failures = []
    for index, (a, b) in enumerate(zip(actual, expected)):
        if a != b:
            failures.append(index)
            (destination/f'failure-{index}.json').write_text(json.dumps({'input':cases[index], 'director':b, 'rust':a}, indent=2))
    report = {'passed':not failures, 'cases':len(cases), 'failures':failures, 'python':sys.version, 'unicode':unicodedata.unidata_version, 'cpu_only':True, 'coverage':'Every Python alnum range boundary, all Python whitespace, Unicode case-insensitive ASCII extras, dialogue, channel split and protection'}
    (destination/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({**report, 'failures':len(failures)}))
    raise SystemExit(bool(failures))

if __name__ == '__main__':
    main()
