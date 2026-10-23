#!/usr/bin/env python3
"""Source-only Director hard-cut timing oracle, synthetic CPU tensors, no runtime imports."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

OWN = Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')
DIRECTOR = Path('/home/thhel/git/comfyui-helto-director/shared')

def main():
    import torch
    destination = Path(sys.argv[1])
    assert destination.parent == OWN/'evidence' and not destination.exists()
    mount = json.loads(subprocess.check_output(['findmnt','-J','-T',str(OWN)]))['filesystems'][0]
    assert (mount['target'],mount['source'],mount['fstype']) == ('/home','/dev/mapper/cryptroot[/@home]','btrfs') and not os.statvfs(OWN).f_flag & os.ST_RDONLY
    source = DIRECTOR/'timeline/sequence_assembly.py'
    names = {'_assemble_clip_frames','_append_with_boundary','_copy_frames_into','_entry_frame_shape','_load_entry_frames','_resize_sequence_entry','_release_entry_frames','_boundaries_by_pair','_boundary_overlap_frames','_boundary_overlap_frame_count'}
    tree = ast.parse(source.read_text())
    functions = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in functions} == names
    namespace = {'torch':torch,'SequenceAssemblyError':ValueError}
    for n in ast.parse((DIRECTOR/'contracts/video_timeline.py').read_text()).body:
        if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant):
            for target in n.targets:
                if isinstance(target,ast.Name) and target.id.startswith('BOUNDARY_MODE_'):
                    namespace[target.id] = ast.literal_eval(n.value)
    future = ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future,*functions],type_ignores=[]))
    exec(compile(module,str(source),'exec'),namespace)
    rows=[]
    for target in [3,12,100]:
        entries=[{'shot_id':'own_a','timeline_start_time':0.,'timeline_end_time':1.,'frames':torch.full((24,2,2,3),.25)}, {'shot_id':'own_b','timeline_start_time':5.,'timeline_end_time':7.,'frames':torch.full((48,2,2,3),.75)}]
        output=namespace['_assemble_clip_frames'](entries,{'project':{'duration_seconds':target},'sequence':{'boundaries':[]}}, {'resolution':{},'boundaries':[]},24.)
        assert output.shape[0]==72 and torch.all(output[:24]==.25) and torch.all(output[24:]==.75)
        rows.append({'project_duration':target,'timeline_ranges':[[0,1],[5,7]],'output_frames':output.shape[0],'output_seconds':output.shape[0]/24,'output_ranges':[[e['output_start_frame'],e['output_end_frame_exclusive']] for e in entries]})
    report={'source':str(source),'actual_functions':sorted(names),'scope':'Actual hard-cut assembly body with own CPU tensors, same-size frames; no media, browser, config or Director module import','passed':True,'cases':rows,'finding':'Director concatenates included Scene durations; timeline gaps and project editing length do not insert blank frames or pad the final output.'}
    destination.write_text(json.dumps(report,indent=2))
    print(json.dumps(report))

if __name__=='__main__':main()
