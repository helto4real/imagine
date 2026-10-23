#!/usr/bin/env python3
"""Source-only Director main-target resolver/normalizer oracle; no module imports."""
import ast,copy,json,os,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DIRECTOR=Path('/home/thhel/git/comfyui-helto-director/shared')
def selected(path,names,scope):
    nodes=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name in names]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),scope)
def main():
    scope={'deepcopy':copy.deepcopy,'json':json,'LORA_MERGE_MODE_INHERIT_GLOBAL':'Inherit Global','LORA_MERGE_MODE_ADD_TO_GLOBAL':'Add To Global','LORA_MERGE_MODE_REPLACE_GLOBAL':'Replace Global','LORA_MERGE_MODE_DISABLE_LORAS':'Disable LoRAs'}
    selected(DIRECTOR/'timeline/defaults.py',{'create_default_lora_stack'},scope)
    selected(DIRECTOR/'timeline/planner_context.py',{'resolve_model_loras_for_shot','_project_lora_stack','_copy_lora_stack','_merge_lora_stack','_raw_dict'},scope)
    norm={'os':os,'re':re,'LORA_CONFIG_VERSION':1,'_available_loras':lambda:None}
    selected(DIRECTOR/'lora/config.py',{'normalize_lora_config','_iter_raw_loras','_sort_lora_key','_resolve_lora_name'},norm)
    row=lambda name,m=1.,c=1.,enabled=True:dict(name=name,strength_model=m,strength_clip=c,enabled=enabled)
    stack=lambda rows,match='':dict(version=1,loras=rows,ui=dict(show_strengths='single',match=match))
    global_stack=stack([row('/own/global.weights',.75,-2),row('None'),row('/own/off',enabled=False),row('/own/zero',0,0)],'global')
    local_stack=stack([row('/own/local.weights',-1.25,.4),row('/own/global.weights',.25,.25)],'scene')
    cases=[]
    for enabled in [False,True]:
        for mode in ['Inherit Global','Add To Global','Replace Global','Disable LoRAs']:
            scene=dict(enabled=enabled,merge_mode=mode,main=local_stack)
            shot={'lora_overrides':dict(enabled=enabled,merge_mode=mode,targets={'minimax_h3':{'main':local_stack}})}
            resolved=scope['resolve_model_loras_for_shot']({'global':{'minimax_h3':{'main':global_stack}}},shot,model_key='minimax_h3',target_keys=['main'])['main']
            active=norm['normalize_lora_config'](resolved)['loras']
            cases.append(dict(global_stack=global_stack,scene=scene,resolved=resolved,active=active))
    path=ROOT/'comfyui/ui-groups-20261021/source/crates/video-contract/src/fixtures/director-h3-loras.json';path.parent.mkdir(exist_ok=True);path.write_text(json.dumps({'source':'AST-selected actual Director functions; no modules/config directories/runtime IO','cases':cases},indent=2))
    print(json.dumps({'fixture':str(path),'cases':len(cases)}))
if __name__=='__main__':main()
