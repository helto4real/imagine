#!/usr/bin/env python3
"""Compare synthetic catalog inputs with Director's pure source validator; no config load."""
import importlib.util,json,os,subprocess,sys,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OWN=Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')
def main():
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(OWN)]))['filesystems'][0]
    assert (mount['target'],mount['source'],mount['fstype'])==('/home','/dev/mapper/cryptroot[/@home]','btrfs') and not os.statvfs(OWN).f_flag&os.ST_RDONLY
    spec=importlib.util.spec_from_file_location('director_catalog_source','/home/thhel/git/comfyui-helto-director/shared/prompt_intellisense.py')
    director=importlib.util.module_from_spec(spec);sys.modules[spec.name]=director;spec.loader.exec_module(director)
    def document(version='1.1'):
        choice={'key':'meet','aliases':['greet'],'label':'Meeting','append':'space','segments':[{'text':'Hello '}]}
        if version=='1.1':
            choice['placeholders']={'who':{'label':'Person','initial_text':'@keeper','hint':'Choose person'}}
            choice['segments'] += [{'placeholder_ref':'who'},{'text':' greets '},{'placeholder_ref':'who'}]
        else:choice['segments'] += [{'placeholder':{'id':'who','label':'Person','initial_text':'Å🙂','hint':'Choose person'}}]
        return {'schema_version':version,'mode':'replace','namespaces':[{'key':'own','label':'Own','description':'Own source fixtures','choices':[choice]}]}
    def normalized(c):
        def nodes(ns):return [dict(key=n['key'],label=n.get('label',''),description=n.get('description',''),groups=nodes(n.get('groups',[])),choices=[dict(key=c['key'],label=c['label'],description=c.get('description',''),aliases=c.get('aliases',[]),append=c['append'],segments=c['segments']) for c in n.get('choices',[])]) for n in ns]
        return {'schema_version':c['schema_version'],'namespaces':nodes(c['namespaces'])}
    base=document();merge={'schema_version':'1.1','mode':'merge','namespaces':[{'key':'own','choices':[{'key':'meet','label':'Replaced','append':'none','segments':[{'text':'Own replacement'}]}]}]}
    cases=[('named-repeat',{'own.yaml':base}),('inline-1.0',{'own.yaml':document('1.0')}),('replace-order',{'z-base.yaml':base,'a-merge.yaml':merge})]
    bad=document();bad['namespaces'][0]['choices'][0]['description']='';cases.append(('empty-description',{'own.yaml':bad}))
    bad=document('1.0');bad['namespaces'][0]['choices'][0]['description']='Description';cases.append(('1.0-description',{'own.yaml':bad}))
    bad=document();bad['namespaces'][0]['choices'][0]['segments'].append({'placeholder_ref':'missing'});cases.append(('missing-ref',{'own.yaml':bad}))
    cases.append(('multiple-replace',{'a.yaml':base,'b.yaml':base}))
    sources=ROOT/'comfyui/ui-groups-20261021/source/assets/video/prompt-catalog'
    cases.append(('public-source-templates',{p.name:p.read_text() for p in sources.glob('*.yaml')}))
    composer_cases=[{'text':text,'previous':{'who':{'label':'Own person','initial_text':'@keeper','hint':'Own hint'}}} for text in ['@WHO meets @who beside @@literal','email@host @new-person','Å🙂 @who @-bad @@ @','@','A new plain phrase']]
    duplicate_cases=[{'choices':[dict(base['namespaces'][0]['choices'][0]),{'key':'meet_2','label':'Meeting 2','aliases':['greet_2'],'append':'none','segments':[{'text':'Other'}]}],'groups':[{'key':'meet_3'}]}]
    results=[]
    for name,raw in cases:
        sources={n:t if isinstance(t,str)else json.dumps(t,ensure_ascii=False) for n,t in raw.items()}
        try:expected=director.validate_prompt_intellisense_catalog_sources(sources);accepted=True
        except director.PromptCatalogError:expected=None;accepted=False
        queries=['#ow','#own.greet','Å🙂 #own.me','text#own.me','#own.'] if name=='named-repeat' else []
        run=subprocess.run([str(ROOT/'comfyui/ui-groups-20261021/target/debug/examples/prompt_catalog_probe')],input=json.dumps({'sources':sources,'queries':queries,'composer':composer_cases,'duplicates':duplicate_cases}),text=True,capture_output=True,check=True)
        actual=json.loads(run.stdout);assert actual['ok']==accepted,(name,actual)
        if accepted:assert normalized(actual['catalog'])==normalized(expected),(name,actual,expected)
        if queries:
            node=r'''import fs from 'node:fs';
const source=fs.readFileSync('/home/thhel/git/comfyui-helto-director/web/timeline/prompt_intellisense.js','utf8');
const m=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const r=JSON.parse(fs.readFileSync(0,'utf8')),catalog=m.normalizePromptIntellisenseCatalog(r.catalog);
const chars=(t,n)=>[...t.slice(0,n)].length;
console.log(JSON.stringify(r.queries.map(text=>{const context=m.promptCatalogCompletionContext(text,text.length);return {text,items:m.filterPromptCatalogCompletions(catalog,context).map(e=>{let expansion=null;if(e.choice){const x=m.expandPromptCatalogChoice(text,context,e.choice);expansion={text:x.replacement,slots:x.placeholders.map(p=>({id:p.id,label:p.label,hint:p.hint,occurrences:p.occurrences.map(o=>({start:chars(x.value,o.start)-chars(text,context.replaceStart),end:chars(x.value,o.end)-chars(text,context.replaceStart)}))}))};}return {path:[e.namespace.key,...(e.choicePath??e.groupPath??[])],label:e.label,expansion};})};})));
'''
            js=subprocess.run(['node','--input-type=module','-e',node],input=json.dumps({'catalog':expected,'queries':queries}),text=True,capture_output=True,check=True)
            assert actual['queries']==json.loads(js.stdout),(actual['queries'],js.stdout)
            editor=r'''import fs from 'node:fs';
const source=fs.readFileSync('/home/thhel/git/comfyui-helto-director/web/timeline/prompt_catalog_editor.js','utf8').replace(/^import .*;\s*$/gm,'');
const m=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const cases=JSON.parse(fs.readFileSync(0,'utf8'));
console.log(JSON.stringify(cases.map(r=>{const p=m.parsePhraseComposer(r.text,r.previous);return {parsed:{...p,occurrences:Object.fromEntries(p.occurrences)},text:m.phraseComposerText(p)};})));
'''
            js=subprocess.run(['node','--input-type=module','-e',editor],input=json.dumps(composer_cases),text=True,capture_output=True,check=True)
            assert actual['composer']==json.loads(js.stdout),(actual['composer'],js.stdout)
            duplicate_js=editor[:editor.index('const cases=')]+"const cases=JSON.parse(fs.readFileSync(0,'utf8'));console.log(JSON.stringify(cases.map(c=>{const v=m.duplicatePromptCatalogChoice(c,c.choices[0]);return {key:v.key,label:v.label,aliases:v.aliases};})));"
            js=subprocess.run(['node','--input-type=module','-e',duplicate_js],input=json.dumps(duplicate_cases),text=True,capture_output=True,check=True)
            assert actual['duplicates']==json.loads(js.stdout),(actual['duplicates'],js.stdout)
        results.append({'name':name,'accepted':accepted,'matched':True,'frontend_queries':len(queries),'composer_cases':len(composer_cases) if queries else 0,'duplicate_cases':len(duplicate_cases) if queries else 0})
    path=OWN/'evidence'/('catalog-oracle-'+uuid.uuid4().hex[:8]+'.json');path.write_text(json.dumps({'passed':True,'source_only':True,'director_config_read':False,'cases':results},indent=2))
    print(json.dumps({'passed':True,'cases':len(results),'report':str(path)}))
if __name__=='__main__':main()
