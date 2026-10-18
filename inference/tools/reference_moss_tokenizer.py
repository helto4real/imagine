#!/usr/bin/env python3
"""Original MOSS/Qwen3.5 tokenizer on source-owned text and image placeholders."""
from pathlib import Path
loader=Path(__file__).with_name('reference_qwen_tokenizer.py')
exec(compile(loader.read_text().split("\nmodel=ns['QwenImage21Tokenizer']()",1)[0],str(loader),'exec'))
for name in ['comfy/text_encoders/qwen35.py','comfy/text_encoders/moss_image.py']:
 path=a.comfy/name;expected=next(x['sha256']for x in manifest if x['root']=='comfy'and x['path']==name)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name;sources['comfy/'+name]=expected
tree=ast.parse((a.comfy/'comfy/text_encoders/qwen35.py').read_text())
constant=next(n for n in tree.body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='QWEN35_MODELS')
exec(compile(ast.Module(body=[constant],type_ignores=[]),str(a.comfy/'comfy/text_encoders/qwen35.py'),'exec'),ns)
definitions('comfy/text_encoders/qwen35.py',['Qwen35Tokenizer','Qwen35ImageTokenizer'])
ns['comfy'].text_encoders.qwen35=NS(Qwen35ImageTokenizer=ns['Qwen35ImageTokenizer'])
definitions('comfy/text_encoders/moss_image.py',['MOSSImageTokenizer'])
model=ns['MOSSImageTokenizer']();sd=model.qwen35_9b;raw=sd.tokenizer
old=json.loads((project/'inference/fixtures/qwen-image21-tokenizer.json').read_text())
texts=[c['text']for c in old['cases']if c['operation']=='raw'];cases=[]
for text in texts:cases.append(dict(operation='raw',text=text,ids=raw(text)['input_ids']))
options=[{},dict(thinking=True),dict(thinking=False),dict(prevent_empty_text=True),dict(images=1),dict(images=3),
    dict(system_prompt='A synthetic system.'),dict(llama_template='prefix {} suffix'),
    dict(llama_template='<|im_start|>system\nsynthetic<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n',system_prompt='Injected.'),
    dict(min_length=120),dict(max_length=32),dict(min_padding=3)]
for opts in options:
 for i,text in enumerate(texts[:19]+['  hello  ','\u001c \t\n','\u0085\u2003hello\u2003','<|im_start|>user\nraw<|im_end|>']):
  kw=dict(opts);count=kw.pop('images',0);images=[torch.full((1,16,16,3),float(k))for k in range(count)]
  sd.max_length=kw.pop('max_length',99999999);sd.min_padding=kw.pop('min_padding',None)
  c=dict(operation='prompt',text=text,options=opts,word_ids=i%2==0)
  result=model.tokenize_with_weights(text,images=images,return_word_ids=c['word_ids'],**kw)
  def token(t):return dict(image=next(i for i,image in enumerate(images)if t['data']is image))if isinstance(t,dict)else t
  c['batches']=[[[token(t[0]),*t[1:]]for t in batch]for batch in result['qwen35_9b']];cases.append(c)
a.output.write_text(json.dumps(dict(schema=1,sources=sources,cases=cases,
    provenance='Original MOSS/Qwen3.5 wrappers and Qwen2 BPE; own text and placeholders; no trained model'),ensure_ascii=True,separators=(',',':'))+'\n')
print('Captured',len(cases),'MOSS tokenizer cases',flush=True)
