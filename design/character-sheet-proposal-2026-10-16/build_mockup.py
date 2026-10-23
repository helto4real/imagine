from pathlib import Path
from html import escape
P=Path(__file__).parent
C={'bg':'#181825','surface':'#1e1e2e','field':'#313244','raised':'#45475a','media':'#11111b','text':'#cdd6f4','dim':'#a6adc8','accent':'#fab387'}
icons={
'image':'<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8" cy="8" r="1.5"/><path d="M3 17l6-6 4 4 3-3 5 5"/>',
'video':'<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M7 3v18M17 3v18M3 7h4m10 0h4M3 12h4m10 0h4M3 17h4m10 0h4"/>',
'sheet':'<rect x="2" y="3" width="20" height="18" rx="2"/><path d="M11 3v18M14 8h5m-5 4h5m-5 4h5"/><circle cx="6.5" cy="9" r="2"/><path d="M4 17v-2a2.5 2.5 0 015 0v2"/>',
'plus':'<path d="M12 4v16M4 12h16"/>',
'up':'<path d="M12 20V4m-7 7 7-7 7 7"/>',
'down':'<path d="M12 4v16m-7-7 7 7 7-7"/>',
'trash':'<path d="M4 6h16M9 6V3h6v3M6 6l1 15h10l1-15M10 10v7m4-7v7"/>',
'fit':'<path d="M3 9V3h6m6 0h6v6M3 15v6h6m6 0h6v-6M3 3l6 6m6 6 6 6M21 3l-6 6M9 15l-6 6"/>',
'actual':'<rect x="2" y="3" width="20" height="18" rx="1"/><path d="m6 10 2-2v8m8-6 2-2v8M12 10v1m0 3v1"/>',
'hand':'<path d="M8 12V6a2 2 0 014 0v5-7a2 2 0 014 0v7-5a2 2 0 014 0v10l-4 5h-6l-6-8a2 2 0 012-3z"/>',
'generate':'<path d="m6 3 15 9-15 9z"/>',
'retry':'<path d="M20 9A8 8 0 105 18M20 3v6h-6"/>',
'stop':'<rect x="5" y="5" width="14" height="14" rx="1"/>',
'eye':'<path d="M2 12l3-5 4-2h6l4 2 3 5-3 5-4 2H9l-4-2z"/><circle cx="12" cy="12" r="3"/>',
'settings':'<path d="m10 2 4 0 1 3 3 1 3 3-2 3 2 3-3 3-3 1-1 3h-4l-1-3-3-1-3-3 2-3-2-3 3-3 3-1z"/><circle cx="12" cy="12" r="3"/>'}
def render(sheet):
 s=['<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="800" viewBox="0 0 1280 800" role="img" aria-label="Imagine character sheet design proposal">', '<defs><linearGradient id="panel" x2="0" y2="1"><stop stop-color="#1e1e2e"/><stop offset="1" stop-color="#181825"/></linearGradient><linearGradient id="button" x2="0" y2="1"><stop stop-color="#45475a"/><stop offset="1" stop-color="#313244"/></linearGradient><linearGradient id="active" x2="0" y2="1"><stop stop-color="#4f3a2a"/><stop offset="1" stop-color="#3d2d20"/></linearGradient><linearGradient id="header" x2="0" y2="1"><stop stop-color="#313244"/><stop offset="1" stop-color="#1e1e2e"/></linearGradient></defs>']
 def rect(x,y,w,h,fill,stroke='none',r=5):s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{C.get(fill,fill)}" stroke="{C.get(stroke,stroke)}"/>')
 def text(x,y,t,size=14,color='text',weight='normal',family='Noto Sans, sans-serif'):s.append(f'<text x="{x}" y="{y}" fill="{C.get(color,color)}" font-family="{family}" font-size="{size}" font-weight="{weight}">{escape(t)}</text>')
 def line(x,y,x2,y2,color='field'):s.append(f'<path d="M{x} {y}H{x2}" stroke="{C.get(color,color)}"/>' if y==y2 else f'<path d="M{x} {y}L{x2} {y2}" stroke="{C.get(color,color)}"/>')
 def btn(x,y,icon,active=False,disabled=False):
  s.append(f'<g opacity="{.35 if disabled else 1}">');rect(x,y,32,32,'url(#active)' if active else 'url(#button)','#93664a' if active else 'raised');s.append(f'<g transform="translate({x+7} {y+7}) scale(.75)" fill="none" stroke="{"#f38ba8" if icon=="trash" else "#fddcc4" if active else C["text"]}" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round">{icons[icon]}</g></g>')
 def field(y,label,lines,h):
  text(987,y,label);rect(987,y+12,273,h,'field','raised');
  for i,t in enumerate(lines):text(997,y+34+i*19,t,13,'text',family='monospace')
 def control(y,label,value,chevron=False,disabled=False):
  text(987,y+21,label,color='dim' if disabled else 'text');rect(1067,y,193,32,'field','raised');text(1077,y+21,value,color='dim' if disabled else 'text');
  if chevron:text(1242,y+20,'⌄',14,'dim')
 def person(x,y,scale=1,back=False):
  s.append(f'<g transform="translate({x} {y}) scale({scale})" fill="#c7c7c7" stroke="#9c9c9c" stroke-width="1.5">')
  s.append('<ellipse cx="0" cy="22" rx="13" ry="19"/><path d="M-7 39v10h14V39M-19 51Q0 40 19 51l-5 45 19 56h-66l19-56z" fill="#80938d"/><path d="m-19 53-13 57-5 12m56-69 13 57 5 12M-14 153l-1 46-6 6h18l1-52m16 0 1 46 6 6H3l-1-52"/>')
  if not back:s.append('<path d="M-7 20h3m8 0h3m-9 11h4" fill="none"/>')
  s.append('</g>')
 rect(0,0,1280,800,'bg',r=0);rect(0,0,1280,64,'url(#header)',r=0)
 text(18,39,'Imagine',20,weight='bold');line(112,23,112,43,'raised');text(130,38,'Qwen Image 2.1',14,'dim')
 btn(585,16,'image',not sheet);btn(623,16,'sheet',sheet);btn(661,16,'video')
 s.append('<circle cx="1081" cy="32" r="3.5" fill="#a6e3a1"/>');text(1092,37,'Connected',14,'dim');btn(1190,16,'settings');btn(1228,16,'eye')
 for x,w in [(8,246),(262,705),(975,297)]:rect(x,72,w,720,'media' if x==262 else 'url(#panel)','field',6)
 text(18,103,'References · 1/'+('8' if sheet else '10'));btn(212,82,'plus');line(18,122,244,122)
 rect(18,132,186,195,'media',r=0)
 # Synthetic reference illustration, no runtime assets.
 s.append('<g transform="translate(40 143)"><path d="M18 170Q16 88 29 43Q68-9 108 43Q128 96 126 170" fill="#b9afa0"/><ellipse cx="72" cy="91" rx="42" ry="61" fill="#d8d5cf"/><path d="M29 68Q50 25 96 34L117 65Q114 16 72 14Q30 16 29 68" fill="#b9afa0"/><path d="M49 87h9m28 0h9M67 123h12" stroke="#77736e" stroke-width="3"/></g>')
 rect(18,132,24,24,'surface');text(26,149,'1',12);btn(212,174,'up',disabled=True);btn(212,212,'down',disabled=True);btn(212,250,'trash')
 if sheet:
  for x,w,label,sel in [(272,62,'Sheet',True),(340,78,'Portrait',False),(424,72,'Views',False)]:
   rect(x,82,w,32,'url(#active)' if sel else 'url(#button)','#93664a' if sel else 'raised');text(x+10,103,label,color='accent' if sel else 'text')
 btn(849,82,'fit',True);btn(887,82,'actual');btn(925,82,'hand')
 if sheet:
  rect(272,226,685,385,'#ededeb',r=0)
  s.append('<g transform="translate(285 236)"><path d="M21 365Q3 215 37 83Q115-46 234 67Q290 177 271 365" fill="#b9afa0"/><ellipse cx="148" cy="184" rx="104" ry="149" fill="#d8d5cf"/><path d="M45 137Q81 37 208 39l45 83Q240-5 136 4Q36 4 45 137" fill="#b9afa0"/><path d="M86 173h22m71 0h22m-68 86h30" stroke="#77736e" stroke-width="5"/></g>')
  person(670,253,1.55);person(853,253,1.55,True)
  field(101,'Character',['Adult woman, blonde hair,','blue eyes, natural skin.','Preserve the reference','identity.'],112)
  field(253,'Clothing & accessories',['Forest-green summer dress,','small gold earrings and a','delicate necklace.'],96)
  negative_y=381;controls_y=455
 else:
  rect(272,226,685,385,'#ededeb',r=0);person(615,242,1.7)
  field(101,'Prompt',['An adult woman in a green','summer dress, standing in','soft natural light.','Preserve her identity from','the reference image.'],172)
  negative_y=317;controls_y=391
 rect(987,negative_y,273,38,'field','raised');text(999,negative_y+24,'›  Negative prompt')
 control(controls_y,'Ratio','16:9',not sheet,sheet);control(controls_y+42,'Width' if sheet else 'Max side','1536' if sheet else '1600')
 text(1067,controls_y+104,'1536 × 864 px' if sheet else '1600 × 896 px',14,'dim');control(controls_y+122,'Seed','Random',True)
 text(1147,773,'Seed 24681357',11,'dim')
 line(272,726,957,726);btn(272,738,'generate',True);btn(310,738,'retry');btn(348,738,'stop',disabled=True)
 rect(398,751,398,5,'field',r=2);rect(398,751,398,5,'accent',r=2);text(810,759,'100%',14);text(862,759,'Saved',14,'dim')
 s.append('</svg>');return ''.join(s)
for name,sheet in [('character-sheet',True),('image-comparison',False)]:
 (P/(name+'.svg')).write_text(render(sheet))
