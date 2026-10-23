// Opt-in source-only oracle. No Director config, browser, server or user media.
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import {execFileSync} from 'node:child_process';

const director='/home/thhel/git/comfyui-helto-director/web/timeline';
const own='/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e/ui-groups-01';
const destination=path.resolve(process.argv[2]??'');
if(!destination.startsWith(own+'/')||!destination.endsWith('.json'))throw Error('Expected a new own encrypted oracle file');
const mount=JSON.parse(execFileSync('/usr/bin/findmnt',['-J','-T',own],{encoding:'utf8'})).filesystems[0];
if(mount.target!=='/run/media/veracrypt1'||mount.source!=='/dev/mapper/veracrypt1'||mount.fstype!=='ext3'||!mount.options.split(',').includes('rw'))throw Error('Own encrypted destination is unavailable or read-only');
if(fs.realpathSync(path.dirname(destination))!==path.dirname(destination))throw Error('Expected a direct own destination');
const sources=['schema.js','geometry.js','operations.js'].map(name=>{
  const source=fs.readFileSync(path.join(director,name),'utf8');
  // Keep the actual pure source declarations, without evaluating module imports
  // that connect global_settings to browser/privacy infrastructure.
  return source.replace(/^import\s[\s\S]*?from\s+"[^"]+";\s*$/gm,'').replace(/^export\s+/gm,'');
});
const reference=vm.runInNewContext(sources.join('\n')+'\n({createDefaultVideoTimeline,setSelection,selectItem,toggleSelectItem,selectItemRange,getSelectedItemIds,insertShotAfterCurrent,addSection,resizeSection,duplicateSelectedSection,moveSelectedItems,hasDirectorSectionOverflow,canFitLastDirectorSectionToDuration,fitLastDirectorSectionToDuration,fitDirectorSectionsEvenlyToDuration});',{structuredClone,normalizeGlobalSettings:v=>v??{timeline:{minimum_section_duration_seconds:0.25}}},{timeout:3000});
function timeline(sections,duration=12){
  const t=reference.createDefaultVideoTimeline();t.project.duration_seconds=duration;t.ui_state.snap_mode='None';
  t.director_track.sections=sections.map(([a,b],i)=>({item_id:'own_section_'+i,start_time:a,end_time:b,prompt:'Own '+i,media_bindings:[]}));return t;
}
const positions=t=>t.director_track.sections.map(s=>[s.item_id,s.start_time,s.end_time]).sort((a,b)=>a[1]-b[1]||a[2]-b[2]);
const cases=[];
for(const [name,selected,playhead]of[['scene_add_selected_forward','own_scene_0',9],['scene_add_last_no_fallback','own_scene_2',0],['scene_add_playhead',null,5],['scene_add_earlier_fallback',null,12],['scene_add_section_owner','own_section_1',0]]){
 const t=timeline([[0,2],[4,6],[10,12]]);t.sequence.shots=t.director_track.sections.map((s,i)=>({shot_id:'own_scene_'+i,type:'Generated',start_time:s.start_time,end_time:s.end_time,section_ids:[s.item_id],takes:[]}));t.ui_state.playhead_time=playhead;if(selected)reference.selectItem(t,selected);const added=reference.insertShotAfterCurrent(t);
 cases.push({name,selected,playhead,added:added?[added.start_time,added.end_time,added.section_ids.length]:null});
}
for(const [name,actions]of[
 ['selection_mixed_toggle_primary',[['select','own_section_0'],['toggle','own_bare'],['toggle','own_audio_0'],['toggle','own_section_1'],['toggle','own_section_1']]],
 ['selection_shift_replaces_mixed',[['select','own_section_0'],['toggle','own_audio_0'],['range','own_section_2']]],
 ['selection_audio_range_lane_order',[['select','own_audio_2'],['range','own_audio_0']]],
 ['selection_bare_scene_range',[['select','own_scene_0'],['range','own_bare']]],
]){
 const t=timeline([[0,1],[3,4],[10,11]]);t.sequence.shots=t.director_track.sections.map((s,i)=>({shot_id:'own_scene_'+i,type:'Generated',start_time:s.start_time,end_time:s.end_time,section_ids:[s.item_id],takes:[]}));t.sequence.shots.push({shot_id:'own_bare',type:'Placeholder',start_time:6,end_time:7,section_ids:[],takes:[]});t.audio_tracks=[{track_id:'own_track',clips:[0,1,2].map(lane=>({item_id:'own_audio_'+lane,start_time:0,end_time:5-lane,lane}))}];
 for(const [operation,id]of actions)reference[operation==='select'?'selectItem':operation==='toggle'?'toggleSelectItem':'selectItemRange'](t,id);
 cases.push({name,actions,selection:reference.getSelectedItemIds(t)});
}
for(const [name,input,edge,value,mode]of[
 ['resize_trim_next',[[0,1],[1,4],[6,7]],'end',2,'Trim Neighbor'],
 ['resize_trim_previous',[[0,2],[2,4],[6,7]],'start',1,'Trim Neighbor'],
 ['resize_ripple_end',[[0,1],[3,4],[10,11]],'end',3,'Ripple Edit'],
 ['resize_ripple_start',[[0,1],[3,4],[10,11]],'start',2,'Ripple Edit'],
 ['resize_ripple_limit',[[0,1],[3,4],[10,11]],'end',100,'Ripple Edit'],
 ['resize_trim_gap',[[0,1],[3,4],[10,11]],'end',2,'Trim Neighbor'],
]){
 const t=timeline(input);t.ui_state.section_edit_mode=mode;
 const id='own_section_'+(edge==='start'?1:0);
 reference.resizeSection(t,id,edge,value,{globalSettings:{timeline:{minimum_section_duration_seconds:0.25}}});
 cases.push({name,input,id,edge,value,mode,sections:positions(t)});
}
for(const [name,input,owner,imported]of[
 ['add_standalone',[[0,1],[3,4],[10,11]],null,false],
 ['add_owner_extend',[[0,1],[3,4],[10,11]],0,false],
 ['add_owner_no_later_gap',[[0,1],[10,12]],1,false],
 ['add_imported_standalone',[[0,1],[3,4],[10,11]],0,true],
]){
 const t=timeline(input);
 if(owner!==null){
  t.sequence.shots=[{shot_id:'own_owner',type:imported?'Imported':'Generated',start_time:input[owner][0],end_time:input[owner][1],section_ids:['own_section_'+owner],takes:[]}];
  reference.setSelection(t,['own_section_'+owner]);
 }
 const added=reference.addSection(t);
 cases.push({name,input,owner,imported,added:added?[added.start_time,added.end_time]:null,owner_timing:t.sequence.shots.find(s=>s.shot_id==='own_owner')??null});
}
for(const [name,input,selected]of[
  ['duplicate_forward',[[0,1],[3,4],[10,11]],['own_section_0','own_section_1']],
  ['duplicate_earlier_fallback',[[4,6],[8,12]],['own_section_1']],
  ['duplicate_no_gap',[[0,6],[6,12]],['own_section_1']],
]){
  const t=timeline(input);reference.setSelection(t,selected);const copied=reference.duplicateSelectedSection(t)??[];
  cases.push({name,input,selected,copied:positions(t).filter(s=>copied.includes(s[0])).map(s=>s.slice(1)),new_shots:t.sequence.shots.map(s=>({start:s.start_time,end:s.end_time,model_mode:s.model_mode,takes:s.takes}))});
}
{
  const t=timeline([[0,1],[3,4],[10,11]]);
  t.audio_tracks=[{track_id:'own_audio_track',clips:[{item_id:'own_audio',start_time:1,end_time:2,locked:false},{item_id:'own_locked',start_time:5,end_time:6,locked:true}]}];
  reference.setSelection(t,['own_section_0','own_section_1','own_audio','own_locked']);reference.moveSelectedItems(t,'own_section_0',100);
  cases.push({name:'mixed_move_upper_bound',sections:positions(t),audio:t.audio_tracks.flatMap(t=>t.clips).map(c=>[c.item_id,c.start_time,c.end_time,c.lane])});
}
for(const [name,duration,all]of[['fit_all',8,true],['fit_last',10.5,false],['fit_last_unavailable',8,false],['fit_without_overflow',12,true]]){
  const t=timeline([[0,1],[3,4],[10,11]],duration);
  const overflow=reference.hasDirectorSectionOverflow(t),can_last=reference.canFitLastDirectorSectionToDuration(t);
  const changed=all?reference.fitDirectorSectionsEvenlyToDuration(t):reference.fitLastDirectorSectionToDuration(t);
  cases.push({name,duration,all,overflow,can_last,changed,sections:positions(t)});
}
// These pure renderer functions have no browser/runtime dependencies either.
const renderer=fs.readFileSync(path.join(director,'renderer.js'),'utf8');
const start=renderer.indexOf('export function waveformPeakCountForWidth('),end=renderer.indexOf('\nfunction sectionThumbnailUrl(',start);
if(start<0||end<start)throw Error('Director waveform source changed');
const functions=renderer.slice(start,end).replace(/^export\s+/gm,'');
const wave=vm.runInNewContext('const MIN_WAVEFORM_PEAKS=16,MAX_WAVEFORM_PEAKS=512;\n'+functions+'\n({waveformPeakCountForWidth,waveformPeakRequestForClip,waveformPeaksForClip});\nfunction clampNumber(v,min,max,fallback){v=Number(v);return Number.isFinite(v)?Math.max(min,Math.min(max,v)):fallback;}',{}, {timeout:3000});
const clip={start_time:0,end_time:1,source_in:5,source_out:7.5,volume:50};const payload={duration_seconds:10,peaks:Array.from({length:10},(_,i)=>i/10)};
cases.push({name:'waveform_crop_volume',clip,payload,requested:wave.waveformPeakRequestForClip(49,clip,10),bars:wave.waveformPeaksForClip(payload,clip,4),width_counts:[1,49,9999].map(w=>wave.waveformPeakCountForWidth(w))});
fs.writeFileSync(destination,JSON.stringify({source:'Actual Director pure source, isolated VM; no browser/config/runtime',cases},null,2),{flag:'wx'});
console.log(JSON.stringify({destination,cases:cases.length}));
