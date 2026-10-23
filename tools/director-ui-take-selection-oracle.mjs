// Source-only A/B selection oracle: no browser, models, runtime or user files.
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import {execFileSync} from 'node:child_process';

const directory='/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e/ui-groups-01/evidence';
const destination=path.resolve(process.argv[2]??'');
if(path.dirname(destination)!==directory||!destination.endsWith('.json'))throw Error('Expected a new own encrypted evidence JSON');
const mount=JSON.parse(execFileSync('/usr/bin/findmnt',['-J','-T',directory],{encoding:'utf8'})).filesystems[0];
if(mount.target!=='/run/media/veracrypt1'||mount.source!=='/dev/mapper/veracrypt1'||mount.fstype!=='ext3'||!mount.options.split(',').includes('rw'))throw Error('Own encrypted evidence destination is unavailable or read-only');
if(fs.realpathSync(directory)!==directory)throw Error('Expected direct own evidence destination');

const source=fs.readFileSync('/home/thhel/git/comfyui-helto-director/web/timeline/take_review.js','utf8');
const start=source.indexOf('export function defaultTakeReviewSelection(');
const end=source.indexOf('\nexport function takeReviewMetadataRows(',start);
const valid=source.match(/^function validTake\(take\) \{[\s\S]*?^\}/m)?.[0];
if(start<0||end<start||!valid)throw Error('Director pure selection source changed');
const functions=source.slice(start,end).replace(/^export\s+/gm,'');
const reference=vm.runInNewContext(functions+'\n'+valid+'\n({defaultTakeReviewSelection,normalizeTakeReviewSelection,updateTakeReviewSelection});',{}, {timeout:3000});
const cases=[
  {name:'empty',ids:[],accepted:null,selection:[null,null]},
  {name:'single',ids:['own_a'],accepted:'own_a',selection:['missing','missing']},
  {name:'accepted_and_later_rejected',ids:['own_a','own_b','own_c'],accepted:'own_a',statuses:['Accepted','Candidate','Rejected'],selection:[null,null]},
  {name:'accepted_is_latest',ids:['own_a','own_b','own_c'],accepted:'own_c',selection:[null,null]},
  {name:'append_order_not_timestamps',ids:['own_a','own_b','own_c'],accepted:'missing',timestamps:[30,20,10],selection:[null,null]},
  {name:'explicit_valid_pair',ids:['own_a','own_b','own_c'],accepted:'own_b',selection:['own_c','own_a']},
  {name:'stale_side_repaired',ids:['own_a','own_b','own_c'],accepted:'own_a',selection:['missing','own_b']},
  {name:'duplicate_selection_repaired',ids:['own_a','own_b','own_c'],accepted:null,selection:['own_b','own_b']},
  {name:'change_a_swaps',ids:['own_a','own_b','own_c'],accepted:null,selection:['own_a','own_c'],change:{side:0,id:'own_c'}},
  {name:'change_b_swaps',ids:['own_a','own_b','own_c'],accepted:null,selection:['own_a','own_c'],change:{side:1,id:'own_a'}},
  {name:'single_change_b',ids:['own_a'],accepted:null,selection:[null,null],change:{side:1,id:'own_a'}},
  {name:'invalid_change',ids:['own_a','own_b'],accepted:null,selection:['missing','own_b'],change:{side:2,id:'missing'}},
];
const pair=value=>[value.aId||null,value.bId||null];
for(const row of cases){
  const shot={takes:row.ids.map((id,i)=>({take_id:id,status:row.statuses?.[i]??'Candidate',created_at:row.timestamps?.[i]??i})),accepted_take_id:row.accepted};
  const selection={aId:row.selection[0],bId:row.selection[1]};
  row.default=pair(reference.defaultTakeReviewSelection(shot));
  row.normalized=pair(reference.normalizeTakeReviewSelection(shot,selection));
  if(row.change)row.updated=pair(reference.updateTakeReviewSelection(shot,selection,['A','B'][row.change.side]??'invalid',row.change.id));
}
fs.writeFileSync(destination,JSON.stringify({source:'Actual Director take_review.js pure source executed in an isolated VM',cases},null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({destination,cases:cases.length}));
