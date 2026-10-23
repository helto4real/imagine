// Source-only default/type mutation oracle; no browser, services or runtime data.
import { SHOT_TYPES, ASSET_TYPE_VIDEO, createDefaultShot } from "/home/thhel/git/comfyui-helto-director/web/timeline/schema.js";
import { changeShotType, setClipInstanceFromAsset } from "/home/thhel/git/comfyui-helto-director/web/timeline/operations.js";
const shot=createDefaultShot(1);
const timeline={sequence:{shots:[shot]},assets:[{asset_id:"own-video",type:ASSET_TYPE_VIDEO}]};
if(!setClipInstanceFromAsset(timeline,shot.shot_id,"own-video"))throw Error("Own clip fixture failed");
const clip=JSON.stringify(shot.clip_instance);
const results=SHOT_TYPES.map(type=>{
 if(!changeShotType(timeline,shot.shot_id,type))throw Error("Own type fixture failed");
 return {type:shot.type,clip_retained:JSON.stringify(shot.clip_instance)===clip};
});
console.log(JSON.stringify({default_type:createDefaultShot(1).type,types:SHOT_TYPES,results},null,2));
