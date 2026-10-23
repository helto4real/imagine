// Opt-in source-only comparison. No connected ComfyUI or runtime data is accessed.
import { createDefaultVideoTimeline } from "/home/thhel/git/comfyui-helto-director/web/timeline/schema.js";
import { addAudioClip, autoStackAudioLanes, resizeAudioClip } from "/home/thhel/git/comfyui-helto-director/web/timeline/operations.js";
const timeline=createDefaultVideoTimeline();
const clip=structuredClone(addAudioClip(timeline,0,1));
delete clip.item_id;
const clips=[clip];
for (const [start,duration] of [[0,2],[1,2],[2,1]]) addAudioClip(timeline,start,duration);
autoStackAudioLanes(timeline);
const stacking=timeline.audio_tracks.flatMap(t=>t.clips).map(c=>({start:c.start_time,end:c.end_time,lane:c.lane}));
const trimTimeline=createDefaultVideoTimeline();
const trimmed=addAudioClip(trimTimeline,1,3);
resizeAudioClip(trimTimeline,trimmed.item_id,"start",2);
resizeAudioClip(trimTimeline,trimmed.item_id,"end",5);
console.log(JSON.stringify({project:timeline.project.audio,clip:clips[0],stacking,trim:{start:trimmed.start_time,end:trimmed.end_time,source_in:trimmed.source_in,source_out:trimmed.source_out}},null,2));
