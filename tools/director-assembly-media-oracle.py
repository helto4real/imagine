#!/usr/bin/env python3
"""Opt-in whole raw montage oracle. Only new procedural media in the owned test root."""
import ast
import collections
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import typing
import uuid

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import av
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dsp_oracle", ROOT / "tools/director-assembly-oracle.py")
dsp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dsp)


def fraction(n, d=1):
    return {"num": n, "den": d}


def seconds(t):
    return t["num"] / t["den"]


def selected_functions(own):
    def resolver(value):
        path = Path(value).resolve()
        assert path.is_relative_to(own)
        return path
    media = dsp.pure(dsp.DIRECTOR / "media.py", ["decode_video_frames", "_stream_fps", "trim_video_source_frames", "_safe_float"], lambda n: False, {"av": av, "np": np, "resolve_media_path": resolver, "DecodedVideo": collections.namedtuple("DecodedVideo", "frames frame_rate frame_count"), "SelectedVideoFrames": collections.namedtuple("SelectedVideoFrames", "frames metadata")})
    audio = dsp.pure(dsp.DIRECTOR / "audio.py", ["decode_audio_file", "_resampled_tensors", "mix_audio_clips", "trim_audio", "_fit_waveform_samples", "apply_gain_and_fades", "normalize_audio_waveform", "_safe_float", "volume_to_gain", "mix_audio_values", "_audio_value_waveform", "_resample_waveform", "_pad_audio"], lambda n: n.startswith("AUDIO_NORMALIZE_") or n == "TARGET_SAMPLE_RATE", {"av": av, "resolve_media_path": resolver})
    _, sequence = dsp.source_functions()
    constants = {}
    for node in ast.parse((dsp.DIRECTOR / "contracts/video_timeline.py").read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {"BOUNDARY_MODE_HARD_CUT", "BOUNDARY_MODE_CONTINUOUS_SHOT", "BOUNDARY_MODE_BLEND_SEAM", "BOUNDARY_MODE_TRANSITION"} for t in node.targets):
            constants[node.targets[0].id] = ast.literal_eval(node.value)
    timing = dsp.pure(dsp.DIRECTOR / "timeline/time_mapping.py", ["seconds_to_frame", "time_range_to_frames", "_validate_frame_rate"], lambda n: False)
    sequence.update(dsp.pure(dsp.DIRECTOR / "timeline/sequence_assembly.py", ["_append_with_boundary", "_copy_frames_into", "_can_blend", "_boundary_overlap_frame_count", "_add_warning", "_assembly_audio_time_ranges", "_map_timeline_audio_entry", "_safe_float"], lambda n: False, {**sequence, **constants, "deepcopy": copy.deepcopy, "time_range_to_frames": timing["time_range_to_frames"]}))
    return media, audio, sequence


def make_video(path, w, h, count, fps, seed, vfr=False):
    frames = np.fromfunction(lambda f,y,x,c: (x*13+y*7+c*41+f*17+seed)%256, (count,h,w,3), dtype=int).astype(np.uint8)
    command = ["/usr/bin/ffmpeg", "-v", "error", "-threads", "1", "-filter_threads", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-framerate", str(fps), "-i", "pipe:0", "-f", "lavfi", "-i", f"aevalsrc=0.08*sin(2*PI*{220+seed}*t)|0.06*sin(2*PI*{330+seed}*t):s=44100", "-c:v", "libx264", "-crf", "19", "-pix_fmt", "yuv420p", "-vf", "scale=out_color_matrix=bt709", "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-c:a", "aac", "-t", str(count/fps), str(path)]
    if vfr:
        command[command.index("-vf")+1] += ",setpts=if(lt(N\\,12)\\,N/(24*TB)\\,(0.5+(N-12)/12)/TB)"
        command[command.index("-t")+1] = "2"
        command[-1:-1] = ["-fps_mode", "vfr"]
    subprocess.run(command, input=frames.tobytes(), check=True)


def make_project(case, config):
    project_dir = case / "project"
    (project_dir / "media").mkdir(parents=True)
    model = {"adapter":"unused", "model":"unused", "variant":"unused", "profile":"unused", "profile_revision":"1"}
    project = {"schema_version":1,"id":"project_"+uuid.uuid4().hex,"project_revision":0,"default_model":model,"delivery":{"width":80,"height":64,"fps":config.get("fps",fraction(24))},"global_prompt":"Own synthetic oracle","assets":{},"scenes":[],"audio_tracks":[],"assembly":{"clips":{},"boundaries":[]}}
    resources = []
    paths = {}
    def attach(name, path, kind):
        data=path.read_bytes();digest=hashlib.sha256(data).hexdigest();target=project_dir/"media"/digest;target.write_bytes(data)
        project["assets"][name]={"id":name,"revision":1,"kind":kind}
        resources.append({"id":"resource_"+name,"kind":{"kind":"reference","asset_id":name,"revision":1},"sha256":digest,"bytes":len(data)})
        paths[name]=target
    source_a=case/("source_a."+config.get("format","mp4"))
    make_video(source_a,64,40,config.get("source_a_frames",24),24,0,config.get("vfr",False))
    make_video(case/"source_b.mp4",32,48,30,30,37)
    attach("video_a",source_a,"video");attach("video_b",case/"source_b.mp4","video")
    for i,duration in enumerate(config["durations"]):
        scene_id=f"scene_{i}"
        project["scenes"].append({"id":scene_id,"start":fraction(i*2),"duration":duration,"description":"Own synthetic scene","model":"project_default","settings":[],"sections":[],"bindings":[],"continuity":None,"accepted_take":None})
        project["assembly"]["clips"][scene_id]={"asset":"video_a" if i%2==0 else "video_b","source_in":config.get("source_in",fraction(0)),"source_out":None,"speed":1.,"include_audio":True}
        if i:
            mode,overlap=config["boundaries"][i-1]
            project["assembly"]["boundaries"].append({"left":f"scene_{i-1}","right":scene_id,"mode":mode,"blend_frames":overlap})
    if config.get("timeline"):
        wav=case/"source_timeline.wav"
        subprocess.run(["/usr/bin/ffmpeg","-v","error","-f","lavfi","-i","aevalsrc=0.03*sin(2*PI*880*t)|0.04*sin(2*PI*990*t):s=44100","-t","7","-c:a","pcm_f32le",str(wav)],check=True)
        attach("audio_timeline",wav,"audio")
        project["audio_tracks"]=[{"id":"audio_track","clips":[{"id":"audio_0","asset":"audio_timeline","start":fraction(1,8),"duration":fraction(5),"source_in":fraction(5,24),"source_out":None,"volume":73.,"normalization":{},"fade_in":0.03,"fade_out":0.05,"enabled":True,"locked":False,"name":"Own stereo marker","lane":0,"usage":{"generation_reference":False,"final_mix":True},"reference":{"purpose":"ambience_sfx","target_reference_id":None,"relationship":"reference","description":"","strength":1.}}]}]
    if config.get("clip_normalization"):
        for track in project["audio_tracks"]:
            for clip in track["clips"]: clip["normalization"]=config["clip_normalization"]
    if config.get("video_soundtrack"):
        usage=config["video_soundtrack"]
        project['creative']={'references':[],'sections':{}}
        for i,scene in enumerate(project['scenes']):
            section_id='section_'+str(i)
            section={'id':section_id,'start':fraction(1,8),'duration':fraction(1,2),'prompt':'Own soundtrack region'}
            scene['sections']=[section]
            binding={'id':'soundtrack_'+str(i),'asset':'video_b','role':'video_reference','enabled':True,'video':{'role':'video_reference','source_in':fraction(1,10),'source_out':fraction(3,5),'soundtrack_usage':usage}}
            project['creative']['sections'][section_id]={'images':[],'videos':[binding]}
    document={"schema_version":1,"project":project,"resources":resources,"takes":[],"origin":None}
    (project_dir/"project.json").write_text(json.dumps(document))
    options={"audio_mode":config.get("mode","combined"),"generated_gain":0.7,"timeline_gain":1.3,"duck_generated":config.get("duck",False),"ducking_strength":0.6,"normalize_mix":config.get("normalize",False),"continuous_tone_match":True,"bit_depth":config.get("depth",8)}
    (case/"options.json").write_text(json.dumps(options))
    return project,paths,options


def reference(project,paths,options,media,audio,sequence):
    fps=seconds(project["delivery"]["fps"]);entries=[];fitted=[];cursor=0
    modes={"hard_cut":"Hard Cut","continuous_shot":"Continuous Shot","blend_seam":"Blend Seam"}
    for i,scene in enumerate(project["scenes"]):
        clip=project["assembly"]["clips"][scene["id"]];decoded=media["decode_video_frames"](str(paths[clip["asset"]]))
        selected=media["trim_video_source_frames"](decoded.frames,decoded.frame_rate,{"source_in":seconds(clip["source_in"]),"source_out":None if clip["source_out"] is None else seconds(clip["source_out"])})
        frames=sequence["_fit_frames_to_count"](selected.frames,max(1,round(seconds(scene["duration"])*fps)))
        if frames.shape[2]!=project["delivery"]["width"] or frames.shape[1]!=project["delivery"]["height"]:
            frames=sequence["resize"](frames,project["delivery"]["width"],project["delivery"]["height"],"Pad",1)
        boundary=project["assembly"]["boundaries"][i-1] if i else None
        boundary=None if boundary is None else dict(boundary,mode=modes[boundary["mode"]])
        overlap=sequence["_boundary_overlap_frame_count"](cursor,frames.shape[0],boundary)
        start=cursor-overlap;cursor=start+frames.shape[0]
        if entries:entries[-1]["audio_fade_out_frames"]=overlap
        entries.append({"shot_id":scene["id"],"timeline_start_time":seconds(scene["start"]),"timeline_end_time":seconds(scene["start"])+seconds(scene["duration"]),"output_start_frame":start,"output_end_frame_exclusive":cursor,"audio_fade_in_frames":overlap,"audio_fade_out_frames":0,"source_range":selected.metadata["source_range"],"path":str(paths[clip["asset"]]),"boundary":boundary})
        fitted.append(frames)
    out=torch.empty((sum(f.shape[0] for f in fitted),project["delivery"]["height"],project["delivery"]["width"],3),dtype=torch.float32);cursor=0;debug={"boundaries":[],"warnings":[]}
    for i,frames in enumerate(fitted):
        if i==0:out[:len(frames)]=frames;cursor=len(frames)
        else:cursor=sequence["_append_with_boundary"](out,cursor,frames,entries[i]["boundary"],entries[i-1],entries[i],debug,frame_rate=fps,continuous_tone_match=options["continuous_tone_match"])
    out=out[:cursor];duration=cursor/fps
    source_plan=[]
    for entry in entries:
        source_plan.append({"path":entry["path"],"start_time":entry["output_start_frame"]/fps,"end_time":entry["output_end_frame_exclusive"]/fps,"source_in":entry["source_range"]["start"],"source_out":entry["source_range"]["end"],"volume":100.,"fade_in":entry["audio_fade_in_frames"]/fps,"fade_out":entry["audio_fade_out_frames"]/fps,"fit_to_timeline_duration":True})
    timeline_plan=[];ranges=sequence["_assembly_audio_time_ranges"](entries,fps)
    for track in project["audio_tracks"]:
        for clip in track["clips"]:
            timeline_plan.extend(sequence["_map_timeline_audio_entry"]({"item_id":clip["id"],"path":str(paths[clip["asset"]]),"start_time":seconds(clip["start"]),"end_time":seconds(clip["start"])+seconds(clip["duration"]),"source_in":seconds(clip["source_in"]),"source_out":None,"volume":clip["volume"],"fade_in":clip["fade_in"],"fade_out":clip["fade_out"]},ranges,fps))
    if project.get('creative'):
        # Actual Director planner builds source entries; actual sequence code maps them.
        planner=dsp.pure(dsp.DIRECTOR/'h3/planner.py',['_build_audio_plan','_resolve_asset','_asset_path'],lambda n:False,{'ASSET_TYPE_VIDEO':'Video','VIDEO_SOUNDTRACK_FINAL_MIX':'final_mix','VIDEO_SOUNDTRACK_BOTH':'both'})
        sections=[]
        for scene in project['scenes']:
            for section in scene['sections']:
                source=project['creative']['sections'][section['id']]
                bindings=[{'binding_id':b['id'],'asset_id':b['asset'],'media_type':'Video',**b['video'],'source_in':seconds(b['video']['source_in']),'source_out':seconds(b['video']['source_out'])}for b in source['videos']]
                start=seconds(scene['start'])+seconds(section['start'])
                sections.append({'item_id':section['id'],'start_time':start,'end_time':start+seconds(section['duration']),'media_bindings':bindings})
        timeline={'assets':[{'asset_id':key,'path':str(path)}for key,path in paths.items()],'audio_tracks':[],'director_track':{'sections':sections}}
        for entry in planner['_build_audio_plan'](timeline):
            timeline_plan.extend(sequence['_map_timeline_audio_entry'](entry,ranges,fps))
    empty={"waveform":torch.zeros((1,2,max(1,math.ceil(duration*44100)))),"sample_rate":44100}
    generated=empty if options["audio_mode"]=="timeline" else audio["mix_audio_clips"](source_plan,duration,normalize=False)[0]
    timeline=empty if options["audio_mode"]=="generated" else audio["mix_audio_clips"](timeline_plan,duration,normalize=False)[0]
    pcm=audio["mix_audio_values"](generated,timeline,generated_gain=options["generated_gain"],timeline_gain=options["timeline_gain"],duck_generated=options["duck_generated"],ducking_strength=options["ducking_strength"],normalize=options["normalize_mix"])["waveform"][0].T.contiguous().numpy()
    scale=65535 if options["bit_depth"]==10 else 255
    rgb=np.clip(out.numpy()*scale+0.5,0,scale).astype('<u2' if options["bit_depth"]==10 else np.uint8)
    return rgb,pcm,timeline_plan,out.numpy()


def main():
    own=Path(sys.argv[1]);assert own.is_relative_to(dsp.OWN) and own.resolve()==own
    mount=json.loads(subprocess.check_output(["findmnt","-J","-T",str(dsp.OWN)]))["filesystems"][0]
    assert (mount["target"],mount["source"],mount["fstype"])==('/run/media/veracrypt1','/dev/mapper/veracrypt1','ext3') and not(os.statvfs(dsp.OWN).f_flag&os.ST_RDONLY)
    own.mkdir(exist_ok=False);torch.set_num_threads(1);media,audio,sequence=selected_functions(own)
    configs=[
        {"name":"cross_clip_blend","durations":[fraction(1,2),fraction(1,24),fraction(1)],"boundaries":[("hard_cut",0),("blend_seam",6)],"timeline":True},
        {"name":"short_continuous","durations":[fraction(1,2),fraction(1,24),fraction(1)],"boundaries":[("hard_cut",0),("continuous_shot",0)],"timeline":True,"duck":True},
        {"name":"whole_clip_blend","durations":[fraction(1),fraction(1),fraction(1,2)],"boundaries":[("blend_seam",24),("blend_seam",6)],"timeline":True},
        {"name":"fractional_fps","durations":[fraction(7,10),fraction(9,10)],"boundaries":[("blend_seam",3)],"fps":fraction(30000,1001),"timeline":True,"normalize":True},
        {"name":"depth10_30fps","durations":[fraction(2,5),fraction(3,5)],"boundaries":[("blend_seam",4)],"fps":fraction(30),"depth":10,"mode":"generated"},
        {"name":"half_sample_source","durations":[fraction(1,3)],"boundaries":[],"source_a_frames":5,"timeline":True,"mode":"timeline"},
        {"name":"matroska_source","durations":[fraction(4,5)],"boundaries":[],"format":"mkv","timeline":True},
        {"name":"vfr_source","durations":[fraction(6,5)],"boundaries":[],"vfr":True,"timeline":True},
        {"name":"clip_peak_override","durations":[fraction(1)],"boundaries":[],"timeline":True,"clip_normalization":{"normalization_mode":"Peak","true_peak_limit_db":-3.}},
        {"name":"clip_lufs_override","durations":[fraction(1)],"boundaries":[],"timeline":True,"clip_normalization":{"normalization_mode":"Integrated LUFS","target_lufs":-20.,"true_peak_limit_db":-2.}},
    ]
    configs=[{'name':'video_soundtrack_'+usage+'_'+mode,'durations':[fraction(1),fraction(1)],'boundaries':[('blend_seam',3)],'video_soundtrack':usage,'mode':mode,'timeline':mode=='combined'}for usage,mode in [('final_mix','timeline'),('both','combined'),('reference','timeline'),('ignore','combined')]] if '--video-only'in sys.argv else configs
    report=[]
    for config in configs:
        case=own/config["name"];case.mkdir();project,paths,options=make_project(case,config)
        probe=ROOT/"comfyui/h3-video-night-20261021/target/debug/examples/assembly_media_probe"
        result=subprocess.run([str(probe),str(case)],capture_output=True,text=True);(case/"probe-stderr.txt").write_text(result.stderr)
        if result.returncode:report.append({"name":config["name"],"error":result.stderr,"passed":False});continue
        rgb,pcm,timeline_plan,raw=reference(project,paths,options,media,audio,sequence)
        rgb.tofile(case/"python-rgb.raw");pcm.astype('<f4').tofile(case/"python-pcm.raw");(case/"python-audio-plan.json").write_text(json.dumps(timeline_plan,indent=2))
        raw.astype('<f4').tofile(case/'python-rgb-f32.raw')
        rust_raw=np.fromfile(case/'rust-rgb-f32.raw',dtype='<f4')
        raw_max=float(abs(rust_raw-raw.ravel()).max()) if rust_raw.size==raw.size else math.inf
        rust_rgb=np.fromfile(case/"rust-rgb.raw",dtype=rgb.dtype);rust_pcm=np.fromfile(case/"rust-pcm.raw",dtype='<f4')
        rgb_equal=rust_rgb.size==rgb.size and np.array_equal(rust_rgb,rgb.ravel());pcm_equal_shape=rust_pcm.size==pcm.size
        difference=rust_pcm.astype(np.float64)-pcm.ravel().astype(np.float64) if pcm_equal_shape else np.array([math.inf])
        maximum=float(abs(difference).max());metadata=json.loads(subprocess.check_output(['/usr/bin/ffprobe','-v','error','-count_frames','-show_entries','format=duration:stream=codec_name,pix_fmt,nb_read_frames,sample_rate,channels,duration','-of','json',str(case/'delivery.mp4')]))
        integer_error=int(abs(rust_rgb.astype(np.int64)-rgb.ravel().astype(np.int64)).max()) if rust_rgb.size==rgb.size else None
        report.append({"name":config["name"],"rgb_byteidentical":rgb_equal,"rgb_f32_max":raw_max,"rgb_max_integer_error":integer_error,"frames":rgb.shape[0],"samples_per_channel":pcm.shape[0],"pcm_max":maximum,"pcm_mae":float(abs(difference).mean()),"pcm_rmse":float(np.sqrt(np.mean(difference*difference))),"probe":metadata,"passed":raw_max<=2e-6 and integer_error is not None and integer_error<=1 and pcm_equal_shape and maximum<=2e-6})
        print(json.dumps(report[-1]),flush=True)
    (own/'report.json').write_text(json.dumps({'cases':report,'passed':all(r['passed'] for r in report),'cpu_only':True,'director_import':'AST pure functions only; restricted own-media resolver'},indent=2)+'\n')
    raise SystemExit(0 if all(r['passed'] for r in report) else 1)


if __name__=='__main__':main()
