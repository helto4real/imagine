#!/usr/bin/env python3
"""CPU-only viewing export of explicit own H3 technical-probe tensor outputs."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--site', type=Path, required=True)
p.add_argument('--own-root', type=Path, required=True)
p.add_argument('--run', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
project = Path(__file__).resolve().parents[2]
assert a.own_root.resolve().is_relative_to(project / 'comfyui/h3-stage4-20261020')
assert a.run.resolve().is_relative_to(a.own_root.resolve())
assert a.output.resolve().is_relative_to(a.own_root.resolve())
assert not a.output.exists()
sys.path.insert(0, str(a.site))
import numpy as np


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor(phase, name, shape):
    root = a.run / phase
    report = json.loads((root / 'phase.json').read_text())
    assert report.get('result', 'passed') == 'passed'
    metadata = json.loads((root / (name + '.json')).read_text())
    path = root / (name + '.bin')
    assert sha(path) == metadata['sha256'] and path.stat().st_size == metadata['bytes']
    assert metadata['shape'] == shape
    dtype = {'float16': '<f2', 'float32': '<f4'}[metadata['dtype']]
    data = np.fromfile(path, dtype=dtype).reshape(shape)
    assert np.isfinite(data).all(), 'Do not export nonfinite probe output'
    return data, dict(path=str(path), **metadata)


frames, video_record = tensor('video', 'frames', [1, 3, 39, 256, 384])
pcm, audio_record = tensor('audio', 'pcm', [1, 2, 52000])
# The selected original video VAE already finalizes to [0,1]. This conversion
# and lossy codec are viewing-only, outside exact tensor acceptance.
rgb = np.rint(np.clip(frames[0].transpose(1, 2, 3, 0).astype(np.float32), 0, 1) * 255).astype(np.uint8)
audio_path = a.output.with_suffix('.stereo-f32le')
pcm[0].T.astype('<f4').tofile(audio_path)
command = ['/usr/bin/ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
           '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', '384x256',
           '-framerate', '24', '-i', 'pipe:0', '-f', 'f32le', '-ar', '32000',
           '-ac', '2', '-i', str(audio_path), '-map', '0:v:0', '-map', '1:a:0',
           '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
           '-c:a', 'aac', '-b:a', '192k', '-t', '1.625', '-frames:v', '39',
           '-movflags', '+faststart', str(a.output)]
subprocess.run(command, input=rgb.tobytes(), check=True)
streams = json.loads(subprocess.check_output([
    '/usr/bin/ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(a.output)]))['streams']
video = next(s for s in streams if s['codec_type'] == 'video')
audio = next(s for s in streams if s['codec_type'] == 'audio')
assert video['nb_frames'] == '39' and video['avg_frame_rate'] == '24/1'
assert audio['sample_rate'] == '32000' and audio['channels'] == 2
assert video['duration'] == audio['duration'] == '1.625000'
record = dict(schema=1, result='passed', purpose='Own technical viewing export; no quality acceptance',
              source_video=video_record, source_audio=audio_record, command=command,
              video=video, audio=audio, export_sha256=sha(a.output),
              ffmpeg_sha256=sha(Path('/usr/bin/ffmpeg')),
              ffprobe_sha256=sha(Path('/usr/bin/ffprobe')))
a.output.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
print('PASS own clip: 39 frames / 24 fps, stereo 32000 Hz, 1.625 seconds')
