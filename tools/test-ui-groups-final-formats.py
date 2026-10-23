#!/usr/bin/env python3
"""Two focused Final oracles on procedural media, without jobs or model output.

Uses permitted Director pure functions and the actual Rust freeze/prepare/render
example. All fixture documents and bytes are created within this new case.
No runtime job database, prompts, logs or previously generated media is read.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / 'comfyui/ui-groups-20261021'
OWN = Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    assert len(sys.argv) == 2 and sys.argv[1].isascii() and sys.argv[1].isalnum()
    mount = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(OWN)]))['filesystems'][0]
    assert (mount['target'], mount['source'], mount['fstype']) == ('/home', '/dev/mapper/cryptroot[/@home]', 'btrfs')
    assert not os.statvfs(OWN).f_flag & os.ST_RDONLY
    own = OWN / sys.argv[1]
    own.mkdir()
    (own / 'tmp').mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               TMPDIR=str(own/'tmp'), CARGO_TARGET_DIR=str(C/'target'),
               CARGO_BUILD_JOBS='2', CARGO_INCREMENTAL='0',
               CARGO_PROFILE_DEV_DEBUG='0', CARGO_PROFILE_TEST_DEBUG='0')
    diagnostics = own/'compiler.txt'
    with diagnostics.open('x') as output:
        build = subprocess.run(['cargo', 'build', '--offline', '--locked', '--example', 'assembly_media_probe'],
                               cwd=C/'source', env=env, stdout=output, stderr=subprocess.STDOUT)
    assert build.returncode == 0, str(diagnostics)
    oracle = load('final_oracle', ROOT/'tools/director-assembly-media-oracle.py')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    media, audio, sequence = oracle.selected_functions(own)
    f = oracle.fraction
    cases = [
        dict(name='mov24_mp4_30_to_fractional', format='mov', durations=[f(7, 10), f(9, 10)],
             boundaries=[('continuous_shot', 0)], fps=f(30000, 1001), timeline=True),
        dict(name='vfr_mkv_mp4_30', format='mkv', vfr=True, durations=[f(6, 5), f(4, 5)],
             boundaries=[('blend_seam', 3)], timeline=True),
    ]
    report = dict(passed=False, cpu_only=True, model_generated_media=False,
                  runtime_logs_jobs_or_submitted_prompts_read=False, cases=[])
    for config in cases:
        case = own/config['name']
        case.mkdir()
        project, paths, options = oracle.make_project(case, config)
        # These are fixture documents made above, not existing runtime data.
        result = subprocess.run([str(C/'target/debug/examples/assembly_media_probe'), str(case)],
                                env=env, capture_output=True, text=True)
        (case/'component-diagnostics.txt').write_text(result.stderr)
        assert result.returncode == 0, str(case/'component-diagnostics.txt')
        rgb, pcm, _, raw = oracle.reference(project, paths, options, media, audio, sequence)
        rust_raw = np.fromfile(case/'rust-rgb-f32.raw', dtype='<f4')
        rust_rgb = np.fromfile(case/'rust-rgb.raw', dtype=rgb.dtype)
        rust_pcm = np.fromfile(case/'rust-pcm.raw', dtype='<f4')
        assert rust_raw.size == raw.size and rust_rgb.size == rgb.size and rust_pcm.size == pcm.size
        delta_rgb = rust_raw.astype(np.float64)-raw.ravel().astype(np.float64)
        delta_pcm = rust_pcm.astype(np.float64)-pcm.ravel().astype(np.float64)
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
            '-show_entries', 'format=duration:stream=codec_type,codec_name,pix_fmt,nb_read_frames,sample_rate,channels,color_range,color_space,color_transfer,color_primaries',
            '-of', 'json', str(case/'delivery.mp4')], env=env))
        video = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        sound = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
        expected_seconds = rgb.shape[0]/oracle.seconds(project['delivery']['fps'])
        row = dict(name=config['name'], frames=rgb.shape[0], samples_per_channel=pcm.shape[0],
                   rgb_f32_mae=float(abs(delta_rgb).mean()), rgb_f32_rmse=float(np.sqrt(np.mean(delta_rgb**2))),
                   rgb_f32_max=float(abs(delta_rgb).max()),
                   rgb_integer_max=int(abs(rust_rgb.astype(np.int64)-rgb.ravel().astype(np.int64)).max()),
                   pcm_mae=float(abs(delta_pcm).mean()), pcm_rmse=float(np.sqrt(np.mean(delta_pcm**2))),
                   pcm_max=float(abs(delta_pcm).max()), pcm_rms=float(np.sqrt(np.mean(rust_pcm.astype(np.float64)**2))),
                   expected_seconds=expected_seconds, probe=probe)
        row['passed'] = (row['rgb_f32_max'] <= 2e-6 and row['rgb_integer_max'] <= 1 and row['pcm_max'] <= 2e-6
                         and video['codec_name'] == 'h264' and video['pix_fmt'] == 'yuv420p'
                         and int(video['nb_read_frames']) == rgb.shape[0]
                         and sound['codec_name'] == 'aac' and sound['channels'] == 2 and sound['sample_rate'] == '44100'
                         and video['color_space'] == video['color_transfer'] == video['color_primaries'] == 'bt709'
                         and video['color_range'] == 'tv'
                         and abs(float(probe['format']['duration'])-expected_seconds) < .002)
        report['cases'].append(row)
        print(json.dumps(row), flush=True)
    report['passed'] = all(r['passed'] for r in report['cases'])
    with (own/'component-result.json').open('x') as output:
        json.dump(report, output, indent=2)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
