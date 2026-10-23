"""Authorized synthetic task relocation; historical evidence stays immutable."""
from pathlib import Path

OLD_PREFIX = Path('/run/media/veracrypt1/imagine')
TASK_PREFIX = Path('/home/thhel/tmp')
TASK_NAMES = frozenset([
    '.codex-video-night-20261020-a6428d9e',
    '.codex-h3-res20-compare-20261020-6f0b84c2',
    '.codex-h3-memory-20261021-c77f4a9b',
    '.codex-worker-a-g14-15-20261022',
    '.codex-k1-k2-worker-b-20261022-01',
    '.codex-g7-lora-worker-b-20261022-01',
    '.codex-controls-review-20261021-9b17e26a',
])


def relocated(path):
    path = Path(path)
    if path.is_relative_to(OLD_PREFIX):
        relative = path.relative_to(OLD_PREFIX)
        assert relative.parts and relative.parts[0] in TASK_NAMES, 'Not a known own task path'
        return TASK_PREFIX/relative
    return path
