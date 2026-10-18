"""Explicit development reference selection; immutable snapshots stay separate."""
import json
import os
from pathlib import Path


def reference_dir(project):
    project = Path(project)
    override = os.environ.get('IMAGINE_REFERENCE_DIR')
    if override:
        root = Path(override)
    else:
        selection = project / 'design/reference-current.json'
        if not selection.is_file():
            return project / 'design/reference-local'
        relative = Path(json.loads(selection.read_text())['directory'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Reference directory must be relative to design/')
        root = project / 'design' / relative
    if not (root / 'reference-lock.json').is_file():
        raise FileNotFoundError(f'Missing explicit reference lock: {root}')
    return root
