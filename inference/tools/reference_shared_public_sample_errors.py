#!/usr/bin/env python3
"""Original sampling entry traceback retains all own parameters and guider."""
from pathlib import Path
loader = Path(__file__).with_name('reference_shared_public_sample.py')
exec(compile(loader.read_text().split('\ndef run(', 1)[0], str(loader), 'exec'))
import weakref

def run(failure):
    probes = {}
    count = [0]
    class Token:
        def __init__(self, name):
            probes[name] = weakref.ref(self)
    def emit():
        index = count[0]
        count[0] += 1
        if index == failure:
            raise ValueError('own public retained failure')
    class Guider:
        def __init__(self, model):
            emit()
            probes['cfg_guider'] = weakref.ref(self)
        def __getattr__(self, name):
            emit()
            def call(*args):
                emit()
            return call
    space = dict(CFGGuider=Guider)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), space)
    def invoke():
        return space['sample'](**{name: Token(name) for name in
            ['model','noise','positive','negative','cfg','device','sampler','sigmas',
             'model_options','latent_image','denoise_mask','callback','disable_pbar','seed']})
    saved = None
    try:
        invoke()
    except ValueError as error:
        tb = error.__traceback__
        while tb.tb_frame.f_code is not space['sample'].__code__:
            tb = tb.tb_next
        saved = error.with_traceback(tb)
        tb = None
    assert saved is not None
    names = sorted(saved.__traceback__.tb_frame.f_locals)
    retained = {name: value() is not None for name, value in probes.items()}
    saved.__traceback__ = None
    released = {name: value() is not None for name, value in probes.items()}
    return dict(failure=failure, calls=count[0], names=names, retained=retained,
                released=released, error=[type(saved).__name__, str(saved)])

cases = [run(index) for index in range(7)]
assert 'torch' not in __import__('sys').modules
a.output.write_text(json.dumps(dict(schema=1, source_sha256=digest,
    trained_model_weights=0, fixture_tensors=0, cases=cases), separators=(',', ':')) + '\n')
print('Captured', len(cases), 'original public sample retained-error histories')
