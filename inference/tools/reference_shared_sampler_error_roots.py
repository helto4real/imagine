#!/usr/bin/env python3
"""Original churn failures and physical traceback storage roots; own tensors."""
from pathlib import Path
loader = Path(__file__).with_name('reference_shared_solvers.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n', 1)[0], str(loader), 'exec'))

def capture(name, device, dtype, failure, index):
    probes = {}
    holder = [data((2, 1, 2, 3), device, dtype, 3)]
    sigmas = torch.tensor([.9, .6, .3, .1, 0.], device=device)
    calls = [0, 0]
    class Failing(Model):
        def __call__(self, x, sigma, **kwargs):
            current = calls[0]
            calls[0] += 1
            if failure == 'model' and current == index:
                raise ValueError('own retained sampler failure')
            return super().__call__(x, sigma, **kwargs)
    sampling = ns['EPS']()
    sampling.sigma_data = 1.
    model = Failing(sampling, [], False)
    def callback(values):
        current = calls[1]
        calls[1] += 1
        if failure == 'callback' and current == index:
            raise ValueError('own retained sampler failure')
    saved = None
    torch.manual_seed(17)
    try:
        ns['sample_' + name](model, holder.pop(), sigmas, callback=callback, disable=True)
    except ValueError as error:
        # Remove this harness frame, as an embedding caller can do. Retain all
        # original decorator/body/provider frames, without a self-owning harness.
        saved = error.with_traceback(error.__traceback__.tb_next)
    assert saved is not None
    def inspect(error):
        tb = error.__traceback__
        while tb.tb_frame.f_code.co_name != 'sample_' + name:
            tb = tb.tb_next
        fields = {}
        for key, value in tb.tb_frame.f_locals.items():
            if isinstance(value, torch.Tensor):
                fields[key] = stamp(value)
                probes[key] = value.untyped_storage()._weak_ref()
        return fields
    fields = inspect(saved)
    live = lambda: {key: not torch.UntypedStorage._expired(value) for key, value in probes.items()}
    before = live()
    saved.__traceback__ = None
    after = live()
    for value in probes.values(): torch.UntypedStorage._free_weak_ref(value)
    return dict(name=name, device=device, kind=str(dtype), failure=failure, index=index,
                calls=calls, fields=fields, retained=before, released=after,
                error=[type(saved).__name__, str(saved)])

cases = []
with torch.no_grad():
    for name in ['euler', 'heun', 'dpm_2', 'heunpp2']:
        for device in ['cpu', 'cuda:0']:
            for dtype in [torch.float32, torch.bfloat16]:
                for failure in ['model', 'callback']:
                    for index in [0, 1]:
                        cases.append(capture(name, device, dtype, failure, index))
a.output.write_text(json.dumps(dict(schema=1, sources=sources, trained_model_weights=0,
    max_latent_elements=12, cases=cases), separators=(',', ':')) + '\n')
print('Captured', len(cases), 'original churn traceback storage histories; no weights')
