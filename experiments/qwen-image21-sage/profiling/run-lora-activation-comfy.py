"""Same headless workload, adding fixed-name host interval timings only."""
from pathlib import Path

source = Path(__file__).resolve().parents[1] / 'comfy-headless/compare-lora-cfg1.py'
text = source.read_text()
marker = '    active=None\n'
assert text.count(marker) == 1
instrumentation = '''    active=None
    # No additional CUDA synchronization and no argument/result serialization.
    phase_fd = os.open(OUT/'phases.tsv', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    def host_interval(obj, attr, label):
        original = getattr(obj, attr)
        @functools.wraps(original)
        def timed(*args, **kwargs):
            start = time.monotonic_ns()
            try:
                return original(*args, **kwargs)
            finally:
                end = time.monotonic_ns()
                os.write(phase_fd, f"{label}\\t{start}\\t{end}\\n".encode())
        setattr(obj, attr, timed)
    import comfy.model_management
    import comfy.model_patcher
    import comfy.sd
    import comfy.lora
    from comfy.weight_adapter.lora import LoRAAdapter
    from comfy_kitchen.tensor import QuantizedTensor
    host_interval(comfy.sd, 'load_lora_for_models', 'lora.attach')
    host_interval(comfy.model_management, 'load_models_gpu', 'model.load_gpu')
    host_interval(comfy.model_patcher.ModelPatcher, 'load', 'patcher.load')
    host_interval(comfy.model_patcher.ModelPatcher, 'patch_weight_to_device', 'patcher.weight')
    host_interval(comfy.lora, 'calculate_weight', 'lora.calculate')
    host_interval(LoRAAdapter, 'calculate_weight', 'lora.product_add')
    host_interval(QuantizedTensor, 'requantize_from_float', 'lora.requantize')
'''
text = text.replace(marker, instrumentation)
exec(compile(text, str(source), 'exec'), {'__file__': str(source), '__name__': '__main__'})
