"""Resource exhaustion is a failed oracle capture, never an expected model error."""

def reject_resource_failure(error):
    if 'out of memory' in str(error).lower():
        raise error

def approved_diagnostic(error, operation, kwargs):
    """Apply only the two recorded set-order criteria after full source checks."""
    reject_resource_failure(error)
    import torch
    import comfy_kitchen as ck
    from comfy_kitchen.exceptions import NoCapableBackendError
    if not isinstance(error, NoCapableBackendError):return str(error), []
    failures={};criteria=[]
    for backend,failure in error.failures.items():
        parameter,sep,reason=failure.partition(': ')
        constraints=ck.registry.get_constraints(backend,operation)
        constraint=constraints.params.get(parameter)if constraints else None
        value=kwargs.get(parameter)
        if sep and reason.startswith('dtype ')and constraint is not None and not constraint.check_dtype(value):
            rejected=value.dtype if isinstance(value,torch.Tensor)else value
            assert reason==f'dtype {rejected} not in {set(constraint.dtypes)}'
            names=sorted(repr(x)for x in set(constraint.dtypes))
            reason=f"dtype {rejected} not in {{{', '.join(names)}}}"if names else f'dtype {rejected} not in set()'
            criteria.append('D-05-REGISTRY-TYPE-SET-ORDER')
        elif sep and reason.startswith('device ')and isinstance(value,torch.Tensor):
            allowed=constraint.devices if constraint.devices is not None else constraints.default_devices
            assert reason==f'device {value.device.type} not in {set(allowed)}'
            assert all(name.isascii()and name.isidentifier()for name in allowed)
            names=sorted(repr(x)for x in set(allowed))
            reason=f"device {value.device.type} not in {{{', '.join(names)}}}"if names else f'device {value.device.type} not in set()'
            criteria.append('D-05-REGISTRY-DEVICE-SET-ORDER')
        failures[backend]=f'{parameter}: {reason}'if sep else failure
    return str(NoCapableBackendError(error.func_name,failures)),sorted(set(criteria))
