import os
import json
import numpy as np
from typing import List

DEBUG_PORT = int(os.environ.get("DEBUG_PORT", 9556))
local_rank = int(os.environ.get("LOCAL_RANK", 0))


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(
        key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu()
                 for k, v in to_return.items()}
    return to_return


def enter_debug_mode(enable=False):
    if enable and local_rank == 0:
        import debugpy
        try:
            debugpy.listen(("localhost", DEBUG_PORT))
            print("Waiting for debugger attach")
            debugpy.wait_for_client()
        except Exception as e:
            raise e


def require_config_keys(required_keys: List[str]):
    def decorator(func):
        def wrapper(config, *args, **kwargs):
            missing = [k for k in required_keys if not hasattr(config, k)]
            if missing:
                raise ValueError(f"Missing required config keys: {missing}")
            return func(config, *args, **kwargs)
        return wrapper
    return decorator


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert ndarray to list
        return super().default(obj)
