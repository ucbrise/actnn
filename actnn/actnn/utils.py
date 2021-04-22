import os
from collections import OrderedDict
import json

import torch
import numpy as np
import json


def swap_to_cpu(tensor):
    tensor_cpu = torch.empty(tensor.shape, dtype=tensor.dtype, device='cpu', pin_memory=True)
    tensor_cpu.copy_(tensor, non_blocking=True)
    return tensor_cpu


def get_memory_usage(print_info=False):
    """Get accurate gpu memory usage by querying torch runtime"""
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    if print_info:
        print("allocated: %.2f MB" % (allocated / 1024 / 1024), flush=True)
        print("reserved:  %.2f MB" % (reserved / 1024 / 1024), flush=True)
    return allocated


def compute_tensor_bytes(tensors):
    """Compute the bytes used by a list of tensors"""
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]

    ret = 0
    for x in tensors:
        if x.dtype in [torch.float32, torch.int]:
            ret += np.prod(x.size()) * 4 
        elif x.dtype in [torch.bfloat16, torch.float16, torch.int16]:
            ret += np.prod(x.size()) * 2
        elif x.dtype in [torch.int8]:
            ret += np.prod(x.size()) * 2

    return ret


def empty_cache(ratio):
    if ratio is None:
        return
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    if reserved > 0 and allocated / reserved < ratio:
        torch.cuda.empty_cache()


def disable_cache_allocator():
    os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'


def enable_cache_allocator():
    del os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING']


class GlobalExpRecorder:
    def __init__(self):
        self.val_dict = OrderedDict()

    def record(self, key, value, float_round=6):
        if isinstance(value, (np.int32, np.int64)):
            value = int(value)
        if isinstance(value, (float, np.float32, np.float64)):
            value = round(value, float_round)

        self.val_dict[key] = value

    def dump(self, filename):
        with open(filename, "a") as fout:
            fout.write(json.dumps(self.val_dict) + '\n')
        print("Save exp results to %s" % filename)

    def clear():
        pass

exp_recorder = GlobalExpRecorder()

