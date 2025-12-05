# Some copy is inspired by https://github.com/vllm-project/vllm/blob/main/vllm/utils/mem_utils.py

# import contextlib
import gc

# import time
# from collections.abc import Generator
from dataclasses import dataclass, field
from functools import cache
from .mem_constants import GiB_bytes

# import psutil
import torch
# import torch.types

from .mem_constants import GiB_bytes

from dataclasses import dataclass, field


@dataclass
class MemoryProfilingResult:
    # GB
    free_memory: float = 0.0
    total_memory: float = 0.0
    torch_memory: float = 0.0


def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def memory_profiling() -> MemoryProfilingResult:
    clear_gpu_memory()
    free_memory, total_memory = torch.cuda.mem_get_info()
    torch_memory = torch.cuda.memory_reserved()
    result = MemoryProfilingResult(
        free_memory=free_memory / GiB_bytes,
        total_memory=total_memory / GiB_bytes,
        torch_memory=torch_memory / GiB_bytes,
    )
    return result

    # result = MemoryProfilingResult()

    # result.before_create = baseline_snapshot
    # # the part of memory used for holding the model weights
    # result.weights_memory = weights_memory

    # result.before_profile.measure()

    # yield result

    # gc.collect()
    # torch.cuda.empty_cache()

    # result.after_profile.measure()

    # diff_profile = result.after_profile - result.before_profile
    # diff_from_create = result.after_profile - result.before_create
    # result.torch_peak_increase = diff_profile.torch_peak
    # result.non_torch_increase = diff_from_create.non_torch_memory
    # result.profile_time = diff_profile.timestamp

    # non_torch_memory = result.non_torch_increase
    # peak_activation_memory = result.torch_peak_increase
    # result.non_kv_cache_memory = (
    #     non_torch_memory + peak_activation_memory + result.weights_memory
    # )  # noqa
