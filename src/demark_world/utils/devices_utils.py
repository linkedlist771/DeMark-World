from functools import lru_cache

import torch
from loguru import logger


@lru_cache()
def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    if torch.backends.mps.is_available():
        device = "mps"
    logger.debug(f"Using device: {device}")
    return torch.device(device)


@lru_cache()
def is_bf16_supported() -> bool:
    """Return True if the current device supports bfloat16 inference."""
    if not torch.cuda.is_available():
        return False
    supported = torch.cuda.is_bf16_supported()
    logger.debug(f"bf16 supported: {supported}")
    return supported
