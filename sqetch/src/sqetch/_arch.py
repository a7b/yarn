"""Runtime CUDA compute-capability detection for the kernel JIT build."""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def detect_sm() -> tuple[int, int]:
    """Return (major, minor) compute capability of CUDA device 0."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available.")
    return torch.cuda.get_device_capability(0)


@functools.lru_cache(maxsize=1)
def sm_arch_tag() -> str:
    """Return ``sm_XX`` tag, e.g. ``sm_89`` or ``sm_120``."""
    cc = detect_sm()
    return f"sm_{cc[0]}{cc[1]}"


def arch_flag() -> str:
    """Return ``-arch=sm_XX`` for the visible GPU."""
    return f"-arch={sm_arch_tag()}"


def build_dir_suffix() -> str:
    """Per-arch build-dir suffix so JIT caches do not collide between GPUs."""
    return f"_{sm_arch_tag()}"


@functools.lru_cache(maxsize=1)
def max_shmem_per_block_optin() -> int:
    """Maximum opt-in dynamic shared memory per CUDA block, in bytes."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available.")
    props = torch.cuda.get_device_properties(0)
    for name in (
        "shared_memory_per_block_optin",
        "max_shared_memory_per_block_optin",
        "shared_memory_per_block",
    ):
        v = getattr(props, name, None)
        if v is not None:
            return int(v)
    raise RuntimeError("No shared-memory property exposed by torch.cuda.get_device_properties.")
