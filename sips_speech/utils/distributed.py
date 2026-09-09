# Copyright (c) 2026 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Utility functions for distributed training and inference with SIPS."""

import os
import re
from contextlib import contextmanager

import torch
import torch.distributed as dist


def setup_distributed():
    if dist.is_initialized():
        return int(os.environ.get("LOCAL_RANK", 0)) if torch.cuda.is_available() else 0

    # torchrun sets RANK, WORLD_SIZE, and LOCAL_RANK.
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if not distributed:
        local_rank = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return local_rank

    if torch.cuda.is_available():
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        local_rank = 0
        backend = "gloo"

    dist.init_process_group(backend=backend)
    return local_rank


def is_rank0():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def print0(*args, **kwargs):
    if is_rank0():
        print(*args, **kwargs)


@contextmanager
def switch_backend(mode: str):
    """
    Temporarily switch PyTorch backend flags.

    mode:
        "train"  - conservative, reproducible training
        "valid"  - fast inference / sampling
    """
    # Save current state
    state = dict(
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        mm_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
    )

    try:
        if mode == "train":
            torch.backends.cudnn.benchmark = state["cudnn_benchmark"]
            torch.backends.cudnn.deterministic = state["cudnn_deterministic"]
            torch.backends.cuda.matmul.allow_tf32 = state["mm_tf32"]
            torch.backends.cudnn.allow_tf32 = state["cudnn_tf32"]
        elif mode == "valid":
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = True
        else:
            raise ValueError(f"Unknown backend mode: {mode}")
        yield
    finally:
        # Restore original state
        torch.backends.cudnn.benchmark = state["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = state["cudnn_deterministic"]
        torch.backends.cuda.matmul.allow_tf32 = state["mm_tf32"]
        torch.backends.cudnn.allow_tf32 = state["cudnn_tf32"]


def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())


def check_ddp_consistency(module, ignore_regex=None):
    if not dist.is_initialized() or get_world_size() == 1:
        return

    assert isinstance(module, torch.nn.Module)
    for name, tensor in named_params_and_buffers(module):
        fullname = type(module).__name__ + "." + name
        if ignore_regex is not None and re.fullmatch(ignore_regex, fullname):
            continue
        tensor = tensor.detach()
        if tensor.is_floating_point():
            tensor = torch.nan_to_num(tensor)
        other = tensor.clone()
        torch.distributed.broadcast(tensor=other, src=0)
        assert (tensor == other).all(), fullname


@contextmanager
def ddp_sync(module, sync):
    assert isinstance(module, torch.nn.Module)
    if sync or not isinstance(module, torch.nn.parallel.DistributedDataParallel):
        yield
    else:
        with module.no_sync():
            yield
