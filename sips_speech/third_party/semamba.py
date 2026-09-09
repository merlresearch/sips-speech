# Copyright (c) 2026 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) 2024 Rong Chao
# Copyright (c) 2024 Tri Dao, Albert Gu
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT
# SPDX-License-Identifier: Apache-2.0
# 
# ================================================================
# Third-Party Code Notice
# ================================================================
# This file includes code taken from the following repositories:
#
# SEMamba (MIT License)
# https://github.com/RoyChao19477/SEMamba
# Specific files:
# - https://github.com/RoyChao19477/SEMamba/blob/750ff5d8f0e112557ca039c268cdcc51e267f342/models/mamba_block.py
# - https://github.com/RoyChao19477/SEMamba/blob/750ff5d8f0e112557ca039c268cdcc51e267f342/models/lsigmoid.py
# - https://github.com/RoyChao19477/SEMamba/blob/750ff5d8f0e112557ca039c268cdcc51e267f342/models/codec_module.py
# - https://github.com/RoyChao19477/SEMamba/blob/750ff5d8f0e112557ca039c268cdcc51e267f342/models/stfts.py
# - https://github.com/RoyChao19477/SEMamba/blob/750ff5d8f0e112557ca039c268cdcc51e267f342/models/generator.py
#
# Copyright (c) 2024 Rong Chao
# Portions copyright:
# - (c) 2023 Ye-Xin Lu (MP-SENet)
# - (c) 2022 Ruizhe Cao (CMGAN)
#
# This software is based on MP-SENet (https://github.com/yxlu-0102/MP-SENet)
# and CMGAN (https://github.com/ruizhecao96/CMGAN), licensed under MIT.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files, to deal in the Software
# without restriction, subject to the conditions of the MIT License.
#
# Unless otherwise stated, modifications and integrations in this file are
# original work.
#
# Mamba (Apache License 2.0)
# https://github.com/state-spaces/mamba
# Specific files:
# - https://github.com/state-spaces/mamba/blob/bd6dc62a724e6d2f1c821a15b3b2cc02e2b810b4/mamba_ssm/utils/determinism.py
# - https://github.com/state-spaces/mamba/blob/bd6dc62a724e6d2f1c821a15b3b2cc02e2b810b4/mamba_ssm/ops/triton/layer_norm.py
# - https://github.com/state-spaces/mamba/blob/bd6dc62a724e6d2f1c821a15b3b2cc02e2b810b4/mamba_ssm/ops/triton/softplus.py
# - https://github.com/state-spaces/mamba/blob/bd6dc62a724e6d2f1c821a15b3b2cc02e2b810b4/mamba_ssm/ops/triton/selective_state_update.py
# - https://github.com/state-spaces/mamba/blob/bd6dc62a724e6d2f1c821a15b3b2cc02e2b810b4/mamba_ssm/ops/selective_scan_interface.py
# - https://github.com/state-spaces/mamba/blob/bd6dc62a724e6d2f1c821a15b3b2cc02e2b810b4/mamba_ssm/utils/torch.py
# - https://github.com/state-spaces/mamba/blob/bd6dc62a724e6d2f1c821a15b3b2cc02e2b810b4/mamba_ssm/modules/mamba_simple.py
# - https://github.com/state-spaces/mamba/blob/bd6dc62a724e6d2f1c821a15b3b2cc02e2b810b4/mamba_ssm/modules/block.py
# - https://github.com/state-spaces/mamba/blob/bd6dc62a724e6d2f1c821a15b3b2cc02e2b810b4/mamba_ssm/models/mixer_seq_simple.py
#
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#
# THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
# ================================================================


import math
import os
import warnings
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from packaging import version
from torch import Tensor
import triton.language as tl

import selective_scan_cuda


causal_conv1d_fn = None
causal_conv1d_update = None
causal_conv1d_fwd_function = None
causal_conv1d_bwd_function = None
causal_conv1d_update_function = None


#----------------------------------------------------------------------------
# https://github.com/state-spaces/mamba/blob/bd6dc62a724e6d2f1c821a15b3b2cc02e2b810b4/mamba_ssm/utils/determinism.py#L9-L27

try:
    import triton
    TRITON_VERSION = version.parse(triton.__version__)
except ImportError:
    TRITON_VERSION = version.parse("0.0.0")

TRITON_HAS_CACHE_RESULTS = TRITON_VERSION >= version.parse("3.4.0")
_autotune_warning_issued = False

_deterministic_override = None


def use_deterministic_mode():
    if _deterministic_override is not None:
        return _deterministic_override
    env = os.environ.get('MAMBA_DETERMINISTIC')
    if env:
        return env[0] == '1'
    return torch.are_deterministic_algorithms_enabled()


#----------------------------------------------------------------------------
# https://github.com/state-spaces/mamba/blob/bd6dc62a724e6d2f1c821a15b3b2cc02e2b810b4/mamba_ssm/utils/determinism.py#L35-L77

def _estimate_config_cost(cfg):
    """Estimate shared memory cost of a config. Lower is cheaper."""
    block_product = 1
    for key, val in cfg.kwargs.items():
        if key.startswith('BLOCK_SIZE_'):
            block_product *= val
    return block_product * (getattr(cfg, 'num_stages', 1) or 1)


def _filter_configs_by_block_sizes(configs):
    """Filter configs by TRITON_AUTOTUNE_BLOCK_SIZE_* env vars."""
    env_filters = {}
    for suffix in ('M', 'N', 'K', 'DSTATE'):
        env_val = os.environ.get(f"TRITON_AUTOTUNE_BLOCK_SIZE_{suffix}")
        if env_val is not None:
            env_filters[f'BLOCK_SIZE_{suffix}'] = int(env_val)
    if not env_filters:
        return None
    matching = configs
    for key, target in env_filters.items():
        matching = [c for c in matching if c.kwargs.get(key) == target]
    return matching[:1] if matching else None


def autotune_configs(configs):
    """Select autotune configs for deterministic mode.
    
    Uses cached autotuning (TRITON_CACHE_AUTOTUNING=1) if Triton >= 3.4.0,
    otherwise auto-selects the cheapest config by block size * stages.
    """
    if not configs or not use_deterministic_mode():
        return configs
    if TRITON_HAS_CACHE_RESULTS and os.environ.get("TRITON_CACHE_AUTOTUNING") == "1":
        return configs
    global _autotune_warning_issued
    if not _autotune_warning_issued:
        _autotune_warning_issued = True
        msg = "Deterministic mode: set TRITON_CACHE_AUTOTUNING=1 for cached autotuning." if TRITON_HAS_CACHE_RESULTS else "Deterministic mode: upgrade to Triton >= 3.4.0 for cached autotuning."
        warnings.warn(msg)
    filtered = _filter_configs_by_block_sizes(configs)
    if filtered:
        return filtered
    return [min(configs, key=_estimate_config_cost)]


#----------------------------------------------------------------------------
# https://github.com/state-spaces/mamba/blob/bd6dc62a724e6d2f1c821a15b3b2cc02e2b810b4/mamba_ssm/ops/triton/layer_norm.py#L130-L420

def config_prune(configs):

    if torch.version.hip:
        try:
            # set warp size based on gcn architecure 
            gcn_arch_name = torch.cuda.get_device_properties(0).gcnArchName
            if "gfx10" in gcn_arch_name or "gfx11" in gcn_arch_name:
                # radeon
                warp_size = 32
            else:
                # instinct
                warp_size = 64
        except AttributeError as e:
            # fall back to crude method to set warp size
            device_name = torch.cuda.get_device_properties(0).name
            if 'instinct' in device_name.lower():
                warp_size = 64
            else:
                warp_size = 32
            warnings.warn(f"{e}, warp size set to {warp_size} based on device name: {device_name}", UserWarning)

    else:
        # cuda 
        warp_size = 32    

    max_block_sz = 1024
    max_num_warps = max_block_sz // warp_size
    pruned_configs = [config for config in configs if config.num_warps <= max_num_warps]
    return pruned_configs

configs_autotune = [
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
        ]

pruned_configs_autotune = config_prune(configs_autotune)

@triton.autotune(
    configs=autotune_configs(pruned_configs_autotune),
    key=["N", "HAS_RESIDUAL", "STORE_RESIDUAL_OUT", "IS_RMS_NORM", "HAS_BIAS"],
)
# @triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
# @triton.heuristics({"HAS_RESIDUAL": lambda args: args["RESIDUAL"] is not None})
@triton.heuristics({"HAS_X1": lambda args: args["X1"] is not None})
@triton.heuristics({"HAS_W1": lambda args: args["W1"] is not None})
@triton.heuristics({"HAS_B1": lambda args: args["B1"] is not None})
@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    RESIDUAL,  # pointer to the residual
    X1,
    W1,
    B1,
    Y1,
    RESIDUAL_OUT,  # pointer to the residual
    ROWSCALE,
    SEEDS,  # Dropout seeds for each row
    DROPOUT_MASK,
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_res_row,
    stride_res_out_row,
    stride_x1_row,
    stride_y1_row,
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    dropout_p,  # Dropout probability
    IS_RMS_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_DROPOUT: tl.constexpr,
    STORE_DROPOUT_MASK: tl.constexpr,
    HAS_ROWSCALE: tl.constexpr,
    HAS_X1: tl.constexpr,
    HAS_W1: tl.constexpr,
    HAS_B1: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row
    if HAS_RESIDUAL:
        RESIDUAL += row * stride_res_row
    if STORE_RESIDUAL_OUT:
        RESIDUAL_OUT += row * stride_res_out_row
    if HAS_X1:
        X1 += row * stride_x1_row
    if HAS_W1:
        Y1 += row * stride_y1_row
    # Compute mean and variance
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    if HAS_ROWSCALE:
        rowscale = tl.load(ROWSCALE + row).to(tl.float32)
        x *= rowscale
    if HAS_DROPOUT:
        # Compute dropout mask
        # 7 rounds is good enough, and reduces register pressure
        keep_mask = tl.rand(tl.load(SEEDS + row).to(tl.uint32), cols, n_rounds=7) > dropout_p
        x = tl.where(keep_mask, x / (1.0 - dropout_p), 0.0)
        if STORE_DROPOUT_MASK:
            tl.store(DROPOUT_MASK + row * N + cols, keep_mask, mask=cols < N)
    if HAS_X1:
        x1 = tl.load(X1 + cols, mask=cols < N, other=0.0).to(tl.float32)
        if HAS_ROWSCALE:
            rowscale = tl.load(ROWSCALE + M + row).to(tl.float32)
            x1 *= rowscale
        if HAS_DROPOUT:
            # Compute dropout mask
            # 7 rounds is good enough, and reduces register pressure
            keep_mask = (
                tl.rand(tl.load(SEEDS + M + row).to(tl.uint32), cols, n_rounds=7) > dropout_p
            )
            x1 = tl.where(keep_mask, x1 / (1.0 - dropout_p), 0.0)
            if STORE_DROPOUT_MASK:
                tl.store(DROPOUT_MASK + (M + row) * N + cols, keep_mask, mask=cols < N)
        x += x1
    if HAS_RESIDUAL:
        residual = tl.load(RESIDUAL + cols, mask=cols < N, other=0.0).to(tl.float32)
        x += residual
    if STORE_RESIDUAL_OUT:
        tl.store(RESIDUAL_OUT + cols, x, mask=cols < N)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        xbar = tl.where(cols < N, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        xbar = tl.where(cols < N, x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    mask = cols < N
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w + b if HAS_BIAS else x_hat * w
    # Write output
    tl.store(Y + cols, y, mask=mask)
    if HAS_W1:
        w1 = tl.load(W1 + cols, mask=mask).to(tl.float32)
        if HAS_B1:
            b1 = tl.load(B1 + cols, mask=mask).to(tl.float32)
        y1 = x_hat * w1 + b1 if HAS_B1 else x_hat * w1
        tl.store(Y1 + cols, y1, mask=mask)


def _layer_norm_fwd(
    x,
    weight,
    bias,
    eps,
    residual=None,
    x1=None,
    weight1=None,
    bias1=None,
    dropout_p=0.0,
    rowscale=None,
    out_dtype=None,
    residual_dtype=None,
    is_rms_norm=False,
    return_dropout_mask=False,
):
    if residual is not None:
        residual_dtype = residual.dtype
    M, N = x.shape
    assert x.stride(-1) == 1
    if residual is not None:
        assert residual.stride(-1) == 1
        assert residual.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    if x1 is not None:
        assert x1.shape == x.shape
        assert rowscale is None
        assert x1.stride(-1) == 1
    if weight1 is not None:
        assert weight1.shape == (N,)
        assert weight1.stride(-1) == 1
    if bias1 is not None:
        assert bias1.shape == (N,)
        assert bias1.stride(-1) == 1
    if rowscale is not None:
        assert rowscale.is_contiguous()
        assert rowscale.shape == (M,)
    # allocate output
    y = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    assert y.stride(-1) == 1
    if weight1 is not None:
        y1 = torch.empty_like(y)
        assert y1.stride(-1) == 1
    else:
        y1 = None
    if (
        residual is not None
        or (residual_dtype is not None and residual_dtype != x.dtype)
        or dropout_p > 0.0
        or rowscale is not None
        or x1 is not None
    ):
        residual_out = torch.empty(
            M, N, device=x.device, dtype=residual_dtype if residual_dtype is not None else x.dtype
        )
        assert residual_out.stride(-1) == 1
    else:
        residual_out = None
    mean = torch.empty((M,), dtype=torch.float32, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    if dropout_p > 0.0:
        seeds = torch.randint(
            2**32, (M if x1 is None else 2 * M,), device=x.device, dtype=torch.int64
        )
    else:
        seeds = None
    if return_dropout_mask and dropout_p > 0.0:
        dropout_mask = torch.empty(M if x1 is None else 2 * M, N, device=x.device, dtype=torch.bool)
    else:
        dropout_mask = None
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    with torch.cuda.device(x.device.index):
        _layer_norm_fwd_1pass_kernel[(M,)](
            x,
            y,
            weight,
            bias,
            residual,
            x1,
            weight1,
            bias1,
            y1,
            residual_out,
            rowscale,
            seeds,
            dropout_mask,
            mean,
            rstd,
            x.stride(0),
            y.stride(0),
            residual.stride(0) if residual is not None else 0,
            residual_out.stride(0) if residual_out is not None else 0,
            x1.stride(0) if x1 is not None else 0,
            y1.stride(0) if y1 is not None else 0,
            M,
            N,
            eps,
            dropout_p,
            is_rms_norm,
            BLOCK_N,
            residual is not None,
            residual_out is not None,
            bias is not None,
            dropout_p > 0.0,
            dropout_mask is not None,
            rowscale is not None,
        )
    # residual_out is None if residual is None and residual_dtype == input_dtype and dropout_p == 0.0
    if dropout_mask is not None and x1 is not None:
        dropout_mask, dropout_mask1 = dropout_mask.tensor_split(2, dim=0)
    else:
        dropout_mask1 = None
    return (
        y,
        y1,
        mean,
        rstd,
        residual_out if residual_out is not None else x,
        seeds,
        dropout_mask,
        dropout_mask1,
    )


#----------------------------------------------------------------------------
# https://github.com/state-spaces/mamba/blob/bd6dc62a724e6d2f1c821a15b3b2cc02e2b810b4/mamba_ssm/ops/triton/layer_norm.py#L423-L982

@triton.autotune(
    configs=autotune_configs(pruned_configs_autotune),
    key=["N", "HAS_DRESIDUAL", "STORE_DRESIDUAL", "IS_RMS_NORM", "HAS_BIAS", "HAS_DROPOUT"],
)
# @triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
# @triton.heuristics({"HAS_DRESIDUAL": lambda args: args["DRESIDUAL"] is not None})
# @triton.heuristics({"STORE_DRESIDUAL": lambda args: args["DRESIDUAL_IN"] is not None})
@triton.heuristics({"HAS_ROWSCALE": lambda args: args["ROWSCALE"] is not None})
@triton.heuristics({"HAS_DY1": lambda args: args["DY1"] is not None})
@triton.heuristics({"HAS_DX1": lambda args: args["DX1"] is not None})
@triton.heuristics({"HAS_B1": lambda args: args["DB1"] is not None})
@triton.heuristics({"RECOMPUTE_OUTPUT": lambda args: args["Y"] is not None})
@triton.jit
def _layer_norm_bwd_kernel(
    X,  # pointer to the input
    W,  # pointer to the weights
    B,  # pointer to the biases
    Y,  # pointer to the output to be recomputed
    DY,  # pointer to the output gradient
    DX,  # pointer to the input gradient
    DW,  # pointer to the partial sum of weights gradient
    DB,  # pointer to the partial sum of biases gradient
    DRESIDUAL,
    W1,
    DY1,
    DX1,
    DW1,
    DB1,
    DRESIDUAL_IN,
    ROWSCALE,
    SEEDS,
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_dy_row,
    stride_dx_row,
    stride_dres_row,
    stride_dy1_row,
    stride_dx1_row,
    stride_dres_in_row,
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    dropout_p,
    rows_per_program,
    IS_RMS_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_DRESIDUAL: tl.constexpr,
    STORE_DRESIDUAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_DROPOUT: tl.constexpr,
    HAS_ROWSCALE: tl.constexpr,
    HAS_DY1: tl.constexpr,
    HAS_DX1: tl.constexpr,
    HAS_B1: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    # Do not early exit if row_start >= M, because we need to write DW and DB
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    X += row_start * stride_x_row
    if HAS_DRESIDUAL:
        DRESIDUAL += row_start * stride_dres_row
    if STORE_DRESIDUAL:
        DRESIDUAL_IN += row_start * stride_dres_in_row
    DY += row_start * stride_dy_row
    DX += row_start * stride_dx_row
    if HAS_DY1:
        DY1 += row_start * stride_dy1_row
    if HAS_DX1:
        DX1 += row_start * stride_dx1_row
    if RECOMPUTE_OUTPUT:
        Y += row_start * stride_y_row
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if RECOMPUTE_OUTPUT and HAS_BIAS:
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_DY1:
        w1 = tl.load(W1 + cols, mask=mask).to(tl.float32)
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAS_BIAS:
        db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAS_DY1:
        dw1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        if HAS_B1:
            db1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    row_end = min((row_block_id + 1) * rows_per_program, M)
    for row in range(row_start, row_end):
        # Load data to SRAM
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        if HAS_DY1:
            dy1 = tl.load(DY1 + cols, mask=mask, other=0).to(tl.float32)
        if not IS_RMS_NORM:
            mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        # Compute dx
        xhat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        xhat = tl.where(mask, xhat, 0.0)
        if RECOMPUTE_OUTPUT:
            y = xhat * w + b if HAS_BIAS else xhat * w
            tl.store(Y + cols, y, mask=mask)
        wdy = w * dy
        dw += dy * xhat
        if HAS_BIAS:
            db += dy
        if HAS_DY1:
            wdy += w1 * dy1
            dw1 += dy1 * xhat
            if HAS_B1:
                db1 += dy1
        if not IS_RMS_NORM:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            c2 = tl.sum(wdy, axis=0) / N
            dx = (wdy - (xhat * c1 + c2)) * rstd
        else:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            dx = (wdy - xhat * c1) * rstd
        if HAS_DRESIDUAL:
            dres = tl.load(DRESIDUAL + cols, mask=mask, other=0).to(tl.float32)
            dx += dres
        # Write dx
        if STORE_DRESIDUAL:
            tl.store(DRESIDUAL_IN + cols, dx, mask=mask)
        if HAS_DX1:
            if HAS_DROPOUT:
                keep_mask = (
                    tl.rand(tl.load(SEEDS + M + row).to(tl.uint32), cols, n_rounds=7) > dropout_p
                )
                dx1 = tl.where(keep_mask, dx / (1.0 - dropout_p), 0.0)
            else:
                dx1 = dx
            tl.store(DX1 + cols, dx1, mask=mask)
        if HAS_DROPOUT:
            keep_mask = tl.rand(tl.load(SEEDS + row).to(tl.uint32), cols, n_rounds=7) > dropout_p
            dx = tl.where(keep_mask, dx / (1.0 - dropout_p), 0.0)
        if HAS_ROWSCALE:
            rowscale = tl.load(ROWSCALE + row).to(tl.float32)
            dx *= rowscale
        tl.store(DX + cols, dx, mask=mask)

        X += stride_x_row
        if HAS_DRESIDUAL:
            DRESIDUAL += stride_dres_row
        if STORE_DRESIDUAL:
            DRESIDUAL_IN += stride_dres_in_row
        if RECOMPUTE_OUTPUT:
            Y += stride_y_row
        DY += stride_dy_row
        DX += stride_dx_row
        if HAS_DY1:
            DY1 += stride_dy1_row
        if HAS_DX1:
            DX1 += stride_dx1_row
    tl.store(DW + row_block_id * N + cols, dw, mask=mask)
    if HAS_BIAS:
        tl.store(DB + row_block_id * N + cols, db, mask=mask)
    if HAS_DY1:
        tl.store(DW1 + row_block_id * N + cols, dw1, mask=mask)
        if HAS_B1:
            tl.store(DB1 + row_block_id * N + cols, db1, mask=mask)


def _layer_norm_bwd(
    dy,
    x,
    weight,
    bias,
    eps,
    mean,
    rstd,
    dresidual=None,
    dy1=None,
    weight1=None,
    bias1=None,
    seeds=None,
    dropout_p=0.0,
    rowscale=None,
    has_residual=False,
    has_x1=False,
    is_rms_norm=False,
    x_dtype=None,
    recompute_output=False,
):
    M, N = x.shape
    assert x.stride(-1) == 1
    assert dy.stride(-1) == 1
    assert dy.shape == (M, N)
    if dresidual is not None:
        assert dresidual.stride(-1) == 1
        assert dresidual.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    if dy1 is not None:
        assert weight1 is not None
        assert dy1.shape == dy.shape
        assert dy1.stride(-1) == 1
    if weight1 is not None:
        assert weight1.shape == (N,)
        assert weight1.stride(-1) == 1
    if bias1 is not None:
        assert bias1.shape == (N,)
        assert bias1.stride(-1) == 1
    if seeds is not None:
        assert seeds.is_contiguous()
        assert seeds.shape == (M if not has_x1 else M * 2,)
    if rowscale is not None:
        assert rowscale.is_contiguous()
        assert rowscale.shape == (M,)
    # allocate output
    dx = (
        torch.empty_like(x)
        if x_dtype is None
        else torch.empty(M, N, dtype=x_dtype, device=x.device)
    )
    dresidual_in = (
        torch.empty_like(x)
        if has_residual
        and (dx.dtype != x.dtype or dropout_p > 0.0 or rowscale is not None or has_x1)
        else None
    )
    dx1 = torch.empty_like(dx) if (has_x1 and dropout_p > 0.0) else None
    y = torch.empty(M, N, dtype=dy.dtype, device=dy.device) if recompute_output else None
    if recompute_output:
        assert weight1 is None, "recompute_output is not supported with parallel LayerNorm"

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    _dw = torch.empty((sm_count, N), dtype=torch.float32, device=weight.device)
    _db = (
        torch.empty((sm_count, N), dtype=torch.float32, device=bias.device)
        if bias is not None
        else None
    )
    _dw1 = torch.empty_like(_dw) if weight1 is not None else None
    _db1 = torch.empty_like(_db) if bias1 is not None else None
    rows_per_program = math.ceil(M / sm_count)
    grid = (sm_count,)
    with torch.cuda.device(x.device.index):
        _layer_norm_bwd_kernel[grid](
            x,
            weight,
            bias,
            y,
            dy,
            dx,
            _dw,
            _db,
            dresidual,
            weight1,
            dy1,
            dx1,
            _dw1,
            _db1,
            dresidual_in,
            rowscale,
            seeds,
            mean,
            rstd,
            x.stride(0),
            0 if not recompute_output else y.stride(0),
            dy.stride(0),
            dx.stride(0),
            dresidual.stride(0) if dresidual is not None else 0,
            dy1.stride(0) if dy1 is not None else 0,
            dx1.stride(0) if dx1 is not None else 0,
            dresidual_in.stride(0) if dresidual_in is not None else 0,
            M,
            N,
            eps,
            dropout_p,
            rows_per_program,
            is_rms_norm,
            BLOCK_N,
            dresidual is not None,
            dresidual_in is not None,
            bias is not None,
            dropout_p > 0.0,
        )
    dw = _dw.sum(0).to(weight.dtype)
    db = _db.sum(0).to(bias.dtype) if bias is not None else None
    dw1 = _dw1.sum(0).to(weight1.dtype) if weight1 is not None else None
    db1 = _db1.sum(0).to(bias1.dtype) if bias1 is not None else None
    # Don't need to compute dresidual_in separately in this case
    if has_residual and dx.dtype == x.dtype and dropout_p == 0.0 and rowscale is None:
        dresidual_in = dx
    if has_x1 and dropout_p == 0.0:
        dx1 = dx
    return (
        (dx, dw, db, dresidual_in, dx1, dw1, db1)
        if not recompute_output
        else (dx, dw, db, dresidual_in, dx1, dw1, db1, y)
    )


class LayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias,
        residual=None,
        x1=None,
        weight1=None,
        bias1=None,
        eps=1e-6,
        dropout_p=0.0,
        rowscale=None,
        prenorm=False,
        residual_in_fp32=False,
        is_rms_norm=False,
        return_dropout_mask=False,
    ):
        x_shape_og = x.shape
        # reshape input data into 2D tensor
        x = x.reshape(-1, x.shape[-1])
        if x.stride(-1) != 1:
            x = x.contiguous()
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape(-1, residual.shape[-1])
            if residual.stride(-1) != 1:
                residual = residual.contiguous()
        if x1 is not None:
            assert x1.shape == x_shape_og
            assert rowscale is None, "rowscale is not supported with parallel LayerNorm"
            x1 = x1.reshape(-1, x1.shape[-1])
            if x1.stride(-1) != 1:
                x1 = x1.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        if weight1 is not None:
            weight1 = weight1.contiguous()
        if bias1 is not None:
            bias1 = bias1.contiguous()
        if rowscale is not None:
            rowscale = rowscale.reshape(-1).contiguous()
        residual_dtype = (
            residual.dtype
            if residual is not None
            else (torch.float32 if residual_in_fp32 else None)
        )
        y, y1, mean, rstd, residual_out, seeds, dropout_mask, dropout_mask1 = _layer_norm_fwd(
            x,
            weight,
            bias,
            eps,
            residual,
            x1,
            weight1,
            bias1,
            dropout_p=dropout_p,
            rowscale=rowscale,
            residual_dtype=residual_dtype,
            is_rms_norm=is_rms_norm,
            return_dropout_mask=return_dropout_mask,
        )
        ctx.save_for_backward(
            residual_out, weight, bias, weight1, bias1, rowscale, seeds, mean, rstd
        )
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.dropout_p = dropout_p
        ctx.is_rms_norm = is_rms_norm
        ctx.has_residual = residual is not None
        ctx.has_x1 = x1 is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        y = y.reshape(x_shape_og)
        y1 = y1.reshape(x_shape_og) if y1 is not None else None
        residual_out = residual_out.reshape(x_shape_og) if residual_out is not None else None
        dropout_mask = dropout_mask.reshape(x_shape_og) if dropout_mask is not None else None
        dropout_mask1 = dropout_mask1.reshape(x_shape_og) if dropout_mask1 is not None else None
        if not return_dropout_mask:
            if weight1 is None:
                return y if not prenorm else (y, residual_out)
            else:
                return (y, y1) if not prenorm else (y, y1, residual_out)
        else:
            if weight1 is None:
                return (
                    (y, dropout_mask, dropout_mask1)
                    if not prenorm
                    else (y, residual_out, dropout_mask, dropout_mask1)
                )
            else:
                return (
                    (y, y1, dropout_mask, dropout_mask1)
                    if not prenorm
                    else (y, y1, residual_out, dropout_mask, dropout_mask1)
                )

    @staticmethod
    def backward(ctx, dy, *args):
        x, weight, bias, weight1, bias1, rowscale, seeds, mean, rstd = ctx.saved_tensors
        dy = dy.reshape(-1, dy.shape[-1])
        if dy.stride(-1) != 1:
            dy = dy.contiguous()
        assert dy.shape == x.shape
        if weight1 is not None:
            dy1, args = args[0], args[1:]
            dy1 = dy1.reshape(-1, dy1.shape[-1])
            if dy1.stride(-1) != 1:
                dy1 = dy1.contiguous()
            assert dy1.shape == x.shape
        else:
            dy1 = None
        if ctx.prenorm:
            dresidual = args[0]
            dresidual = dresidual.reshape(-1, dresidual.shape[-1])
            if dresidual.stride(-1) != 1:
                dresidual = dresidual.contiguous()
            assert dresidual.shape == x.shape
        else:
            dresidual = None
        dx, dw, db, dresidual_in, dx1, dw1, db1 = _layer_norm_bwd(
            dy,
            x,
            weight,
            bias,
            ctx.eps,
            mean,
            rstd,
            dresidual,
            dy1,
            weight1,
            bias1,
            seeds,
            ctx.dropout_p,
            rowscale,
            ctx.has_residual,
            ctx.has_x1,
            ctx.is_rms_norm,
            x_dtype=ctx.x_dtype,
        )
        return (
            dx.reshape(ctx.x_shape_og),
            dw,
            db,
            dresidual_in.reshape(ctx.x_shape_og) if ctx.has_residual else None,
            dx1.reshape(ctx.x_shape_og) if dx1 is not None else None,
            dw1,
            db1,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def layer_norm_fn(
    x,
    weight,
    bias,
    residual=None,
    x1=None,
    weight1=None,
    bias1=None,
    eps=1e-6,
    dropout_p=0.0,
    rowscale=None,
    prenorm=False,
    residual_in_fp32=False,
    is_rms_norm=False,
    return_dropout_mask=False,
):
    return LayerNormFn.apply(
        x,
        weight,
        bias,
        residual,
        x1,
        weight1,
        bias1,
        eps,
        dropout_p,
        rowscale,
        prenorm,
        residual_in_fp32,
        is_rms_norm,
        return_dropout_mask,
    )


def rms_norm_fn(
    x,
    weight,
    bias,
    residual=None,
    x1=None,
    weight1=None,
    bias1=None,
    eps=1e-6,
    dropout_p=0.0,
    rowscale=None,
    prenorm=False,
    residual_in_fp32=False,
    return_dropout_mask=False,
):
    return LayerNormFn.apply(
        x,
        weight,
        bias,
        residual,
        x1,
        weight1,
        bias1,
        eps,
        dropout_p,
        rowscale,
        prenorm,
        residual_in_fp32,
        True,
        return_dropout_mask,
    )


class RMSNorm(torch.nn.Module):

    def __init__(self, hidden_size, eps=1e-5, dropout_p=0.0, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        if dropout_p > 0.0:
            self.drop = torch.nn.Dropout(dropout_p)
        else:
            self.drop = None
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        return rms_norm_fn(
            x,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
            dropout_p=self.drop.p if self.drop is not None and self.training else 0.0,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )
    
# ----------------------------------------------------------------------------
# https://github.com/state-spaces/mamba/blob/bd6dc62a724e6d2f1c821a15b3b2cc02e2b810b4/mamba_ssm/ops/triton/softplus.py#L5-L15

TRITON3 = version.parse(triton.__version__) >= version.parse("3.0.0")


if TRITON3:
    @triton.jit
    def softplus(dt):
        return tl.math.log(tl.math.exp(dt) + 1)
else:
    @triton.jit
    def softplus(dt):
        return tl.math.log1p(tl.exp(dt))
    

#  ----------------------------------------------------------------------------
# https://github.com/state-spaces/mamba/blob/bd6dc62a724e6d2f1c821a15b3b2cc02e2b810b4/mamba_ssm/ops/triton/selective_state_update.py#L18-L221

@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics({"HAS_STATE_BATCH_INDICES": lambda args: args["state_batch_indices_ptr"] is not None})
@triton.heuristics({"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])})
@triton.jit
def _selective_scan_update_kernel(
    # Pointers to matrices
    state_ptr, x_ptr, dt_ptr, dt_bias_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr, out_ptr, state_batch_indices_ptr,
    # Matrix dimensions
    batch, nheads, dim, dstate, nheads_ngroups_ratio,
    # Strides
    stride_state_batch, stride_state_head, stride_state_dim, stride_state_dstate,
    stride_x_batch, stride_x_head, stride_x_dim,
    stride_dt_batch, stride_dt_head, stride_dt_dim,
    stride_dt_bias_head, stride_dt_bias_dim,
    stride_A_head, stride_A_dim, stride_A_dstate,
    stride_B_batch, stride_B_group, stride_B_dstate,
    stride_C_batch, stride_C_group, stride_C_dstate,
    stride_D_head, stride_D_dim,
    stride_z_batch, stride_z_head, stride_z_dim,
    stride_out_batch, stride_out_head, stride_out_dim,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    TIE_HDIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_STATE_BATCH_INDICES: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head
    out_ptrs = out_ptr + offs_m * stride_out_dim

    if HAS_STATE_BATCH_INDICES:
        state_batch_indices_ptr += pid_b
        state_batch_idx = tl.load(state_batch_indices_ptr)
        # Skip padding tokens
        if state_batch_idx < 0:
            tl.store(out_ptrs, 0.0, mask=offs_m < dim)
            return
        state_ptr += state_batch_idx * stride_state_batch + pid_h * stride_state_head
    else:
        state_ptr += pid_b * stride_state_batch + pid_h * stride_state_head

    x_ptr += pid_b * stride_x_batch + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_h * stride_dt_head
    if HAS_DT_BIAS:
        dt_bias_ptr += pid_h * stride_dt_bias_head
    A_ptr += pid_h * stride_A_head
    B_ptr += pid_b * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
    C_ptr += pid_b * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group
    if HAS_Z:
        z_ptr += pid_b * stride_z_batch + pid_h * stride_z_head

    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    state_ptrs = state_ptr + (offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate)
    x_ptrs = x_ptr + offs_m * stride_x_dim
    dt_ptrs = dt_ptr + offs_m * stride_dt_dim
    if HAS_DT_BIAS:
        dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim
    if HAS_D:
        D_ptr += pid_h * stride_D_head
    A_ptrs = A_ptr + (offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate)
    B_ptrs = B_ptr + offs_n * stride_B_dstate
    C_ptrs = C_ptr + offs_n * stride_C_dstate
    if HAS_D:
        D_ptrs = D_ptr + offs_m * stride_D_dim
    if HAS_Z:
        z_ptrs = z_ptr + offs_m * stride_z_dim

    state = tl.load(state_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0)
    x = tl.load(x_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    if not TIE_HDIM:
        dt = tl.load(dt_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if HAS_DT_BIAS:
            dt += tl.load(dt_bias_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if DT_SOFTPLUS:
            dt = tl.where(dt <= 20.0, softplus(dt), dt)
        A = tl.load(A_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
        dA = tl.exp(A * dt[:, None])
    else:
        dt = tl.load(dt_ptr).to(tl.float32)
        if HAS_DT_BIAS:
            dt += tl.load(dt_bias_ptr).to(tl.float32)
        if DT_SOFTPLUS:
            dt = tl.where(dt <= 20.0, softplus(dt), dt)
        A = tl.load(A_ptr).to(tl.float32)
        dA = tl.exp(A * dt)  # scalar, not a matrix

    B = tl.load(B_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
    C = tl.load(C_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
    if HAS_D:
        D = tl.load(D_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    if HAS_Z:
        z = tl.load(z_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)

    if not TIE_HDIM:
        dB = B[None, :] * dt[:, None]
    else:
        dB = B * dt  # vector of size (dstate,)
    state = state * dA + dB * x[:, None]
    tl.store(state_ptrs, state, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate))
    out = tl.sum(state * C[None, :], axis=1)
    if HAS_D:
        out += x * D
    if HAS_Z:
        out *= z * tl.sigmoid(z)
    tl.store(out_ptrs, out, mask=offs_m < dim)


def selective_state_update(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False,
                           state_batch_indices=None):
    """
    Argument:
        state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        x: (batch, dim) or (batch, nheads, dim)
        dt: (batch, dim) or (batch, nheads, dim)
        A: (dim, dstate) or (nheads, dim, dstate)
        B: (batch, dstate) or (batch, ngroups, dstate)
        C: (batch, dstate) or (batch, ngroups, dstate)
        D: (dim,) or (nheads, dim)
        z: (batch, dim) or (batch, nheads, dim)
        dt_bias: (dim,) or (nheads, dim)
    Return:
        out: (batch, dim) or (batch, nheads, dim)
    """
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    _, nheads, dim, dstate = state.shape
    batch = x.shape[0]
    if x.shape != (batch, nheads, dim):
        print(f"{state.shape} {x.shape} {batch} {nheads} {dim}")
    assert x.shape == (batch, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[1]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
    if state_batch_indices is not None:
        assert state_batch_indices.shape == (batch,)
    out = torch.empty_like(x)
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE_M']), batch, nheads)
    z_strides = ((z.stride(0), z.stride(1), z.stride(2)) if z is not None else (0, 0, 0))
    # We don't want autotune since it will overwrite the state
    # We instead tune by hand.
    BLOCK_SIZE_M, num_warps = ((32, 4) if dstate <= 16
                               else ((16, 4) if dstate <= 32 else
                                     ((8, 4) if dstate <= 64 else
                                      ((4, 4) if dstate <= 128 else
                                       ((4, 8))))))
    tie_hdim = A.stride(-1) == 0 and A.stride(-2) == 0 and dt.stride(-1) == 0 and dt_bias.stride(-1) == 0
    with torch.cuda.device(x.device.index):
        _selective_scan_update_kernel[grid](
            state, x, dt, dt_bias, A, B, C, D, z, out, state_batch_indices,
            batch, nheads, dim, dstate, nheads // ngroups,
            state.stride(0), state.stride(1), state.stride(2), state.stride(3),
            x.stride(0), x.stride(1), x.stride(2),
            dt.stride(0), dt.stride(1), dt.stride(2),
            *(dt_bias.stride(0), dt_bias.stride(1)) if dt_bias is not None else 0,
            A.stride(0), A.stride(1), A.stride(2),
            B.stride(0), B.stride(1), B.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            *(D.stride(0), D.stride(1)) if D is not None else 0,
            z_strides[0], z_strides[1], z_strides[2],
            out.stride(0), out.stride(1), out.stride(2),
            dt_softplus,
            tie_hdim,
            BLOCK_SIZE_M,
            num_warps=num_warps,
        )
    if not has_heads:
        out = out.squeeze(1)
    return out

# ----------------------------------------------------------------------------
# https://github.com/state-spaces/mamba/blob/bd6dc62a724e6d2f1c821a15b3b2cc02e2b810b4/mamba_ssm/ops/selective_scan_interface.py#L23-L112

class SelectiveScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)

        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
        if not ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out if not return_last_state else (out, last_state)
        else:
            ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
            out_z = rest[0]
            return out_z if not return_last_state else (out_z, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        if not ctx.has_z:
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            z = None
            out = None
        else:
            u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        # Here we just pass in None and dz will be allocated in the C++ code.
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, ctx.delta_softplus,
            False  # option to recompute out_z, not used here
        )
        dz = rest[0] if ctx.has_z else None
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC,
                dD if D is not None else None,
                dz,
                ddelta_bias if delta_bias is not None else None,
                None,
                None)


def rms_norm_forward(
    x,
    weight,
    bias,
    eps=1e-6,
    is_rms_norm=True,
):
    # x (b l) d
    if x.stride(-1) != 1:
        x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    y = _layer_norm_fwd(
        x, weight, bias, eps, None, residual_dtype=None, is_rms_norm=is_rms_norm
    )[0]
    # y (b l) d
    return y


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                     return_last_state=False):
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    """
    return SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)
    

# ---------------------------------------------------------------------------
# https://github.com/state-spaces/mamba/blob/bd6dc62a724e6d2f1c821a15b3b2cc02e2b810b4/mamba_ssm/utils/torch.py#L5-L21

def custom_amp_decorator(dec: Callable, cuda_amp_deprecated: bool):
    def decorator(*args, **kwargs):
        if cuda_amp_deprecated:
            kwargs["device_type"] = "cuda"
        return dec(*args, **kwargs)
    return decorator


if hasattr(torch.amp, "custom_fwd"): # type: ignore[attr-defined]
    deprecated = True
    from torch.amp import custom_fwd, custom_bwd # type: ignore[attr-defined]
else:
    deprecated = False
    from torch.cuda.amp import custom_fwd, custom_bwd

custom_fwd = custom_amp_decorator(custom_fwd, deprecated)
custom_bwd = custom_amp_decorator(custom_bwd, deprecated)


# ---------------------------------------------------------------------------
# https://github.com/state-spaces/mamba/blob/bd6dc62a724e6d2f1c821a15b3b2cc02e2b810b4/mamba_ssm/ops/selective_scan_interface.py#L184-L381

class MambaInnerFn(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                out_proj_weight, out_proj_bias,
                A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
                C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1, b_rms_weight=None, c_rms_weight= None, dt_rms_weight= None, b_c_dt_rms_eps=1e-6):
        """
             xz: (batch, dim, seqlen)
        """
        assert causal_conv1d_fwd_function is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."
        assert checkpoint_lvl in [0, 1]
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        if torch.is_autocast_enabled():
            x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_weight = out_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_bias = (out_proj_bias.to(dtype=torch.get_autocast_gpu_dtype())
                             if out_proj_bias is not None else None)
        if xz.stride(-1) != 1:
            xz = xz.contiguous()
        conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
        x, z = xz.chunk(2, dim=1)
        conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
        conv1d_out = causal_conv1d_fwd_function(
            x, conv1d_weight, conv1d_bias, None, None, None, True
        )
        # We're being very careful here about the layout, to avoid extra transposes.
        # We want delta to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
        delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l = L)
        ctx.is_variable_B = B is None
        ctx.is_variable_C = C is None
        ctx.B_proj_bias_is_None = B_proj_bias is None
        ctx.C_proj_bias_is_None = C_proj_bias is None
        if B is None:  # variable B
            B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl dstate)
            if B_proj_bias is not None:
                B = B + B_proj_bias.to(dtype=B.dtype)
            if not A.is_complex():
                # B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if B.stride(-1) != 1:
                B = B.contiguous()
        if C is None:  # variable C
            C = x_dbl[:, -d_state:]  # (bl dstate)
            if C_proj_bias is not None:
                C = C + C_proj_bias.to(dtype=C.dtype)
            if not A.is_complex():
                # C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if C.stride(-1) != 1:
                C = C.contiguous()
        if D is not None:
            D = D.contiguous()
            
        if b_rms_weight is not None:
            B = rearrange(B, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
            B = rms_norm_forward(B, b_rms_weight, bias=None, eps=b_c_dt_rms_eps)
            B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        if c_rms_weight is not None:
            C = rearrange(C, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
            C = rms_norm_forward(C, c_rms_weight, bias=None, eps=b_c_dt_rms_eps)
            C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        if dt_rms_weight is not None:
            delta = rearrange(delta, "b d l -> (b l) d", l=L).contiguous()
            delta = rms_norm_forward(delta, dt_rms_weight, bias=None, eps=b_c_dt_rms_eps)
            delta = rearrange(delta, "(b l) d -> b d l", l=L).contiguous()
        
        out, scan_intermediates, out_z = selective_scan_cuda.fwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
        )
        ctx.delta_softplus = delta_softplus
        ctx.out_proj_bias_is_None = out_proj_bias is None
        ctx.checkpoint_lvl = checkpoint_lvl
        ctx.b_rms_weight = b_rms_weight
        ctx.c_rms_weight = c_rms_weight
        ctx.dt_rms_weight = dt_rms_weight
        ctx.b_c_dt_rms_eps = b_c_dt_rms_eps
        if checkpoint_lvl >= 1:  # Will recompute conv1d_out and delta in the backward pass
            conv1d_out, delta = None, None
        ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
                              delta_proj_weight, out_proj_weight, conv1d_out, delta,
                              A, B, C, D, delta_bias, scan_intermediates, b_rms_weight, c_rms_weight, dt_rms_weight, out)
        return F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)

    @staticmethod
    @custom_bwd
    def backward(ctx, dout):
        # dout: (batch, seqlen, dim)
        assert causal_conv1d_fwd_function is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."
        (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, out_proj_weight,
         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, b_rms_weight, c_rms_weight, dt_rms_weight, out) = ctx.saved_tensors
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        x, z = xz.chunk(2, dim=1)
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if ctx.checkpoint_lvl == 1:
            conv1d_out = causal_conv1d_fwd_function(
                x, conv1d_weight, conv1d_bias, None, None, None, True
            )
            delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
                              "d (b l) -> b d l", l = L)
            if dt_rms_weight is not None:
                delta = rearrange(delta, "b d l -> (b l) d", l=L).contiguous()
                delta = rms_norm_forward(delta, ctx.dt_rms_weight, None, ctx.b_c_dt_rms_eps)
                delta = rearrange(delta, "(b l) d -> b d l", l=L).contiguous()
            if b_rms_weight is not None:
                # Recompute & RMSNorm B
                B = rearrange(B, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
                B = rms_norm_forward(
                    B, ctx.b_rms_weight, None, ctx.b_c_dt_rms_eps
                )
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            if c_rms_weight is not None:
                # Recompute & RMSNorm C
                C = rearrange(C, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
                C = rms_norm_forward(
                    C, ctx.c_rms_weight, None, ctx.b_c_dt_rms_eps
                )
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
        dx, dz = dxz.chunk(2, dim=1)
        dout = rearrange(dout, "b l e -> e (b l)")
        dout_y = rearrange(out_proj_weight.t() @ dout, "d (b l) -> b d l", l=L)
        dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, dout_y, scan_intermediates, out, dz,
            ctx.delta_softplus,
            True  # option to recompute out_z
        )
        dout_proj_weight = torch.einsum("eB,dB->ed", dout, rearrange(out_z, "b d l -> d (b l)"))
        dout_proj_bias = dout.sum(dim=(0, 1)) if not ctx.out_proj_bias_is_None else None
        dD = dD if D is not None else None
        dx_dbl = torch.empty_like(x_dbl)
        dB_proj_bias = None
        if ctx.is_variable_B:
            if not A.is_complex():
                dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
            dx_dbl[:, delta_rank:delta_rank + d_state] = dB  # (bl d)
            dB = None
        dC_proj_bias = None
        if ctx.is_variable_C:
            if not A.is_complex():
                dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
            dx_dbl[:, -d_state:] = dC  # (bl d)
            dC = None
        ddelta = rearrange(ddelta, "b d l -> d (b l)")
        ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
        dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
        dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
        dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
        dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
        dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
        # backward of conv1d with the backward of chunk).
        dx, dconv1d_weight, dconv1d_bias, *_ = causal_conv1d_bwd_function(
            x, conv1d_weight, conv1d_bias, dconv1d_out, None, None, None, dx, False, True
        )
        dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
        dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
                dout_proj_weight, dout_proj_bias,
                dA, dB, dC, dD,
                ddelta_bias if delta_bias is not None else None,
                # 6-None are delta_softplus, checkpoint_lvl, b_rms_weight, c_rms_weight, dt_rms_weight, b_c_dt_rms_eps
                dB_proj_bias, dC_proj_bias, None, None, None, None, None, None)


def mamba_inner_fn(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1, b_rms_weight= None, c_rms_weight= None, dt_rms_weight= None, b_c_dt_rms_eps=1e-6
):
    return MambaInnerFn.apply(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                              out_proj_weight, out_proj_bias,
                              A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus, checkpoint_lvl, b_rms_weight, c_rms_weight, dt_rms_weight, b_c_dt_rms_eps)

# ---------------------------------------------------------------------------
# https://github.com/state-spaces/mamba/blob/bd6dc62a724e6d2f1c821a15b3b2cc02e2b810b4/mamba_ssm/modules/mamba_simple.py#L31-L294

class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


# --------------------------------------------------------------------------------------------------------------------------------------------
# https://github.com/state-spaces/mamba/blob/bd6dc62a724e6d2f1c821a15b3b2cc02e2b810b4/mamba_ssm/modules/block.py#L10-L91

class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
        else:
            self.mlp = None
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, **mixer_kwargs
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm)
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)

        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps,
                    is_rms_norm=isinstance(self.norm2, RMSNorm)
                )
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


# --------------------------------------------------------------------------------------------------------------------------------------------
# https://github.com/state-spaces/mamba/blob/bd6dc62a724e6d2f1c821a15b3b2cc02e2b810b4/mamba_ssm/models/mixer_seq_simple.py#L85-L115

# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


# --------------------------------------------------------------------------------------------------------------------------------------------
# https://github.com/RoyChao19477/SEMamba/blob/750ff5d8f0e112557ca039c268cdcc51e267f342/models/mamba_block.py#L15-L109
# Edited: add mlp_cls=nn.Identity in Block

# github: https://github.com/state-spaces/mamba/blob/9127d1f47f367f5c9cc49c73ad73557089d02cb8/mamba_ssm/models/mixer_seq_simple.py
def create_block(
    d_model, cfg, layer_idx=0, rms_norm=True, fused_add_norm=False, residual_in_fp32=False, 
    ):
    d_state = cfg['model_cfg']['d_state'] # 16
    d_conv = cfg['model_cfg']['d_conv'] # 4
    expand = cfg['model_cfg']['expand'] # 4
    norm_epsilon = cfg['model_cfg']['norm_epsilon'] # 0.00001

    mixer_cls = partial(Mamba, layer_idx=layer_idx, d_state=d_state, d_conv=d_conv, expand=expand)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon
    )
    block = Block(
            d_model,
            mixer_cls,
            mlp_cls=nn.Identity,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            )
    block.layer_idx = layer_idx
    return block

class MambaBlock(nn.Module):
    def __init__(self, in_channels, cfg):
        super(MambaBlock, self).__init__()
        n_layer = 1
        self.forward_blocks  = nn.ModuleList( create_block(in_channels, cfg) for i in range(n_layer) )
        self.backward_blocks = nn.ModuleList( create_block(in_channels, cfg) for i in range(n_layer) )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
            )
        )

    def forward(self, x):
        x_forward, x_backward = x.clone(), torch.flip(x, [1])
        resi_forward, resi_backward = None, None

        # Forward
        for layer in self.forward_blocks:
            x_forward, resi_forward = layer(x_forward, resi_forward)
        y_forward = (x_forward + resi_forward) if resi_forward is not None else x_forward

        # Backward
        for layer in self.backward_blocks:
            x_backward, resi_backward = layer(x_backward, resi_backward)
        y_backward = torch.flip((x_backward + resi_backward), [1]) if resi_backward is not None else torch.flip(x_backward, [1])

        return torch.cat([y_forward, y_backward], -1)

class TFMambaBlock(nn.Module):
    """
    Temporal-Frequency Mamba block for sequence modeling.
    
    Attributes:
    cfg (Config): Configuration for the block.
    time_mamba (MambaBlock): Mamba block for temporal dimension.
    freq_mamba (MambaBlock): Mamba block for frequency dimension.
    tlinear (ConvTranspose1d): ConvTranspose1d layer for temporal dimension.
    flinear (ConvTranspose1d): ConvTranspose1d layer for frequency dimension.
    """
    def __init__(self, cfg):
        super(TFMambaBlock, self).__init__()
        self.cfg = cfg
        self.hid_feature = cfg['model_cfg']['hid_feature']
        
        # Initialize Mamba blocks
        self.time_mamba = MambaBlock(in_channels=self.hid_feature, cfg=cfg)
        self.freq_mamba = MambaBlock(in_channels=self.hid_feature, cfg=cfg)
        
        # Initialize ConvTranspose1d layers
        self.tlinear = nn.ConvTranspose1d(self.hid_feature * 2, self.hid_feature, 1, stride=1)
        self.flinear = nn.ConvTranspose1d(self.hid_feature * 2, self.hid_feature, 1, stride=1)
    
    def forward(self, x):
        """
        Forward pass of the TFMamba block.
        
        Parameters:
        x (Tensor): Input tensor with shape (batch, channels, time, freq).
        
        Returns:
        Tensor: Output tensor after applying temporal and frequency Mamba blocks.
        """
        b, c, t, f = x.size()

        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.tlinear( self.time_mamba(x).permute(0,2,1) ).permute(0,2,1) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.flinear( self.freq_mamba(x).permute(0,2,1) ).permute(0,2,1) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x


# --------------------------------------------------------------------------------------------------------------------------------------------
# https://github.com/RoyChao19477/SEMamba/blob/750ff5d8f0e112557ca039c268cdcc51e267f342/models/lsigmoid.py#L37-L66

class LearnableSigmoid2D(nn.Module):
    """
    Learnable Sigmoid Activation Function for 2D inputs.
    
    This module applies a learnable slope parameter to the sigmoid activation function for 2D inputs.
    """
    def __init__(self, in_features, beta=1):
        """
        Initialize the LearnableSigmoid2D module.
        
        Args:
        - in_features (int): Number of input features.
        - beta (float, optional): Scaling factor for the sigmoid function. Defaults to 1.
        """
        super(LearnableSigmoid2D, self).__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requires_grad = True

    def forward(self, x):
        """
        Forward pass for the LearnableSigmoid2D module.
        
        Args:
        - x (torch.Tensor): Input tensor.
        
        Returns:
        - torch.Tensor: Output tensor after applying the learnable sigmoid activation.
        """
        return self.beta * torch.sigmoid(self.slope * x)


# --------------------------------------------------------------------------------------------------------------------------------------------
# https://github.com/RoyChao19477/SEMamba/blob/750ff5d8f0e112557ca039c268cdcc51e267f342/models/codec_module.py#L8-L183

def get_padding(kernel_size, dilation=1):
    """
    Calculate the padding size for a convolutional layer.
    
    Args:
    - kernel_size (int): Size of the convolutional kernel.
    - dilation (int, optional): Dilation rate of the convolution. Defaults to 1.
    
    Returns:
    - int: Calculated padding size.
    """
    return int((kernel_size * dilation - dilation) / 2)

def get_padding_2d(kernel_size, dilation=(1, 1)):
    """
    Calculate the padding size for a 2D convolutional layer.
    
    Args:
    - kernel_size (tuple): Size of the convolutional kernel (height, width).
    - dilation (tuple, optional): Dilation rate of the convolution (height, width). Defaults to (1, 1).
    
    Returns:
    - tuple: Calculated padding size (height, width).
    """
    return (int((kernel_size[0] * dilation[0] - dilation[0]) / 2), 
            int((kernel_size[1] * dilation[1] - dilation[1]) / 2))

class DenseBlock(nn.Module):
    """
    DenseBlock module consisting of multiple convolutional layers with dilation.
    """
    def __init__(self, cfg, kernel_size=(3, 3), depth=4):
        super(DenseBlock, self).__init__()
        self.cfg = cfg
        self.depth = depth
        self.dense_block = nn.ModuleList()
        self.hid_feature = cfg['model_cfg']['hid_feature']

        for i in range(depth):
            dil = 2 ** i
            dense_conv = nn.Sequential(
                nn.Conv2d(self.hid_feature * (i + 1), self.hid_feature, kernel_size, 
                          dilation=(dil, 1), padding=get_padding_2d(kernel_size, (dil, 1))),
                nn.InstanceNorm2d(self.hid_feature, affine=True),
                nn.PReLU(self.hid_feature)
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        """
        Forward pass for the DenseBlock module.
        
        Args:
        - x (torch.Tensor): Input tensor.
        
        Returns:
        - torch.Tensor: Output tensor after processing through the dense block.
        """
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x

class DenseEncoder(nn.Module):
    """
    DenseEncoder module consisting of initial convolution, dense block, and a final convolution.
    """
    def __init__(self, cfg):
        super(DenseEncoder, self).__init__()
        self.cfg = cfg
        self.input_channel = cfg['model_cfg']['input_channel']
        self.hid_feature = cfg['model_cfg']['hid_feature']

        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(self.input_channel, self.hid_feature, (1, 1)),
            nn.InstanceNorm2d(self.hid_feature, affine=True),
            nn.PReLU(self.hid_feature)
        )

        self.dense_block = DenseBlock(cfg, depth=4)

        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(self.hid_feature, self.hid_feature, (1, 3), stride=(1, 2)),
            nn.InstanceNorm2d(self.hid_feature, affine=True),
            nn.PReLU(self.hid_feature)
        )

    def forward(self, x):
        """
        Forward pass for the DenseEncoder module.
        
        Args:
        - x (torch.Tensor): Input tensor.
        
        Returns:
        - torch.Tensor: Encoded tensor.
        """
        x = self.dense_conv_1(x)  # [batch, hid_feature, time, freq]
        x = self.dense_block(x)   # [batch, hid_feature, time, freq]
        x = self.dense_conv_2(x)  # [batch, hid_feature, time, freq//2]
        return x

class MagDecoder(nn.Module):
    """
    MagDecoder module for decoding magnitude information.
    """
    def __init__(self, cfg):
        super(MagDecoder, self).__init__()
        self.dense_block = DenseBlock(cfg, depth=4)
        self.hid_feature = cfg['model_cfg']['hid_feature']
        self.output_channel = cfg['model_cfg']['output_channel']
        self.n_fft = cfg['stft_cfg']['n_fft']
        self.beta = cfg['model_cfg']['beta']

        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(self.hid_feature, self.hid_feature, (1, 3), stride=(1, 2)),
            nn.Conv2d(self.hid_feature, self.output_channel, (1, 1)),
            nn.InstanceNorm2d(self.output_channel, affine=True),
            nn.PReLU(self.output_channel),
            nn.Conv2d(self.output_channel, self.output_channel, (1, 1))
        )
        self.lsigmoid = LearnableSigmoid2D(self.n_fft // 2 + 1, beta=self.beta)

    def forward(self, x):
        """
        Forward pass for the MagDecoder module.
        
        Args:
        - x (torch.Tensor): Input tensor.
        
        Returns:
        - torch.Tensor: Decoded tensor with magnitude information.
        """
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = rearrange(x, 'b c t f -> b f t c').squeeze(-1)
        x = self.lsigmoid(x)
        x = rearrange(x, 'b f t -> b t f').unsqueeze(1)
        return x

class PhaseDecoder(nn.Module):
    """
    PhaseDecoder module for decoding phase information.
    """
    def __init__(self, cfg):
        super(PhaseDecoder, self).__init__()
        self.dense_block = DenseBlock(cfg, depth=4)
        self.hid_feature = cfg['model_cfg']['hid_feature']
        self.output_channel = cfg['model_cfg']['output_channel']

        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(self.hid_feature, self.hid_feature, (1, 3), stride=(1, 2)),
            nn.InstanceNorm2d(self.hid_feature, affine=True),
            nn.PReLU(self.hid_feature)
        )

        self.phase_conv_r = nn.Conv2d(self.hid_feature, self.output_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(self.hid_feature, self.output_channel, (1, 1))

    def forward(self, x):
        """
        Forward pass for the PhaseDecoder module.
        
        Args:
        - x (torch.Tensor): Input tensor.
        
        Returns:
        - torch.Tensor: Decoded tensor with phase information.
        """
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x


# --------------------------------------------------------------------------------------------------------------------------------------------
# https://github.com/RoyChao19477/SEMamba/blob/750ff5d8f0e112557ca039c268cdcc51e267f342/models/stfts.py#L4-L73

def mag_phase_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True, addeps=False):
    """
    Compute magnitude and phase using STFT.

    Args:
        y (torch.Tensor): Input audio signal.
        n_fft (int): FFT size.
        hop_size (int): Hop size.
        win_size (int): Window size.
        compress_factor (float, optional): Magnitude compression factor. Defaults to 1.0.
        center (bool, optional): Whether to center the signal before padding. Defaults to True.
        eps (bool, optional): Whether adding epsilon to magnitude and phase or not. Defaults to False. 

    Returns:
        tuple: Magnitude, phase, and complex representation of the STFT.
    """
    #eps = torch.finfo(y.dtype).eps
    eps = 1e-10
    hann_window = torch.hann_window(win_size).to(y.device)
    stft_spec = torch.stft(
                    y, n_fft, 
                    hop_length=hop_size, 
                    win_length=win_size, 
                    window=hann_window,
                    center=center, 
                    pad_mode='reflect', 
                    normalized=False, 
                    return_complex=True)

    if addeps==False:
        mag = torch.abs(stft_spec)
        pha = torch.angle(stft_spec)
    else:
        real_part = stft_spec.real
        imag_part = stft_spec.imag
        mag = torch.sqrt(real_part.pow(2) + imag_part.pow(2) + eps)
        pha = torch.atan2(imag_part + eps, real_part + eps)
    # Compress the magnitude
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)
    return mag, pha, com


def mag_phase_istft(mag, pha, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    """
    Inverse STFT to reconstruct the audio signal from magnitude and phase.

    Args:
        mag (torch.Tensor): Magnitude of the STFT.
        pha (torch.Tensor): Phase of the STFT.
        n_fft (int): FFT size.
        hop_size (int): Hop size.
        win_size (int): Window size.
        compress_factor (float, optional): Magnitude compression factor. Defaults to 1.0.
        center (bool, optional): Whether to center the signal before padding. Defaults to True.

    Returns:
        torch.Tensor: Reconstructed audio signal.
    """
    mag = torch.pow(mag, 1.0 / compress_factor)
    com = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)
    wav = torch.istft(
                    com, 
                    n_fft, 
                    hop_length=hop_size, 
                    win_length=win_size, 
                    window=hann_window, 
                    center=center)
    return wav

# --------------------------------------------------------------------------------------------------------------------------------------------
# https://github.com/RoyChao19477/SEMamba/blob/750ff5d8f0e112557ca039c268cdcc51e267f342/models/generator.py#L7-L72

class SEMamba(nn.Module):
    """
    SEMamba model for speech enhancement using Mamba blocks.
    
    This model uses a dense encoder, multiple Mamba blocks, and separate magnitude
    and phase decoders to process noisy magnitude and phase inputs.
    """
    def __init__(self, cfg):
        """
        Initialize the SEMamba model.
        
        Args:
        - cfg: Configuration object containing model parameters.
        """
        super(SEMamba, self).__init__()
        self.cfg = cfg
        self.num_tscblocks = cfg['model_cfg']['num_tfmamba'] if cfg['model_cfg']['num_tfmamba'] is not None else 4  # default tfmamba: 4

        # Initialize dense encoder
        self.dense_encoder = DenseEncoder(cfg)

        # Initialize Mamba blocks
        self.TSMamba = nn.ModuleList([TFMambaBlock(cfg) for _ in range(self.num_tscblocks)])

        # Initialize decoders
        self.mask_decoder = MagDecoder(cfg)
        self.phase_decoder = PhaseDecoder(cfg)

    def forward(self, noisy_mag, noisy_pha):
        """
        Forward pass for the SEMamba model.
        
        Args:
        - noisy_mag (torch.Tensor): Noisy magnitude input tensor [B, F, T].
        - noisy_pha (torch.Tensor): Noisy phase input tensor [B, F, T].
        
        Returns:
        - denoised_mag (torch.Tensor): Denoised magnitude tensor [B, F, T].
        - denoised_pha (torch.Tensor): Denoised phase tensor [B, F, T].
        - denoised_com (torch.Tensor): Denoised complex tensor [B, F, T, 2].
        """
        # Reshape inputs
        noisy_mag = rearrange(noisy_mag, 'b f t -> b t f').unsqueeze(1)  # [B, 1, T, F]
        noisy_pha = rearrange(noisy_pha, 'b f t -> b t f').unsqueeze(1)  # [B, 1, T, F]

        # Concatenate magnitude and phase inputs
        x = torch.cat((noisy_mag, noisy_pha), dim=1)  # [B, 2, T, F]

        # Encode input
        x = self.dense_encoder(x)

        # Apply Mamba blocks
        for block in self.TSMamba:
            x = block(x)

        # Decode magnitude and phase
        denoised_mag = rearrange(self.mask_decoder(x) * noisy_mag, 'b c t f -> b f t c').squeeze(-1)
        denoised_pha = rearrange(self.phase_decoder(x), 'b c t f -> b f t c').squeeze(-1)

        # Combine denoised magnitude and phase into a complex representation
        denoised_com = torch.stack(
            (denoised_mag * torch.cos(denoised_pha), denoised_mag * torch.sin(denoised_pha)),
            dim=-1
        )

        return denoised_mag, denoised_pha, denoised_com

# --------------------------------------------------------------------------------------------------------------------------------------------

    def enhance(self, y):
        with torch.no_grad():
            # If mono expected (librosa.load defaults to mono=True)
            if y.shape[0] > 1:
                y = y.mean(dim=0, keepdim=True)

            noisy_wav = y.squeeze(0).to('cuda')  
            norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to('cuda')
            noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
            noisy_amp, noisy_pha, _ = mag_phase_stft(noisy_wav, n_fft=400, hop_size=100, win_size=400, compress_factor=0.3)

            amp_g, pha_g, _ = self(noisy_amp, noisy_pha)
            
            x_pred = mag_phase_istft(amp_g, pha_g, n_fft=400, hop_size=100, win_size=400, compress_factor=0.3)
            x_pred = x_pred / norm_factor

            # Adjust x_pred length to match y if needed (crop or pad with zeros)
            if x_pred.shape[1] > y.shape[1]:
                x_pred = x_pred[:, :y.shape[1]]
            elif x_pred.shape[1] < y.shape[1]:
                padding = y.shape[1] - x_pred.shape[1]
                x_pred = torch.nn.functional.pad(x_pred, (0, padding))

            return x_pred

