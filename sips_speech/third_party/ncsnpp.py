# Copyright (c) 2026 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) 2022 Signal Processing (SP), Universität Hamburg
# Copyright (c) 2020 The Google Research Authors
# Copyright (c) 2019 Samuel G. Fadel
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT
# SPDX-License-Identifier: Apache-2.0
# 
# ================================================================
# Third-Party Code Notice
# ================================================================
# This file includes code taken from the following repository:
#
# StoRM
# https://github.com/sp-uhh/storm
# Specific files:
# - https://github.com/sp-uhh/storm/blob/257e9636a7251ca40aa200753d5c0fe918e31879/sgmse/model.py
# - https://github.com/sp-uhh/storm/blob/257e9636a7251ca40aa200753d5c0fe918e31879/sgmse/data_module.py
# - https://github.com/fadel/pytorch_ema/blob/d661859ad4206217afb274be709837d19218b3f8/torch_ema/ema.py
#
# Copyright (c) 2022 Signal Processing (SP), Universität Hamburg
#
# Licensed under MIT License.
# 
# Score-Based Generative Modeling through Stochastic Differential Equations
# https://github.com/yang-song/score_sde_pytorch
# 
# Specific files:
# - https://github.com/yang-song/score_sde_pytorch/blob/cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44/op/upfirdn2d.py
# - https://github.com/yang-song/score_sde_pytorch/blob/cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44/models/up_or_down_sampling.py
# - https://github.com/yang-song/score_sde_pytorch/blob/cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44/models/layers.py
# - https://github.com/yang-song/score_sde_pytorch/blob/cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44/models/layerspp.py
# 
# Copyright (c) 2020 The Google Research Authors
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Pytorch EMA
# https://github.com/fadel/pytorch_ema
# 
# Specific file:
# https://github.com/fadel/pytorch_ema/blob/d661859ad4206217afb274be709837d19218b3f8/torch_ema/ema.py
# 
# Copyright (c) 2019 Samuel G. Fadel
# 
# Licensed under the MIT License.
# 
# ================================================================


import contextlib
import copy
import functools
import string
import weakref
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Adapted from https://github.com/yang-song/score_sde_pytorch/blob/cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44/op/upfirdn2d.py#L145
# We use the native PyTorch implementation of upfirdn2d instead of the custom CUDA 
# implementation used in the original code. The native implementation is less efficient 
# but still fast enough for our purposes.

def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    out = upfirdn2d_native(
        input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1]
    )
    return out


# --------------------------------------------------------------------------------------
# https://github.com/yang-song/score_sde_pytorch/blob/cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44/op/upfirdn2d.py#L159-L200

def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.view(-1, channel, out_h, out_w)


# --------------------------------------------------------------------------------------
# https://github.com/yang-song/score_sde_pytorch/blob/cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44/models/up_or_down_sampling.py#L72-L257

def upsample_conv_2d(x, w, k=None, factor=2, gain=1):
  """Fused `upsample_2d()` followed by `tf.nn.conv2d()`.

     Padding is performed only once at the beginning, not between the
     operations.
     The fused op is considerably more efficient than performing the same
     calculation
     using standard TensorFlow ops. It supports gradients of arbitrary order.
     Args:
       x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
         C]`.
       w:            Weight tensor of the shape `[filterH, filterW, inChannels,
         outChannels]`. Grouped convolution can be performed by `inChannels =
         x.shape[0] // numGroups`.
       k:            FIR filter of the shape `[firH, firW]` or `[firN]`
         (separable). The default is `[1] * factor`, which corresponds to
         nearest-neighbor upsampling.
       factor:       Integer upsampling factor (default: 2).
       gain:         Scaling factor for signal magnitude (default: 1.0).

     Returns:
       Tensor of the shape `[N, C, H * factor, W * factor]` or
       `[N, H * factor, W * factor, C]`, and same datatype as `x`.
  """

  assert isinstance(factor, int) and factor >= 1

  # Check weight shape.
  assert len(w.shape) == 4
  convH = w.shape[2]
  convW = w.shape[3]
  inC = w.shape[1]
  outC = w.shape[0]

  assert convW == convH

  # Setup filter kernel.
  if k is None:
    k = [1] * factor
  k = _setup_kernel(k) * (gain * (factor ** 2))
  p = (k.shape[0] - factor) - (convW - 1)

  stride = (factor, factor)

  # Determine data dimensions.
  stride = [1, 1, factor, factor]
  output_shape = ((_shape(x, 2) - 1) * factor + convH, (_shape(x, 3) - 1) * factor + convW)
  output_padding = (output_shape[0] - (_shape(x, 2) - 1) * stride[0] - convH,
                    output_shape[1] - (_shape(x, 3) - 1) * stride[1] - convW)
  assert output_padding[0] >= 0 and output_padding[1] >= 0
  num_groups = _shape(x, 1) // inC

  # Transpose weights.
  w = torch.reshape(w, (num_groups, -1, inC, convH, convW))
  w = w[..., ::-1, ::-1].permute(0, 2, 1, 3, 4)
  w = torch.reshape(w, (num_groups * inC, -1, convH, convW))

  x = F.conv_transpose2d(x, w, stride=stride, output_padding=output_padding, padding=0)
  ## Original TF code.
  # x = tf.nn.conv2d_transpose(
  #     x,
  #     w,
  #     output_shape=output_shape,
  #     strides=stride,
  #     padding='VALID',
  #     data_format=data_format)
  ## JAX equivalent

  return upfirdn2d(x, torch.tensor(k, device=x.device),
                   pad=((p + 1) // 2 + factor - 1, p // 2 + 1))


def conv_downsample_2d(x, w, k=None, factor=2, gain=1):
  """Fused `tf.nn.conv2d()` followed by `downsample_2d()`.

    Padding is performed only once at the beginning, not between the operations.
    The fused op is considerably more efficient than performing the same
    calculation
    using standard TensorFlow ops. It supports gradients of arbitrary order.
    Args:
        x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
          C]`.
        w:            Weight tensor of the shape `[filterH, filterW, inChannels,
          outChannels]`. Grouped convolution can be performed by `inChannels =
          x.shape[0] // numGroups`.
        k:            FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to
          average pooling.
        factor:       Integer downsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of the shape `[N, C, H // factor, W // factor]` or
        `[N, H // factor, W // factor, C]`, and same datatype as `x`.
  """

  assert isinstance(factor, int) and factor >= 1
  _outC, _inC, convH, convW = w.shape
  assert convW == convH
  if k is None:
    k = [1] * factor
  k = _setup_kernel(k) * gain
  p = (k.shape[0] - factor) + (convW - 1)
  s = [factor, factor]
  x = upfirdn2d(x, torch.tensor(k, device=x.device),
                pad=((p + 1) // 2, p // 2))
  return F.conv2d(x, w, stride=s, padding=0)


def _setup_kernel(k):
  k = np.asarray(k, dtype=np.float32)
  if k.ndim == 1:
    k = np.outer(k, k)
  k /= np.sum(k)
  assert k.ndim == 2
  assert k.shape[0] == k.shape[1]
  return k


def _shape(x, dim):
  return x.shape[dim]


def upsample_2d(x, k=None, factor=2, gain=1):
  r"""Upsample a batch of 2D images with the given filter.

    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]`
    and upsamples each image with the given filter. The filter is normalized so
    that
    if the input pixels are constant, they will be scaled by the specified
    `gain`.
    Pixels outside the image are assumed to be zero, and the filter is padded
    with
    zeros so that its shape is a multiple of the upsampling factor.
    Args:
        x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
          C]`.
        k:            FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to
          nearest-neighbor upsampling.
        factor:       Integer upsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of the shape `[N, C, H * factor, W * factor]`
  """
  assert isinstance(factor, int) and factor >= 1
  if k is None:
    k = [1] * factor
  k = _setup_kernel(k) * (gain * (factor ** 2))
  p = k.shape[0] - factor
  return upfirdn2d(x, torch.tensor(k, device=x.device),
                   up=factor, pad=((p + 1) // 2 + factor - 1, p // 2))


def downsample_2d(x, k=None, factor=2, gain=1):
  r"""Downsample a batch of 2D images with the given filter.

    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]`
    and downsamples each image with the given filter. The filter is normalized
    so that
    if the input pixels are constant, they will be scaled by the specified
    `gain`.
    Pixels outside the image are assumed to be zero, and the filter is padded
    with
    zeros so that its shape is a multiple of the downsampling factor.
    Args:
        x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
          C]`.
        k:            FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to
          average pooling.
        factor:       Integer downsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of the shape `[N, C, H // factor, W // factor]`
  """

  assert isinstance(factor, int) and factor >= 1
  if k is None:
    k = [1] * factor
  k = _setup_kernel(k) * gain
  p = k.shape[0] - factor
  return upfirdn2d(x, torch.tensor(k, device=x.device),
                   down=factor, pad=((p + 1) // 2, p // 2))


# --------------------------------------------------------------------------------------
# https://github.com/yang-song/score_sde_pytorch/blob/cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44/models/up_or_down_sampling.py#L13-L56

# Function ported from StyleGAN2
def get_weight(module,
               shape,
               weight_var='weight',
               kernel_init=None):
  """Get/create weight tensor for a convolution or fully-connected layer."""

  return module.param(weight_var, kernel_init, shape)


class Conv2d(nn.Module):
  """Conv2d layer with optimal upsampling and downsampling (StyleGAN2)."""

  def __init__(self, in_ch, out_ch, kernel, up=False, down=False,
               resample_kernel=(1, 3, 3, 1),
               use_bias=True,
               kernel_init=None):
    super().__init__()
    assert not (up and down)
    assert kernel >= 1 and kernel % 2 == 1
    self.weight = nn.Parameter(torch.zeros(out_ch, in_ch, kernel, kernel))
    if kernel_init is not None:
      self.weight.data = kernel_init(self.weight.data.shape)
    if use_bias:
      self.bias = nn.Parameter(torch.zeros(out_ch))

    self.up = up
    self.down = down
    self.resample_kernel = resample_kernel
    self.kernel = kernel
    self.use_bias = use_bias

  def forward(self, x):
    if self.up:
      x = upsample_conv_2d(x, self.weight, k=self.resample_kernel)
    elif self.down:
      x = conv_downsample_2d(x, self.weight, k=self.resample_kernel)
    else:
      x = F.conv2d(x, self.weight, stride=1, padding=self.kernel // 2)

    if self.use_bias:
      x = x + self.bias.reshape(1, -1, 1, 1)

    return x


# --------------------------------------------------------------------------------------
# https://github.com/yang-song/score_sde_pytorch/blob/cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44/models/layers.py#L54-L91

def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
  """Ported from JAX. """

  def _compute_fans(shape, in_axis=1, out_axis=0):
    receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
    fan_in = shape[in_axis] * receptive_field_size
    fan_out = shape[out_axis] * receptive_field_size
    return fan_in, fan_out

  def init(shape, dtype=dtype, device=device):
    fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
    if mode == "fan_in":
      denominator = fan_in
    elif mode == "fan_out":
      denominator = fan_out
    elif mode == "fan_avg":
      denominator = (fan_in + fan_out) / 2
    else:
      raise ValueError(
        "invalid mode for variance scaling initializer: {}".format(mode))
    variance = scale / denominator
    if distribution == "normal":
      return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
    elif distribution == "uniform":
      return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
    else:
      raise ValueError("invalid distribution for variance scaling initializer")

  return init


def default_init(scale=1.):
  """The same initialization used in DDPM."""
  scale = 1e-10 if scale == 0 else scale
  return variance_scaling(scale, 'fan_avg', 'uniform')


# --------------------------------------------------------------------------------------
# https://github.com/yang-song/score_sde_pytorch/blob/cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44/models/layers.py#L532-L555

def _einsum(a, b, c, x, y):
  einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
  return torch.einsum(einsum_str, x, y)


def contract_inner(x, y):
  """tensordot(x, y, 1)."""
  x_chars = list(string.ascii_lowercase[:len(x.shape)])
  y_chars = list(string.ascii_lowercase[len(x.shape):len(y.shape) + len(x.shape)])
  y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
  out_chars = x_chars[:-1] + y_chars[1:]
  return _einsum(x_chars, y_chars, out_chars, x, y)


class NIN(nn.Module):
  def __init__(self, in_dim, num_units, init_scale=0.1):
    super().__init__()
    self.W = nn.Parameter(default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True)
    self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

  def forward(self, x):
    x = x.permute(0, 2, 3, 1)
    y = contract_inner(x, self.W) + self.b
    return y.permute(0, 3, 1, 2)


# --------------------------------------------------------------------------------------
# https://github.com/yang-song/score_sde_pytorch/blob/cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44/models/layers.py#L100-L105

def ddpm_conv1x1(in_planes, out_planes, stride=1, bias=True, init_scale=1., padding=0):
  """1x1 convolution with DDPM initialization."""
  conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=bias)
  conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
  nn.init.zeros_(conv.bias)
  return conv


# --------------------------------------------------------------------------------------
# https://github.com/yang-song/score_sde_pytorch/blob/cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44/models/layers.py#L118-L124

def ddpm_conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1., padding=1):
  """3x3 convolution with DDPM initialization."""
  conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding,
                   dilation=dilation, bias=bias)
  conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
  nn.init.zeros_(conv.bias)
  return conv


# --------------------------------------------------------------------------------------
# Adapted from https://github.com/yang-song/score_sde_pytorch/blob/cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44/models/layerspp.py#L26-L27

conv1x1 = ddpm_conv1x1
conv3x3 = ddpm_conv3x3


# --------------------------------------------------------------------------------------
# https://github.com/yang-song/score_sde_pytorch/blob/cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44/models/layerspp.py#L26-L163
# Edited: up_or_down_sampling.Conv2d to Conv2d
# Edited: up_or_down_sampling.upsample_2d to upsample_2d
# Edited: up_or_down_sampling.downsample_2d to downsample_2d

class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""

  def __init__(self, embedding_size=256, scale=1.0):
    super().__init__()
    self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Combine(nn.Module):
  """Combine information from skip connections."""

  def __init__(self, dim1, dim2, method='cat'):
    super().__init__()
    self.Conv_0 = conv1x1(dim1, dim2)
    self.method = method

  def forward(self, x, y):
    h = self.Conv_0(x)
    if self.method == 'cat':
      return torch.cat([h, y], dim=1)
    elif self.method == 'sum':
      return h + y
    else:
      raise ValueError(f'Method {self.method} not recognized.')


class AttnBlockpp(nn.Module):
  """Channel-wise self-attention block. Modified from DDPM."""

  def __init__(self, channels, skip_rescale=False, init_scale=0.):
    super().__init__()
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels,
                                  eps=1e-6)
    self.NIN_0 = NIN(channels, channels)
    self.NIN_1 = NIN(channels, channels)
    self.NIN_2 = NIN(channels, channels)
    self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
    self.skip_rescale = skip_rescale

  def forward(self, x):
    B, C, H, W = x.shape
    h = self.GroupNorm_0(x)
    q = self.NIN_0(h)
    k = self.NIN_1(h)
    v = self.NIN_2(h)

    w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))
    w = torch.reshape(w, (B, H, W, H * W))
    w = F.softmax(w, dim=-1)
    w = torch.reshape(w, (B, H, W, H, W))
    h = torch.einsum('bhwij,bcij->bchw', w, v)
    h = self.NIN_3(h)
    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)


class Upsample(nn.Module):
  def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
               fir_kernel=(1, 3, 3, 1)):
    super().__init__()
    out_ch = out_ch if out_ch else in_ch
    if not fir:
      if with_conv:
        self.Conv_0 = conv3x3(in_ch, out_ch)
    else:
      if with_conv:
        self.Conv2d_0 = Conv2d(in_ch, out_ch,
                                                 kernel=3, up=True,
                                                 resample_kernel=fir_kernel,
                                                 use_bias=True,
                                                 kernel_init=default_init())
    self.fir = fir
    self.with_conv = with_conv
    self.fir_kernel = fir_kernel
    self.out_ch = out_ch

  def forward(self, x):
    B, C, H, W = x.shape
    if not self.fir:
      h = F.interpolate(x, (H * 2, W * 2), 'nearest')
      if self.with_conv:
        h = self.Conv_0(h)
    else:
      if not self.with_conv:
        h = upsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = self.Conv2d_0(x)

    return h


class Downsample(nn.Module):
  def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
               fir_kernel=(1, 3, 3, 1)):
    super().__init__()
    out_ch = out_ch if out_ch else in_ch
    if not fir:
      if with_conv:
        self.Conv_0 = conv3x3(in_ch, out_ch, stride=2, padding=0)
    else:
      if with_conv:
        self.Conv2d_0 = Conv2d(in_ch, out_ch,
                                                 kernel=3, down=True,
                                                 resample_kernel=fir_kernel,
                                                 use_bias=True,
                                                 kernel_init=default_init())
    self.fir = fir
    self.fir_kernel = fir_kernel
    self.with_conv = with_conv
    self.out_ch = out_ch

  def forward(self, x):
    B, C, H, W = x.shape
    if not self.fir:
      if self.with_conv:
        x = F.pad(x, (0, 1, 0, 1))
        x = self.Conv_0(x)
      else:
        x = F.avg_pool2d(x, 2, stride=2)
    else:
      if not self.with_conv:
        x = downsample_2d(x, self.fir_kernel, factor=2)
      else:
        x = self.Conv2d_0(x)

    return x
  

# --------------------------------------------------------------------------------------
# https://github.com/yang-song/score_sde_pytorch/blob/cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44/models/layerspp.py#L212-L274
# Edited: up_or_down_sampling.upsample_2d to upsample_2d
# Edited: up_or_down_sampling.downsample_2d to downsample_2d
# Edited: Removed cases for self.rir == False

class ResnetBlockBigGANpp(nn.Module):
  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, up=False, down=False,
               dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1),
               skip_rescale=True, init_scale=0.):
    super().__init__()

    out_ch = out_ch if out_ch else in_ch
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
    self.up = up
    self.down = down
    self.fir = fir
    self.fir_kernel = fir_kernel

    self.Conv_0 = conv3x3(in_ch, out_ch)
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(temb_dim, out_ch)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
      nn.init.zeros_(self.Dense_0.bias)

    self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
    self.Dropout_0 = nn.Dropout(dropout)
    self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
    if in_ch != out_ch or up or down:
      self.Conv_2 = conv1x1(in_ch, out_ch)

    self.skip_rescale = skip_rescale
    self.act = act
    self.in_ch = in_ch
    self.out_ch = out_ch

  def forward(self, x, temb=None):
    h = self.act(self.GroupNorm_0(x))

    if self.up:
      if self.fir:
        h = upsample_2d(h, self.fir_kernel, factor=2)
        x = upsample_2d(x, self.fir_kernel, factor=2)
      else:
        raise NotImplementedError('Naive upsampling not implemented for now.')
    elif self.down:
      if self.fir:
        h = downsample_2d(h, self.fir_kernel, factor=2)
        x = downsample_2d(x, self.fir_kernel, factor=2)
      else:
        raise NotImplementedError('Naive downsampling not implemented for now.')

    h = self.Conv_0(h)
    # Add bias to each feature map conditioned on the time embedding
    if temb is not None:
      h += self.Dense_0(self.act(temb))[:, :, None, None]
    h = self.act(self.GroupNorm_1(h))
    h = self.Dropout_0(h)
    h = self.Conv_1(h)

    if self.in_ch != self.out_ch or self.up or self.down:
      x = self.Conv_2(x)

    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)


# --------------------------------------------------------------------------------------
# Adapted from https://github.com/sp-uhh/storm/blob/257e9636a7251ca40aa200753d5c0fe918e31879/sgmse/backbones/ncsnpp.py#L37

class NCSNpp(nn.Module):
    def __init__(self, 
        nf = 128,
        ch_mult = (1, 2, 2, 2),
        num_res_blocks = 1,
        attn_resolutions = (0,),
        fir = True,
        fir_kernel = [1, 3, 3, 1],
        init_scale = 0.,
        fourier_scale = 16,
        image_size = 256,
        input_channels = 2,
        spatial_channels = 1,
        dropout = .0,
    ):
        super().__init__()

        self.act = nn.SiLU()
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.input_channels = input_channels
        self.spatial_channels = spatial_channels
        self.total_channels = self.input_channels * self.spatial_channels
        self.output_layer = nn.Conv2d(self.total_channels, 2*self.spatial_channels, 1)
        self.pyramid_upsample = Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        self.pyramid_downsample = Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)

        combiner = functools.partial(Combine, method='sum')
        all_resolutions = [image_size // (2 ** i) for i in range(num_resolutions)]
        AttnBlock = functools.partial(AttnBlockpp, 
            init_scale=init_scale, skip_rescale=True)
        ResnetBlock = functools.partial(ResnetBlockBigGANpp, act=self.act,
            dropout=dropout, fir=fir, fir_kernel=fir_kernel, 
            init_scale=init_scale, skip_rescale=True, temb_dim=nf * 4)

        modules = []
        modules.append(GaussianFourierProjection(
            embedding_size=nf, scale=fourier_scale
        ))

        # Downsampling 
        input_pyramid_ch = self.total_channels

        modules.append(conv3x3(self.total_channels, nf))
        hs_c = [nf]

        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for _ in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                modules.append(ResnetBlock(down=True, in_ch=in_ch))
                modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        # Upsampling
        for i_level in reversed(range(num_resolutions)):
            for _ in range(num_res_blocks + 1):  # +1 blocks in upsampling because of skip connection from combiner
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if i_level == num_resolutions - 1:
                modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), 
                    num_channels=in_ch, eps=1e-6))
                modules.append(conv3x3(in_ch, self.total_channels, init_scale=init_scale))

            else:
                modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                    num_channels=in_ch, eps=1e-6))
                modules.append(conv3x3(in_ch, self.total_channels, bias=True, init_scale=init_scale))

            if i_level != 0:
                modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        self.all_modules = nn.ModuleList(modules)


    def forward(self, x, time_cond=None):
        # timestep/noise_level embedding; only for continuous training
        modules = self.all_modules
        m_idx = 0

        # Convert real and imaginary parts into channel dimensions
        x_chans = []
        for chan in range(self.spatial_channels):
            x_chans.append(torch.cat([ 
                torch.cat([x[:,[chan+in_chan],:,:].real, x[:,[chan+in_chan],:,:].imag ], dim=1) for in_chan in range(self.input_channels // 2)],
                    dim=1)
                )
        x = torch.cat(x_chans, dim=1) #4*D

        # Gaussian Fourier features embeddings.
        m_idx += 1

        temb = None

        # If input data is in [0, 1]
        x = 2 * x - 1.

        # Downsampling 
        input_pyramid = x
        hs = [modules[m_idx](x)]  # Input layer: Conv2d
        m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                # edit: check H dim (-2) not W dim (-1)
                if h.shape[-2] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)

            if i_level != self.num_resolutions - 1:  # Downsampling
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1

                input_pyramid = self.pyramid_downsample(input_pyramid)
                h = modules[m_idx](input_pyramid, h)
                m_idx += 1
                hs.append(h)

        h = hs[-1]
        h = modules[m_idx](h, temb)  # ResNet block
        m_idx += 1
        h = modules[m_idx](h)  # Attention block 
        m_idx += 1
        h = modules[m_idx](h, temb)  # ResNet block
        m_idx += 1

        pyramid = None

        # Upsampling 
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

            # edit: from -1 to -2
            if h.shape[-2] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if i_level == self.num_resolutions - 1:
                pyramid = self.act(modules[m_idx](h))
                m_idx += 1
                pyramid = modules[m_idx](pyramid)
                m_idx += 1
            else:
                pyramid = self.pyramid_upsample(pyramid)
                pyramid_h = self.act(modules[m_idx](h))
                m_idx += 1
                pyramid_h = modules[m_idx](pyramid_h)
                m_idx += 1
                pyramid = pyramid + pyramid_h

            if i_level != 0:
                h = modules[m_idx](h, temb)
                m_idx += 1

        assert not hs

        h = pyramid

        assert m_idx == len(modules)

        # Convert to complex number
        h = self.output_layer(h) #b,D=1,C_out,T
        h = torch.reshape(h, (h.size(0), 2, self.spatial_channels, h.size(2), h.size(3)))
        h = torch.permute(h, (0, 2, 3, 4, 1)).contiguous() # b,2,D,F,T -> b,D,F,T,2
        h = torch.view_as_complex(h) #b,D,F,T
        return h
    

# --------------------------------------------------------------------------------------
# https://github.com/fadel/pytorch_ema/blob/d661859ad4206217afb274be709837d19218b3f8/torch_ema/ema.py#L14-L301

class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.

    Args:
        parameters: Iterable of `torch.nn.Parameter` (typically from
            `model.parameters()`).
            Note that EMA is computed on *all* provided parameters,
            regardless of whether or not they have `requires_grad = True`;
            this allows a single EMA object to be consistantly used even
            if which parameters are trainable changes step to step.

            If you want to some parameters in the EMA, do not pass them
            to the object in the first place. For example:

                ExponentialMovingAverage(
                    parameters=[p for p in model.parameters() if p.requires_grad],
                    decay=0.9
                )

            will ignore parameters that do not require grad.

        decay: The exponential decay.

        use_num_updates: Whether to use number of updates when computing
            averages.
    """
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float,
        use_num_updates: bool = True
    ):
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        parameters = list(parameters)
        self.shadow_params = [
            p.clone().detach()
            for p in parameters
        ]
        self.collected_params = None
        # By maintaining only a weakref to each parameter,
        # we maintain the old GC behaviour of ExponentialMovingAverage:
        # if the model goes out of scope but the ExponentialMovingAverage
        # is kept, no references to the model or its parameters will be
        # maintained, and the model will be cleaned up.
        self._params_refs = [weakref.ref(p) for p in parameters]

    def _get_parameters(
        self,
        parameters: Optional[Iterable[torch.nn.Parameter]]
    ) -> Iterable[torch.nn.Parameter]:
        if parameters is None:
            parameters = [p() for p in self._params_refs]
            if any(p is None for p in parameters):
                raise ValueError(
                    "(One of) the parameters with which this "
                    "ExponentialMovingAverage "
                    "was initialized no longer exists (was garbage collected);"
                    " please either provide `parameters` explicitly or keep "
                    "the model to which they belong from being garbage "
                    "collected."
                )
            return parameters
        else:
            parameters = list(parameters)
            if len(parameters) != len(self.shadow_params):
                raise ValueError(
                    "Number of parameters passed as argument is different "
                    "from number of shadow parameters maintained by this "
                    "ExponentialMovingAverage"
                )
            return parameters

    def update(
        self,
        parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Update currently maintained parameters.

        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; usually the same set of
                parameters used to initialize this object. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(
                decay,
                (1 + self.num_updates) / (10 + self.num_updates)
            )
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            for s_param, param in zip(self.shadow_params, parameters):
                tmp = (s_param - param)
                # tmp will be a new tensor so we can do in-place
                tmp.mul_(one_minus_decay)
                s_param.sub_(tmp)

    def copy_to(
        self,
        parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Copy current averaged parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    def store(
        self,
        parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Save the current parameters for restoring later.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                temporarily stored. If `None`, the parameters of with which this
                `ExponentialMovingAverage` was initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        self.collected_params = [
            param.clone()
            for param in parameters
        ]

    def restore(
        self,
        parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        if self.collected_params is None:
            raise RuntimeError(
                "This ExponentialMovingAverage has no `store()`ed weights "
                "to `restore()`"
            )
        parameters = self._get_parameters(parameters)
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    @contextlib.contextmanager
    def average_parameters(
        self,
        parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ):
        r"""
        Context manager for validation/inference with averaged parameters.

        Equivalent to:

            ema.store()
            ema.copy_to()
            try:
                ...
            finally:
                ema.restore()

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        self.store(parameters)
        self.copy_to(parameters)
        try:
            yield
        finally:
            self.restore(parameters)

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.

        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype)
            if p.is_floating_point()
            else p.to(device=device)
            for p in self.shadow_params
        ]
        if self.collected_params is not None:
            self.collected_params = [
                p.to(device=device, dtype=dtype)
                if p.is_floating_point()
                else p.to(device=device)
                for p in self.collected_params
            ]
        return

    def state_dict(self) -> dict:
        r"""Returns the state of the ExponentialMovingAverage as a dict."""
        # Following PyTorch conventions, references to tensors are returned:
        # "returns a reference to the state and not its copy!" -
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_params": self.shadow_params,
            "collected_params": self.collected_params
        }

    def load_state_dict(self, state_dict: dict) -> None:
        r"""Loads the ExponentialMovingAverage state.

        Args:
            state_dict (dict): EMA state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = copy.deepcopy(state_dict)
        self.decay = state_dict["decay"]
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.num_updates = state_dict["num_updates"]
        assert self.num_updates is None or isinstance(self.num_updates, int), \
            "Invalid num_updates"

        self.shadow_params = state_dict["shadow_params"]
        assert isinstance(self.shadow_params, list), \
            "shadow_params must be a list"
        assert all(
            isinstance(p, torch.Tensor) for p in self.shadow_params
        ), "shadow_params must all be Tensors"

        self.collected_params = state_dict["collected_params"]
        if self.collected_params is not None:
            assert isinstance(self.collected_params, list), \
                "collected_params must be a list"
            assert all(
                isinstance(p, torch.Tensor) for p in self.collected_params
            ), "collected_params must all be Tensors"
            assert len(self.collected_params) == len(self.shadow_params), \
                "collected_params and shadow_params had different lengths"

        if len(self.shadow_params) == len(self._params_refs):
            # Consistant with torch.optim.Optimizer, cast things to consistant
            # device and dtype with the parameters
            params = [p() for p in self._params_refs]
            # If parameters have been garbage collected, just load the state
            # we were given without change.
            if not any(p is None for p in params):
                # ^ parameter references are still good
                for i, p in enumerate(params):
                    self.shadow_params[i] = self.shadow_params[i].to(
                        device=p.device, dtype=p.dtype
                    )
                    if self.collected_params is not None:
                        self.collected_params[i] = self.collected_params[i].to(
                            device=p.device, dtype=p.dtype
                        )
        else:
            raise ValueError(
                "Tried to `load_state_dict()` with the wrong number of "
                "parameters in the saved state."
            )


# --------------------------------------------------------------------------------------
# https://github.com/sp-uhh/storm/blob/257e9636a7251ca40aa200753d5c0fe918e31879/sgmse/util/other.py#L102-L109

def pad_spec(Y):
    T = Y.size(3)
    if T%64 !=0:
        num_pad = 64-T%64
    else:
        num_pad = 0
    pad2d = torch.nn.ZeroPad2d((0, num_pad, 0,0))
    return pad2d(Y)


# --------------------------------------------------------------------------------------
# https://github.com/sp-uhh/storm/blob/257e9636a7251ca40aa200753d5c0fe918e31879/sgmse/data_module.py#L19-L25

def get_window(window_type, window_length):
	if window_type == 'sqrthann':
		return torch.sqrt(torch.hann_window(window_length, periodic=True))
	elif window_type == 'hann':
		return torch.hann_window(window_length, periodic=True)
	else:
		raise NotImplementedError(f"Window type {window_type} not implemented!")


# --------------------------------------------------------------------------------------
# Adapted from https://github.com/sp-uhh/storm/blob/257e9636a7251ca40aa200753d5c0fe918e31879/sgmse/model.py#L320

class NCSNppPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.dnn = NCSNpp()
        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.999)
        self.window = get_window("hann", window_length=510)
        self.windows = {}


    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if mode == False and not no_ema:
            # eval
            self.ema.store(self.parameters())  # store current params in EMA
            self.ema.copy_to(self.parameters())  # copy EMA parameters over current params for evaluation
        else:
            # train
            if self.ema.collected_params is not None:
                self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def to(self, *args, **kwargs):
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def to_audio(self, spec, length=None):
        return self.istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.spec_back(spec)
    
    def _get_window(self, x):
        window = self.windows.get(x.device, None)
        if window is None:
            window = self.window.to(x.device)
            self.windows[x.device] = window
        return window

    @property
    def stft_kwargs(self):
        return {**self.istft_kwargs, "return_complex": True}

    @property
    def istft_kwargs(self):
        return dict(
            n_fft=510, hop_length=128, window=self.window, center=True
        )

    def stft(self, sig):
        window = self._get_window(sig)
        return torch.stft(sig, **{**self.stft_kwargs, "window": window})

    def istft(self, spec, length=None):
        window = self._get_window(spec)
        return torch.istft(spec, **{**self.istft_kwargs, "window": window, "length": length})

    def spec_fwd(self, spec, spec_factor=0.15, spec_abs_exponent=0.5):
        if spec_abs_exponent != 1:
            e = spec_abs_exponent
            spec = spec.abs()**e * torch.exp(1j * spec.angle())
        return spec * spec_factor

    def spec_back(self, spec, spec_factor=0.15, spec_abs_exponent=0.5):
        spec = spec / spec_factor
        if spec_abs_exponent != 1:
            e = spec_abs_exponent
            spec = spec.abs()**(1/e) * torch.exp(1j * spec.angle())
        return spec    
    
    def forward(self, y):
        t = torch.ones(y.shape[0], device=y.device)
        x_hat = self.dnn(y, t)
        return x_hat

    def enhance(self, y, **ignored_kwargs):
        with torch.no_grad():
            norm_factor = y.abs().max().item()
            T_orig = y.size(1)
            Y = self._forward_transform(self.stft((y/norm_factor).cuda())).unsqueeze(0)
            Y = pad_spec(Y)    
            X_hat = self(Y)
            x_hat = self.to_audio(X_hat.squeeze(), T_orig)
            x_hat = (x_hat * norm_factor).unsqueeze(0)
            return x_hat
                    
