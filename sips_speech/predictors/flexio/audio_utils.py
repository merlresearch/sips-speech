# Copyright (C) 2024-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import torch
import torch.nn.functional as F
from scipy import signal


def stft_padding(
    input_signal,
    n_fft,
    window_length,
    hop_length,
    center_pad: bool = True,
    end_pad: bool = True,
    pad_mode: str = "constant",
):
    signal_length = input_signal.shape[-1]
    pad_start = 0
    pad_end = 0
    if center_pad:
        # Do center padding here instead of torch.stft, since we need to do it anyway because they don't
        # support end padding.
        pad_start = int(n_fft // 2)
        pad_end = pad_start
        signal_length = signal_length + pad_start + pad_end
    if end_pad:
        # from scipy.signal.stft
        # Pad to integer number of windowed segments
        # i.e., make signal_length = window_length + (nseg-1)*hop_length, with integer nseg
        nadd = (-(signal_length - window_length) % hop_length) % window_length
        pad_end += nadd

    # do the padding
    signal_dim = input_signal.dim()
    extended_shape = [1] * (3 - signal_dim) + list(input_signal.size())
    input_signal = F.pad(input_signal.view(extended_shape), (pad_start, pad_end), pad_mode)
    input_signal = input_signal.view(input_signal.shape[-signal_dim:])
    return input_signal


def _normalize_options(window, method_str):
    normalize_flag = False
    if method_str == "window":
        window = window / torch.sum(window)
    elif method_str == "default":
        normalize_flag = True
    return window, normalize_flag


def do_stft(
    signal,
    window_length,
    hop_length,
    fft_size: int = None,
    normalize: str = "default",
    window_type: str = "sqrt_hann",
):
    """
    Wrap torch.stft, and return transposed spectrogram for pytorch input

    :param signal: tensor of shape (n_channels, n_samples) or (n_samples)
    :param window_length: int size of stft window
    :param hop_length:  int stride of stft window
    :param fft_size: int geq to window_length, if None set to window_length
    :param normalize: string for determining how to normalize stft output.
                      "window":  divide the window by its sum.  This will give amplitudes of components that match
                                 time domain components, but small values could cause numerical issues
                      "default": default pytorch stft normalization divides by sqrt of window length
                      None:      no normalization of stft outputs
    :return: complex tensor of shape (..., n_frames, n_frequencies) or (n_frames, n_frequencies)
    """
    if fft_size is None:
        fft_size = window_length
    signal = stft_padding(signal, fft_size, window_length, hop_length)
    window = get_window(window=window_type, window_length=window_length, device=signal.device)
    window, normalize_flag = _normalize_options(window, normalize)
    if signal.dim() > 2:
        leading_dims = signal.shape[:-1]
        signal = signal.reshape(-1, signal.size(-1))
    else:
        leading_dims = None

    result = torch.stft(
        signal,
        n_fft=fft_size,
        hop_length=hop_length,
        win_length=window_length,
        window=window,
        normalized=normalize_flag,
        center=False,
        return_complex=True,
    )

    if leading_dims is not None:
        result = result.reshape(leading_dims + result.shape[1:])

    return result.transpose(-1, -2)


def do_istft(
    stft,
    window_length: int = None,
    hop_length: int = None,
    fft_size: int = None,
    normalize: str = "default",
    window_type: str = "sqrt_hann",
):
    """
    Wrap torch.istft and return time domain signal

    :param stft: complex tensor of shape (..., n_frames, n_frequencies) or (n_frames, n_frequencies)
    :param window_length: int size of stft window
    :param hop_length:  int stride of stft window
    :param fft_size: int geq to window_length, if None set to window_length
    :param normalize_window: divide the window by it's sum.  This will give consistent values independent of fft_size,
            but small values could be more susceptible to numerical issues
    :return: tensor of shape (n_samples, n_channels) or (n_samples)
    """

    window = get_window(window=window_type, window_length=window_length, device=stft.device)
    if fft_size is None:
        fft_size = window_length
    window, normalize_flag = _normalize_options(window, normalize)
    # Perfect reconstruction depends on sufficient boundary zeros, a window that can be reused for synthesis, a
    # compatible window_length/hop_length ratio, and the same normalization convention as the forward transform.

    stft = stft.transpose(-1, -2)

    if stft.ndim < 2:
        raise ValueError("STFT input must have at least frame and frequency dimensions")

    if stft.ndim > 2:
        leading_dims = stft.shape[:-2]
        stft = stft.reshape(-1, stft.size(-2), stft.size(-1))
    else:
        leading_dims = None

    signal = torch.istft(
        stft,
        fft_size,
        hop_length=hop_length,
        win_length=window_length,
        window=window,
        center=True,
        normalized=normalize_flag,
    )

    if leading_dims is not None:
        signal = signal.reshape(leading_dims + signal.shape[1:])
    return signal


def get_window(window="sqrt_hann", window_length=1024, device=None):
    if window == "sqrt_hann":
        return sqrt_hann(window_length, device=device)
    elif window == "hann":
        return torch.hann_window(window_length, periodic=True, device=device)
    elif window in ["blackman", "blackmanharris", "hamming"]:
        return torch.tensor(signal.get_window(window, window_length), dtype=torch.float32, device=device)
    else:
        raise ValueError(f"Unsupported window type: {window}")


def sqrt_hann(window_length, device=None):
    """Implement a sqrt-Hann window"""
    return torch.sqrt(torch.hann_window(window_length, periodic=True, device=device))
