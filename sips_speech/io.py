# Copyright (c) 2026 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Audio I/O helpers for SIPS."""

import warnings

import soundfile as sf
import torch

_MULTICHANNEL_WARNING_MESSAGE = "Inference received a multi-channel WAV file; only the first channel will be used."
_warned_multichannel_keys = set()


def load_mono_wav(
    path: str,
    *,
    warn_multichannel_once: bool = False,
    multichannel_warning_message: str = _MULTICHANNEL_WARNING_MESSAGE,
    stacklevel: int = 2,
):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] > 1 and warn_multichannel_once:
        warning_key = multichannel_warning_message
        if warning_key not in _warned_multichannel_keys:
            _warned_multichannel_keys.add(warning_key)
            warnings.warn(
                multichannel_warning_message,
                RuntimeWarning,
                stacklevel=stacklevel,
            )
    audio = torch.from_numpy(audio[:, :1].T.copy())  # [1, T]
    return audio, sr
