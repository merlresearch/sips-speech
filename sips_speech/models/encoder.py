# Copyright (c) 2026 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Spectrogram encoder and decoder for SIPS (Stochastic Interpolants Prior for Speech)
"""

import torch
from einops import rearrange

from sips_speech.third_party.edm2 import const_like

# ---------------------------------------------------------------------------------------
# Spectrogram encoder and decoder.


class SpectrogramEncoder:
    def __init__(
        self,
        sr=16000,
        hop_length=128,
        win_length=512,
        n_fft=512,
        p=0.5,
        b=0.15,
        raw_std=0.08,
        final_std=0.5,
    ):
        super().__init__()

        self.sr = sr
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.p = torch.tensor(p)
        self.b = torch.tensor(b)
        self.scale = float(final_std) / float(raw_std)
        self.window = torch.hann_window(self.win_length, periodic=True)

    def init(self, device):  # force lazy init to happen now
        self.window = self.window.to(device)
        self.b = self.b.to(device)
        self.p = self.p.to(device)

    def amplitude_compression(self, spec):
        # Compress the amplitude of the spectrogram
        spec = self.b * spec.abs() ** self.p * torch.exp(1j * spec.angle())
        return spec

    def amplitude_decompression(self, spec):
        # Decompress the amplitude of the spectrogram
        spec = (spec / self.b).abs() ** (1 / self.p) * torch.exp(1j * (spec / self.b).angle())
        return spec

    def encode(self, x):  # raw latents => final latents
        self.init(x.device)
        x = torch.stft(x, self.n_fft, self.hop_length, self.win_length, window=self.window, return_complex=True)
        # Remove DC component
        x = x[:, 1:, :]
        x = self.amplitude_compression(x)
        x = rearrange(torch.view_as_real(x.squeeze(1)), "b f t c -> b c f t")
        x = x * const_like(x, self.scale).reshape(1, -1, 1, 1)

        return x

    def decode(self, x):  # mels => raw samples
        self.init(x.device)

        x = x / const_like(x, self.scale).reshape(1, -1, 1, 1)
        x = rearrange(x, "b c f t -> b f t c").contiguous()
        x = torch.view_as_complex(x)
        x = self.amplitude_decompression(x)
        # Add back DC component
        batch_size, freq_bins, time_frames = x.shape
        dc_component = torch.zeros((batch_size, 1, time_frames), device=x.device, dtype=x.dtype)
        x = torch.cat((dc_component, x), dim=1)
        x = torch.istft(x, self.n_fft, self.hop_length, self.win_length, window=self.window)
        return x
