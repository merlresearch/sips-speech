# Copyright (c) 2026 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model class for SIPS (Stochastic Interpolant Prior for Speech)
"""

import torch

from sips_speech.models.encoder import SpectrogramEncoder
from sips_speech.third_party.edm2 import MPConv, MPFourier, UNet

# ---------------------------------------------------------------------------------------
# Noise schedule functions.


def gamma(t, c=0.5):
    """
    Noise schedule parameter gamma(t) as defined in the paper.
    """
    return c * torch.sin(t * torch.pi) ** 2


def gamma_dot(t, c=0.5):
    """
    Derivative of the noise parameter gamma(t) as defined in the paper.
    """
    return c * torch.pi * torch.sin(2 * torch.pi * t)


# ---------------------------------------------------------------------------------------
# SIPS model definition.


class SIPS(torch.nn.Module):
    """
    Stochastic Interpolant Prior for Speech model.

    Args:
        freq_resolution: Number of frequency bins expected by the U-Net backbone.
        spec_channels: Number of spectrogram channels, typically real and imaginary parts.
        model_channels: Base channel count for the U-Net backbone.
        channel_mult: Channel multipliers for each U-Net resolution level. The number of
            entries also determines the temporal padding multiple used by ``pad_spec``.
        num_blocks: Number of residual blocks per U-Net resolution level.
        attn_resolutions: Spectrogram resolutions where self-attention is enabled.
        t_eps: Lower bound for sampled interpolation times during training.
        T: Upper bound for sampled interpolation times during training.
        logvar_channels: Fourier feature width for learned loss uncertainty. Set to 0
            to disable log-variance prediction.
        a: Base noise scale added to the stochastic interpolant path.
        c: Amplitude of the sinusoidal noise schedule ``gamma(t)``.
        **unet_kwargs: Additional keyword arguments forwarded to the U-Net backbone.
    """

    def __init__(
        self,
        freq_resolution=256,
        spec_channels=2,
        model_channels=128,
        channel_mult=[1, 1, 2, 2, 2, 2, 2],
        num_blocks=2,
        attn_resolutions=[16],
        t_eps=0.0,
        T=1.0,
        logvar_channels=128,
        a=0.1,
        c=0.5,
        **unet_kwargs,
    ):
        super().__init__()
        # Store hyperparameters.
        self.freq_resolution = freq_resolution
        self.spec_channels = spec_channels
        self.channel_mult = channel_mult
        self.t_eps = t_eps
        self.T = T
        self.logvar_channels = logvar_channels
        self.a = a
        self.c = c
        self.gamma = gamma
        self.gamma_dot = gamma_dot

        # Encoder.
        self.encoder = SpectrogramEncoder()

        # U-Net backbone.
        self.unet = UNet(
            img_resolution=freq_resolution,
            img_channels=spec_channels,
            label_dim=0,
            model_channels=model_channels,
            channel_mult=channel_mult,
            num_blocks=num_blocks,
            attn_resolutions=attn_resolutions,
            **unet_kwargs,
        )

        # Uncertainty estimation components.
        if logvar_channels > 0:
            self.logvar_fourier = MPFourier(logvar_channels)
            self.logvar_linear = MPConv(logvar_channels, 1, kernel=[])
        else:
            self.logvar_fourier = None
            self.logvar_linear = None

    def forward(self, x, t, return_logvar=False):
        # Calculate noise standard deviation.
        t = t.to(torch.float32).reshape(-1, 1, 1, 1)

        # Input preconditioning.
        c_noise = t.flatten()

        # Run the model.
        F_x = self.unet(x=x, noise_labels=c_noise, class_labels=None)

        # Estimate uncertainty if requested.
        if return_logvar:
            logvar = (
                self.logvar_linear(self.logvar_fourier(c_noise)).reshape(-1, 1, 1, 1)
                if self.logvar_linear is not None
                else None
            )
            return F_x, logvar

        return F_x

    def pad_spec(self, spec):
        # Pad the spec length to be divisible by 2**len(model.channel_mult).
        og_spec_length = spec.shape[-1]
        padding_multiple = 2 ** len(self.channel_mult)
        pad = padding_multiple - og_spec_length % padding_multiple if og_spec_length % padding_multiple != 0 else 0
        spec = torch.nn.ReflectionPad2d((0, pad, 0, 0))(spec)
        return spec, og_spec_length
