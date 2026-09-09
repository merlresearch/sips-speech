# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2023 ESPnet Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import torch

from sips_speech.predictors.flexio.audio_utils import do_istft, do_stft
from sips_speech.predictors.flexio.net import MultiTussModel


class SeparationModel(torch.nn.Module):
    def __init__(
        self,
        encoder_name,
        encoder_conf,
        decoder_name,
        decoder_conf,
        separator_name,
        separator_conf,
        variance_normalization: bool = True,
    ):
        super().__init__()
        if separator_name != "mctuss":
            raise NotImplementedError("Only mctuss separator is supported")

        if encoder_name == "stft":
            self.encoder = partial(do_stft, **encoder_conf)
        else:
            raise NotImplementedError("Only stft encoder is supported")

        if decoder_name == "stft":
            self.decoder = partial(do_istft, **decoder_conf)
        else:
            raise NotImplementedError("Only stft decoder is supported")

        if separator_conf["ref_channel"] == 0:
            self.ref_channel = 0
        else:
            raise NotImplementedError("Only ref_channel == 0 is supported")

        self.separator = MultiTussModel(**separator_conf)
        self.variance_normalization = variance_normalization
        self.is_separation_predictor = True
        self.num_speakers = 1

    def forward(
        self,
        mix: torch.Tensor,
        prompts: list[list[str]],
        epsilon_for_normalization: float = 1e-8,
    ):
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)
        elif mix.dim() == 3:
            pass
        else:
            raise ValueError("Input mix must have 2 or 3 dimensions")

        with torch.amp.autocast("cuda", enabled=False):
            if self.variance_normalization:
                std = mix[:, self.ref_channel, :].std(dim=-1, keepdim=True) + epsilon_for_normalization
                mix = mix / std[:, None, :]
            X = self.encoder(mix)

        Y = self.separator(X, prompts)

        with torch.amp.autocast("cuda", enabled=False):
            y = self.decoder(Y)
            if self.variance_normalization:
                y = y * std.unsqueeze(1)
        return y

    def enhance(self, y):
        prompts = [["speech"] * self.num_speakers for _ in range(y.shape[0])]
        enhanced = self(y, prompts)

        if enhanced.shape[0] == 1:
            enhanced = enhanced.squeeze(0)

        return enhanced
