# Copyright (c) 2026, Mitsubishi Electric Research Laboratories. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Loss function for training SIPS as defined in Equation 9 of the paper."""

import torch
import torch.nn.functional as F

from sips_speech.utils.config import EasyDict


class DenoiserLoss:
    def __init__(self, model, eps=1e-8):
        self.eps = float(eps)

        # Access the underlying model in case it's wrapped in DDP.
        m = model.module if hasattr(model, "module") else model
        self.a = m.a
        self.c = m.c
        self.gamma = m.gamma
        self.t_eps = m.t_eps
        self.T = m.T

    def __call__(self, model, speech):

        # Sample random variables.
        t = torch.rand([speech.shape[0], 1, 1, 1], device=speech.device) * (self.T - self.t_eps) + self.t_eps
        gamma_t = self.gamma(t=t, c=self.c)
        z = torch.randn_like(speech)
        x_t = speech + (self.a + gamma_t) * z

        # Forward pass through the network.
        model_out, logvar = model(x=x_t, t=t, return_logvar=True)

        # Compute loss.
        loss_dict = EasyDict()
        unscaled_loss = F.mse_loss(model_out, z, reduction="none")
        unscaled_loss = unscaled_loss.flatten(1).mean(dim=1).reshape(-1, 1, 1, 1)

        if logvar is not None:
            eps = torch.as_tensor(self.eps, device=speech.device, dtype=unscaled_loss.dtype)
            loss = (1 / (logvar.exp() + eps)) * unscaled_loss + logvar
            loss_dict["unscaled_loss"] = unscaled_loss
            loss_dict["logvar"] = logvar
        else:
            loss = unscaled_loss
            loss_dict["unscaled_loss"] = unscaled_loss

        return loss, loss_dict
