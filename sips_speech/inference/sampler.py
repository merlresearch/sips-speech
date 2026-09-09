# Copyright (c) 2026 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Sampling functions for SIPS, including the Euler-Maruyama sampler and a random
generator
"""

import torch

# ---------------------------------------------------------------------------------------
# Euler-Maruyama sampler for SIPS, as described in Algorithm 1 of the paper.


def euler_sampler(
    model,
    rnd,
    y,
    v_hat,
    kappa=0.0,
    num_steps=15,
    t_0=0.0,
    t_1=1.0,
):
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}.")

    # Create uniform time steps
    time_steps = torch.linspace(t_0, t_1, num_steps + 1, device=y.device)
    dt = (t_1 - t_0) / num_steps

    # Initialize x_t with the observation y at time t=0.
    x_t = y

    # Iterate over all but the final time step; the final Euler update reaches t=1.
    for t in time_steps[:-1]:
        gamma_dot_t = model.gamma_dot(t, c=model.c)
        t_batch = t * torch.ones(x_t.shape[0], device=x_t.device)

        # Model forward pass and Euler update.
        z_hat = model(x_t, t_batch)
        x_t = x_t + (v_hat + (gamma_dot_t - kappa) * z_hat) * dt

        # Add noise according to the SDE discretization.
        if kappa > 0:
            z = rnd.randn_like(x_t)
            gamma_t = model.gamma(t_batch, c=model.c).reshape(-1, 1, 1, 1)
            scaled_noise = torch.sqrt(2 * dt * kappa * gamma_t) * z
            x_t = x_t + scaled_noise

    return x_t


# ---------------------------------------------------------------------------------------
# Helper class to generate random numbers with a fixed seed for reproducibility across
# distributed workers.


class RandomGenerator:
    def __init__(self, device, seed):
        self.generator = torch.Generator(device).manual_seed(int(seed) % (1 << 32))

    def randn_like(self, input):
        return torch.randn(
            input.shape,
            dtype=input.dtype,
            layout=input.layout,
            device=input.device,
            generator=self.generator,
        )
