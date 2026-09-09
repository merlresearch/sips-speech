# Copyright (c) 2026, Mitsubishi Electric Research Laboratories. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Learning rate scheduler for training SIPS."""

import numpy as np


def learning_rate_schedule(
    cur_nwav,
    batch_size,
    ref_lr=100e-4,
    ref_batches=70e3,
    rampup_mwav=10,
):
    """Inverse square root learning rate schedule with optional linear rampup."""

    lr = ref_lr

    if ref_batches > 0:
        lr /= np.sqrt(max(cur_nwav / (ref_batches * batch_size), 1))
    if rampup_mwav > 0:
        lr *= min(cur_nwav / (rampup_mwav * 1e6), 1)

    return lr
