# Copyright (c) 2026 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Miscellaneous utility functions for SIPS training and evaluation."""

import hashlib


def _compute_seed(*args) -> int:
    return int(hashlib.md5(repr(args).encode("utf-8"), usedforsecurity=False).hexdigest(), 16) % (1 << 31)


def set_random_seed(*args) -> int:
    import numpy as np
    import torch

    seed = _compute_seed(*args)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


def make_pbar(state, snapshot_nwav):
    from tqdm import tqdm

    from sips_speech.utils.distributed import get_rank

    bounded = snapshot_nwav is not None and snapshot_nwav > 0
    return tqdm(
        total=snapshot_nwav if bounded else None,
        initial=state.cur_nwav % snapshot_nwav if bounded else 0,
        unit="wavs",
        unit_scale=True,
        dynamic_ncols=True,
        disable=(get_rank() != 0),
    )
