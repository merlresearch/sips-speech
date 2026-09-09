# Copyright (c) 2026 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later


def test_set_random_seed_is_stable_across_python_processes():
    from sips_speech.utils.misc import _compute_seed

    assert _compute_seed("Hello World") == 1773455910
