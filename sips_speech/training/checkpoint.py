# Copyright (c) 2026 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Utilities for saving and loading training checkpoints."""

import re
from collections.abc import Mapping
from pathlib import Path

import torch

from sips_speech.utils.distributed import get_rank, print0


class Checkpoint:
    """Helper class for saving and loading checkpointable objects."""

    def __init__(self, **kwargs):
        self._state_objs = kwargs

    def register(self, **kwargs):
        self._state_objs.update(kwargs)

    def state_dict(self):
        return {name: self._get_state(name, obj) for name, obj in self._state_objs.items()}

    def save(self, pt_path):
        pt_path = Path(pt_path)

        if get_rank() == 0:
            pt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), pt_path)

    def load(self, pt_path, map_location="cpu", strict=True):
        data = self._load_checkpoint(pt_path, map_location)

        for name, obj in self._state_objs.items():
            if name not in data:
                if strict:
                    raise KeyError(f"Checkpoint does not contain state object '{name}'.")
                continue

            self._set_state(name, obj, data[name])

        print0("done")

        return dict(data)

    def load_state_dict(self, pt_path, map_location="cpu", strict=False):
        """Load only objects supporting load_state_dict."""
        data = self._load_checkpoint(pt_path, map_location)

        for name, obj in self._state_objs.items():
            if obj is None or not hasattr(obj, "load_state_dict"):
                continue

            if name not in data:
                print0(f"\nWarning: checkpoint does not contain '{name}', skipping.")
                continue

            try:
                out = obj.load_state_dict(data[name], strict=strict)
            except TypeError:
                out = obj.load_state_dict(data[name])

            missing_keys = getattr(out, "missing_keys", None)
            unexpected_keys = getattr(out, "unexpected_keys", None)

            if missing_keys is not None:
                print0(f"\nWarning: '{name}' missing keys: {len(missing_keys)}")

            if unexpected_keys is not None:
                print0(f"Warning: '{name}' unexpected keys: {len(unexpected_keys)}")

        return dict(data)

    def load_latest(
        self,
        run_dir,
        pattern=r"training-state-(\d+).pt",
        map_location="cpu",
        strict=True,
    ):
        pt_path = find_latest_checkpoint(run_dir, pattern)

        if pt_path is None:
            return None

        self.load(
            pt_path,
            map_location=map_location,
            strict=strict,
        )

        return pt_path

    def load_pretrained(
        self,
        run_dir,
        pattern=r"training-state-(\d+).pt",
        map_location="cpu",
        strict=False,
    ):
        pt_path = find_latest_checkpoint(run_dir, pattern)

        if pt_path is None:
            return None

        self.load_state_dict(
            pt_path,
            map_location=map_location,
            strict=strict,
        )

        return pt_path

    @staticmethod
    def _load_checkpoint(pt_path, map_location):
        pt_path = Path(pt_path)

        print0(f"Loading {pt_path} ... ", end="", flush=True)

        data = torch.load(pt_path, map_location=map_location, weights_only=False)

        if not isinstance(data, Mapping):
            raise TypeError(f"Checkpoint must contain a mapping, got {type(data).__name__}.")

        return data

    @staticmethod
    def _get_state(name, obj):
        if obj is None:
            return None

        if isinstance(obj, dict):
            return obj

        if hasattr(obj, "state_dict"):
            return obj.state_dict()

        if hasattr(obj, "__getstate__"):
            return obj.__getstate__()

        if hasattr(obj, "__dict__"):
            return obj.__dict__

        raise ValueError(f"Invalid state object '{name}' of type {type(obj).__name__}.")

    @staticmethod
    def _set_state(name, obj, state):
        if obj is None:
            return

        if isinstance(obj, dict):
            obj.clear()
            obj.update(state)
            return

        if hasattr(obj, "load_state_dict"):
            obj.load_state_dict(state)
            return

        if hasattr(obj, "__setstate__"):
            obj.__setstate__(state)
            return

        if hasattr(obj, "__dict__"):
            obj.__dict__.clear()
            obj.__dict__.update(state)
            return

        raise ValueError(f"Invalid state object '{name}' of type {type(obj).__name__}.")


def find_latest_checkpoint(run_dir, pattern=r"training-state-(\d+).pt"):
    run_dir = Path(run_dir)

    if not run_dir.exists():
        return None

    candidates = []

    for path in run_dir.iterdir():
        if not path.is_file():
            continue

        match = re.fullmatch(pattern, path.name)

        if match is None:
            continue

        if match.lastindex is None or match.lastindex < 1:
            raise ValueError("Checkpoint pattern must contain at least one capture group.")

        candidates.append((float(match.group(1)), path))

    if not candidates:
        return None

    return max(candidates, key=lambda item: item[0])[1]
