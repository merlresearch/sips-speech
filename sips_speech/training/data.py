# Copyright (c) 2026 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Dataset and sampler for SIPS training."""

from glob import glob
from os.path import isdir, join

import numpy as np
import torch
from numpy.random import RandomState

from sips_speech.io import load_mono_wav


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,  # Path to the directory.
        sr=16000,  # Sample rate for audio.
        num_frames=256,  # Number of frames in the spectrogram.
        hop_length=128,  # Hop length for the spectrogram.
        normalize=True,  # Normalize audio to [-1, 1]?
        **super_kwargs,  # Additional arguments for the Dataset base class.
    ):
        self.path = path
        self.sr = sr
        self.num_frames = num_frames
        self.hop_length = hop_length
        self.normalize = normalize

        # Audio paths.
        if not isdir(path):
            raise FileNotFoundError(f"Dataset path does not exist: {path}")

        self.clean_paths = sorted(glob(join(path, "**", "*.wav"), recursive=True))
        if not self.clean_paths:
            raise ValueError(f"No .wav files found in dataset path: {path}")

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        # Load audio.
        x, sr_x = load_mono_wav(self.clean_paths[idx])

        # Check sample rate and shape.
        assert sr_x == self.sr, f"Sample rate mismatch in {self.clean_paths[idx]}: {sr_x} != {self.sr}"

        # Take only the first channel.
        x = x[0]

        # Normalize audio with respect to the maximum value in the clean audio.
        if self.normalize:
            normfac = x.abs().max().clamp_min(1e-8)
            x = x / normfac

        # Cut audio to fixed length.
        target_len = (self.num_frames - 1) * self.hop_length
        current_len = x.size(-1)
        if current_len == 0:
            raise ValueError(f"Audio file is empty: {self.clean_paths[idx]}")
        pad = max(target_len - current_len, 0)
        if pad == 0:
            # Extract random part of the audio file.
            start = int(np.random.uniform(0, current_len - target_len))
            x = x[start : start + target_len]
        else:
            # Repeat the signal to pad to the desired length
            repeat_times = (target_len // current_len) + 1
            x = x.repeat(repeat_times)[:target_len]

        item = {
            "clean_audio": x,
        }

        return item

    def collate_fn(self, batch):
        # List to hold the batched data for each item in the dictionary
        collated_batch = {}

        # Assuming each item in batch is a dictionary with the same keys
        if len(batch) == 0:
            return collated_batch

        first_item = batch[0]
        for key in first_item.keys():
            # Gather all elements under the key across the entire batch
            data_list = [item[key] for item in batch]

            # Stack or pad/stack if necessary
            if isinstance(data_list[0], torch.Tensor):
                collated_batch[key] = torch.stack(data_list)
            else:
                # Additional handling if the type is not a tensor
                if data_list[0] is not None:
                    collated_batch[key] = torch.tensor(np.array(data_list))
                else:
                    collated_batch[key] = None

        return collated_batch


class InfiniteSampler:
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, start_idx=0):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        try:
            super().__init__(dataset)  # older torch
        except TypeError:
            super().__init__()  # newer torch
        self.dataset_size = len(dataset)
        self.start_idx = start_idx + rank
        self.stride = num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        idx = self.start_idx
        epoch = None
        while True:
            if epoch != idx // self.dataset_size:
                epoch = idx // self.dataset_size
                order = np.arange(self.dataset_size)
                if self.shuffle:
                    RandomState(hash((self.seed, epoch)) % (1 << 31)).shuffle(order)
            yield int(order[idx % self.dataset_size])
            idx += self.stride
