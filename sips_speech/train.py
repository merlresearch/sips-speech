# Copyright (c) 2026 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Command-line entry point for SIPS training."""

import json
import os
from argparse import ArgumentParser, BooleanOptionalAction

import torch
import torch.distributed as dist

from sips_speech.training.logger import Logger
from sips_speech.training.training_loop import training_loop
from sips_speech.utils.config import EasyDict
from sips_speech.utils.distributed import is_rank0, setup_distributed


def parse_nwav(s):
    if s is None or isinstance(s, int):
        return s

    s = str(s)
    if s.endswith("K"):
        return int(s[:-1]) << 10
    if s.endswith("M"):
        return int(s[:-1]) << 20
    if s.endswith("G"):
        return int(s[:-1]) << 30
    return int(s)


def parse_int_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        return [int(x) for x in value.split(",") if x]
    except ValueError as exc:
        raise ValueError("Expected a comma-separated list of integers.") from exc


def parse_args():
    parser = ArgumentParser(description="Train SIPS according to the default recipe.")

    # Main options.
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)

    # Model options.
    parser.add_argument("--logvar_channels", type=int, default=128)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--a", type=float, default=0.1)
    parser.add_argument("--c", type=float, default=0.5)
    parser.add_argument("--channel_mult", type=parse_int_list, default=[1, 1, 2, 2, 2, 2, 2])
    parser.add_argument("--attn_resolutions", type=parse_int_list, default=[16])

    # Hyperparameters.
    parser.add_argument("--duration", type=parse_nwav, default=parse_nwav("4096K"))
    parser.add_argument("--num_frames", type=parse_nwav, default=256)
    parser.add_argument("--batch", type=parse_nwav, default=16)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--ref_lr", type=float, default=2.5e-3)
    parser.add_argument("--ref_batches", type=float, default=3e4)
    parser.add_argument("--rampup_mwav", type=int, default=1)

    # Encoder options.
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--b", type=float, default=0.15)

    # Performance-related options.
    parser.add_argument("--batch_gpu", type=parse_nwav, default=0)
    parser.add_argument("--bench", action=BooleanOptionalAction, default=True)

    # I/O-related options.
    parser.add_argument("--snapshot", type=parse_nwav, default=parse_nwav("1024K"))
    parser.add_argument("--checkpoint", type=parse_nwav, default=parse_nwav("2048K"))
    parser.add_argument("--seed", type=int, default=0)

    # Validation-related options.
    parser.add_argument("--noval", action="store_true")
    parser.add_argument("--valid_dir", type=str, default=None)
    parser.add_argument("--valid_pred_dir", type=str, default=None)

    args = parser.parse_args()

    return args


def setup_training_config(args):
    c = EasyDict()

    c.dataset_kwargs = EasyDict(
        class_name="sips_speech.training.data.Dataset",
        path=args.data,
        num_frames=args.num_frames,
    )

    c.audio_encoder_kwargs = EasyDict(
        class_name="sips_speech.models.encoder.SpectrogramEncoder",
        p=args.p,
        b=args.b,
    )

    c.update(
        total_nwav=args.duration,
        batch_size=args.batch,
    )

    c.model_kwargs = EasyDict(
        class_name="sips_speech.models.model.SIPS",
        logvar_channels=args.logvar_channels,
        a=args.a,
        c=args.c,
        channel_mult=args.channel_mult,
        attn_resolutions=args.attn_resolutions,
    )

    c.loss_kwargs = EasyDict(
        class_name="sips_speech.training.loss.DenoiserLoss",
        eps=args.eps,
    )

    c.lr_kwargs = EasyDict(
        func_name="sips_speech.training.schedulers.learning_rate_schedule",
        ref_lr=args.ref_lr,
        ref_batches=args.ref_batches,
        rampup_mwav=args.rampup_mwav,
    )

    c.batch_gpu = args.batch_gpu or None
    c.cudnn_benchmark = args.bench

    c.snapshot_nwav = args.snapshot or None
    c.checkpoint_nwav = args.checkpoint or None
    c.seed = args.seed
    c.noval = args.noval
    c.valid_dir = args.valid_dir
    c.valid_pred_dir = args.valid_pred_dir

    return c


def launch_training(run_dir, c):
    if is_rank0():
        is_new_run = not os.path.isdir(run_dir)
        os.makedirs(os.path.join(run_dir, "snapshots"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

        if is_new_run:
            os.makedirs(run_dir, exist_ok=True)

        config_name = os.path.basename(os.path.normpath(run_dir)) + "_config.json"
        config_path = os.path.join(run_dir, config_name)
        if not os.path.isfile(config_path):
            with open(config_path, "wt", encoding="utf-8") as f:
                json.dump(c, f, indent=2)

    if dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()

    Logger(
        file_name=os.path.join(run_dir, "log.txt"),
        file_mode="a",
        should_flush=True,
    )

    training_loop(run_dir=run_dir, **c)


def main():
    args = parse_args()

    torch.multiprocessing.set_start_method("spawn", force=True)
    _ = setup_distributed()

    if is_rank0():
        print("Start distributed training...")

    c = setup_training_config(args)

    launch_training(run_dir=args.run_dir, c=c)


if __name__ == "__main__":
    main()
