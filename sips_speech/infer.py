# Copyright (c) 2026 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Command-line entry point for SIPS inference."""

import json
import os
from argparse import ArgumentParser, BooleanOptionalAction
from glob import glob

import torch
import torch.distributed as dist
from tqdm import tqdm

from sips_speech.inference.pipeline import get_inference_iterable
from sips_speech.models.model import SIPS
from sips_speech.predictors.registry import build_predictor
from sips_speech.utils.distributed import is_rank0, setup_distributed


def parse_args():
    parser = ArgumentParser(description="Run inference with SIPS to generate enhanced speech files.")
    parser.add_argument(
        "--in_dir",
        type=str,
        required=True,
        help="Path to noisy speech directory.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for enhanced files.",
    )
    parser.add_argument(
        "--proc_dir",
        type=str,
        default=None,
        help="Directory of preprocessed files from predictor models.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/SIPS.pt",
        help="Path to SIPS checkpoint.",
    )
    parser.add_argument(
        "--predictor",
        type=str,
        default=None,
        choices=["semamba", "convtasnet", "ncsnpp", "flexio"],
        help="Predictor model to use.",
    )
    parser.add_argument(
        "--num_speakers",
        type=int,
        default=2,
        help="Number of speakers for separation predictors such as FlexIO.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=15,
        help="Number of sampling steps.",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=0.0,
        help="Kappa parameter for the sampler.",
    )
    parser.add_argument(
        "--postprocess",
        action=BooleanOptionalAction,
        default=False,
        help="Enable predictor postprocessing.",
    )
    return parser.parse_args()


def validate_args(args):
    assert os.path.isfile(args.model), f"Checkpoint not found: {args.model}"

    assert os.path.isdir(args.in_dir), f"Input directory not found: {args.in_dir}"
    in_files = glob(os.path.join(args.in_dir, "**", "*.wav"), recursive=True)
    assert len(in_files) > 0, f"No .wav files found in input directory: {args.in_dir}"

    if args.proc_dir is not None:
        assert os.path.isdir(args.proc_dir), f"Processed directory not found: {args.proc_dir}"

    assert (
        args.proc_dir is not None or args.predictor is not None
    ), "Either --proc_dir or --predictor must be specified."

    if args.postprocess:
        assert args.predictor is not None, "Predictor must be specified if --postprocess is enabled."

    if args.proc_dir is not None and args.predictor is not None:
        assert args.postprocess, "If both --proc_dir and --predictor are specified, " "--postprocess must be enabled."


def load_model(checkpoint_path, device):
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    run_dir = os.path.dirname(checkpoint_dir)
    run_name = os.path.basename(os.path.normpath(run_dir))
    config_candidates = [
        os.path.join(checkpoint_dir, f"{checkpoint_name}_config.json"),
        os.path.join(run_dir, f"{checkpoint_name}_config.json"),
        os.path.join(run_dir, f"{run_name}_config.json"),
    ]
    config_path = next((path for path in config_candidates if os.path.isfile(path)), None)

    assert config_path is not None, "Model config not found. Tried: " + ", ".join(config_candidates)

    with open(config_path, "rt") as f:
        config = json.load(f)

    model_kwargs = config["model_kwargs"].copy()
    model_kwargs.pop("class_name", None)

    model = SIPS(**model_kwargs).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if "ema" in state_dict:
        state_dict = state_dict["ema"]

    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    args = parse_args()
    validate_args(args)

    local_rank = setup_distributed()
    try:
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        model = load_model(args.model, device)
        predictor_kwargs = {}
        if args.predictor == "flexio":
            predictor_kwargs["num_speakers"] = args.num_speakers

        predictor = build_predictor(args.predictor, local_rank, device, **predictor_kwargs)

        inference_iterable = get_inference_iterable(
            model=model,
            in_dir=args.in_dir,
            out_dir=args.out_dir,
            proc_dir=args.proc_dir,
            seed=args.seed,
            kappa=args.kappa,
            predictor=predictor,
            postprocess=args.postprocess,
            num_steps=args.num_steps,
        )

        for _ in tqdm(
            inference_iterable,
            desc="Inference",
            total=len(inference_iterable),
            disable=not is_rank0(),
        ):
            pass
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
