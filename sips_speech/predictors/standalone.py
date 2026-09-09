# Copyright (c) 2026 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Standalone directory inference helpers for predictor models."""

import argparse
from pathlib import Path


def parse_args(predictor_name: str):
    parser = argparse.ArgumentParser(description=f"Run {predictor_name} inference.")
    parser.add_argument("--in_dir", required=True, type=Path, help="Input WAV directory.")
    parser.add_argument("--out_dir", required=True, type=Path, help="Output WAV directory.")
    if predictor_name == "flexio":
        from sips_speech.predictors.flexio.loader import DEFAULT_CHECKPOINT_PATH, DEFAULT_CONFIG_PATH

        parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Config path.")
        parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH, help="Checkpoint path.")
        parser.add_argument("--num_speakers", type=int, default=2, help="Number of speakers in the input mixture.")
    return parser.parse_args()


def main(predictor_name: str):
    args = parse_args(predictor_name)

    import torch
    import torch.distributed as dist
    from tqdm import tqdm

    from sips_speech.predictors.registry import build_predictor
    from sips_speech.utils.distributed import is_rank0, setup_distributed

    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    predictor = build_predictor(predictor_name, local_rank, device, **_predictor_kwargs(args, predictor_name))

    in_paths = sorted(args.in_dir.rglob("*.wav"))
    indices = _rank_indices(len(in_paths))

    try:
        for index in tqdm(indices, desc=f"{predictor_name} inference", disable=not is_rank0()):
            enhance_file(predictor, in_paths[index], args.in_dir, args.out_dir, device)

        if dist.is_initialized():
            dist.barrier()
    except KeyboardInterrupt:
        if dist.is_initialized():
            dist.destroy_process_group()
        raise SystemExit(130)


def enhance_file(predictor, in_path: Path, in_dir: Path, out_dir: Path, device):
    import soundfile as sf
    import torch

    audio, sr = sf.read(in_path, dtype="float32", always_2d=True)
    expected_sr = _predictor_sample_rate(predictor)
    if sr != expected_sr:
        raise ValueError(f"Expected sample rate is {expected_sr}. {sr} is not supported.")

    if getattr(predictor, "is_separation_predictor", False):
        mixture = torch.from_numpy(audio.T).unsqueeze(0).to(device)
    else:
        mixture = torch.from_numpy(audio.T)[0].unsqueeze(0).to(device)

    with torch.no_grad():
        enhanced = predictor.enhance(mixture).detach().cpu()

    enhanced = _prepare_output_audio(enhanced, predictor).numpy()
    enhanced = enhanced[..., : mixture.shape[-1]]
    _write_enhanced(sf, enhanced, in_path, in_dir, out_dir, sr, predictor)


def _write_enhanced(sf, enhanced, in_path: Path, in_dir: Path, out_dir: Path, sr: int, predictor):
    relative_path = in_path.relative_to(in_dir)

    if getattr(predictor, "is_separation_predictor", False):
        for speaker_index, speaker_audio in enumerate(enhanced):
            out_path = out_dir / relative_path.parent / f"s{speaker_index + 1}" / relative_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out_path, speaker_audio, sr)
        return

    out_path = out_dir / relative_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, enhanced, sr)


def _prepare_output_audio(enhanced, predictor):
    if getattr(predictor, "is_separation_predictor", False):
        if enhanced.dim() == 3 and enhanced.shape[0] == 1:
            return enhanced.squeeze(0)
        if enhanced.dim() == 1:
            return enhanced.unsqueeze(0)
        return enhanced

    return enhanced.squeeze()


def _predictor_kwargs(args, predictor_name: str):
    if predictor_name != "flexio":
        return {}

    return {
        "config_path": args.config,
        "checkpoint_path": args.checkpoint,
        "num_speakers": args.num_speakers,
    }


def _predictor_sample_rate(predictor) -> int:
    model = getattr(predictor, "model", None)
    encoder = getattr(model, "encoder", None)
    if encoder is not None and hasattr(encoder, "sr"):
        return encoder.sr

    if hasattr(predictor, "sample_rate"):
        return predictor.sample_rate

    if hasattr(predictor, "sr"):
        return predictor.sr

    return 16000


def _rank_indices(num_paths: int):
    import torch.distributed as dist

    if not dist.is_initialized():
        return range(num_paths)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return range(rank, num_paths, world_size)
