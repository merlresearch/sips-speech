# Copyright (c) 2026 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Inference pipeline for SIPS."""

import os
from glob import glob

import soundfile as sf
import torch
import torch.distributed as dist

from sips_speech.inference.sampler import RandomGenerator, euler_sampler
from sips_speech.io import load_mono_wav


def get_inference_iterable(
    model,
    in_dir: str,
    out_dir: str | None = None,
    proc_dir: str | None = None,
    seed: int = 0,
    kappa: float = 0.0,
    predictor=None,
    postprocess: bool = False,
    num_steps: int = 15,
):
    """Return an iterable over inference results.

    The iterable processes a rank-specific shard of input WAV files, optionally uses
    precomputed predictor outputs from ``proc_dir`` or an online predictor, applies
    SIPS sampling, optionally postprocesses with the predictor, and saves enhanced
    audio to ``out_dir``.

    Args:
        model: SIPS model.
        in_dir: Directory containing noisy input WAV files.
        out_dir: Optional output directory for enhanced WAV files.
        proc_dir: Optional directory containing precomputed predictor WAV files.
        seed: Random seed for the sampler.
        kappa: Noise parameter for the sampler.
        predictor: Optional predictor model with an ``enhance`` method.
        postprocess: Whether to run the predictor after SIPS sampling.
        num_steps: Number of steps for the Euler-Maruyama sampler.

    Returns:
        Iterable yielding enhanced audio arrays.
    """
    in_paths = sorted(glob(os.path.join(in_dir, "**", "*.wav"), recursive=True))

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        indices = list(range(rank, len(in_paths), world_size))
    else:
        indices = list(range(len(in_paths)))

    proc_paths = None
    if proc_dir is not None:
        proc_paths = sorted(glob(os.path.join(proc_dir, "**", "*.wav"), recursive=True))
        assert len(proc_paths) == len(in_paths), (
            f"Number of processed files ({len(proc_paths)}) does not match "
            f"number of corrupted input files ({len(in_paths)})."
        )

    device = next(model.parameters()).device

    class InferenceIterable:
        def __len__(self):
            return len(indices)

        def __iter__(self):
            for i in indices:
                x_est = enhance_file(
                    model=model,
                    in_path=in_paths[i],
                    out_path=_output_path(in_paths[i], in_dir, out_dir),
                    proc_path=None if proc_paths is None else proc_paths[i],
                    predictor=predictor,
                    seed=seed,
                    kappa=kappa,
                    postprocess=postprocess,
                    device=device,
                    num_steps=num_steps,
                )

                yield x_est

    return InferenceIterable()


@torch.no_grad()
def enhance_file(
    model,
    in_path: str,
    out_path: str | None = None,
    proc_path: str | None = None,
    predictor=None,
    seed: int = 0,
    kappa: float = 0.0,
    postprocess: bool = False,
    device: torch.device | None = None,
    num_steps: int = 15,
):
    """Enhance a single WAV file with SIPS."""
    if device is None:
        device = next(model.parameters()).device

    y, sr_y = _load_mono_wav(in_path)

    assert sr_y == model.encoder.sr, f"Expected sampling rate {model.encoder.sr}, but got {sr_y} for file {in_path}"

    y_len = y.size(-1)
    normfac = y.abs().max().clamp_min(1e-8)
    y_norm = y / normfac

    x_proc_norm = None
    if proc_path is not None:
        x_proc, sr_proc = _load_mono_wav(proc_path)

        assert (
            sr_proc == model.encoder.sr
        ), f"Expected sampling rate {model.encoder.sr} for processed file, but got {sr_proc} for file {proc_path}"

        assert os.path.basename(in_path) == os.path.basename(proc_path), (
            f"Input file name {os.path.basename(in_path)} does not match processed "
            f"file name {os.path.basename(proc_path)}"
        )

        x_proc_norm = x_proc / normfac

    y_enc = model.encoder.encode(y_norm)
    y_enc, og_spec_length = model.pad_spec(y_enc)
    y_enc = y_enc.to(device)

    if x_proc_norm is not None:
        x_proc_enc = model.encoder.encode(x_proc_norm.to(device))
        x_proc_enc, _ = model.pad_spec(x_proc_enc)
        v_hat = x_proc_enc - y_enc
    else:
        assert predictor is not None, "Either proc_path or predictor must be provided."

        y_pad = model.encoder.decode(y_enc) * normfac
        x_hat = predictor.enhance(y_pad)
        x_hat = _as_batch_audio(x_hat) / normfac
        x_hat = _match_audio_length(x_hat, y_pad.shape[-1], max_length_mismatch=getattr(model.encoder, "n_fft", 0))
        x_hat_enc = model.encoder.encode(x_hat)
        y_enc = y_enc.expand(x_hat_enc.shape[0], -1, -1, -1)
        v_hat = x_hat_enc - y_enc

    rnd = RandomGenerator(device, seed)

    x_T_enc = euler_sampler(
        model=model,
        rnd=rnd,
        y=y_enc,
        v_hat=v_hat,
        kappa=kappa,
        num_steps=num_steps,
    )

    if postprocess:
        assert predictor is not None, "Predictor must be provided for postprocessing."

        assert not getattr(
            predictor, "is_separation_predictor", False
        ), "Predictor postprocessing is not supported for separation predictors."

        x_T = model.encoder.decode(x_T_enc[..., :og_spec_length]) * normfac
        x_T = predictor.enhance(x_T)
        x_est = _prepare_output_audio(x_T, predictor).cpu().numpy()[..., :y_len]
    else:
        x_T = model.encoder.decode(x_T_enc[..., :og_spec_length])
        x_est = _prepare_output_audio(x_T, predictor).cpu().numpy()[..., :y_len] * normfac.cpu().numpy()

    if out_path is not None:
        _write_enhanced(out_path, x_est, model.encoder.sr, predictor)

    return x_est


def _as_batch_audio(audio):
    """Return predictor audio as ``[batch_or_sources, time]``.

    Predictors may return a single waveform ``[time]``, an already batched or
    source-separated tensor ``[batch_or_sources, time]``, or a singleton batch
    ``[1, batch_or_sources, time]``. The singleton cases are normalized here so
    the encoder receives a 2-D audio tensor.
    """
    if audio.dim() == 1:
        return audio.unsqueeze(0)
    if audio.dim() == 3 and audio.shape[0] == 1:
        return audio.squeeze(0)
    return audio


def _match_audio_length(audio, target_length: int, max_length_mismatch: int = 0):
    """Match audio length when the mismatch is explained by STFT padding."""
    current_length = audio.shape[-1]
    length_mismatch = abs(current_length - target_length)
    if length_mismatch > max_length_mismatch:
        raise ValueError(
            f"Predictor output length {current_length} differs from expected length "
            f"{target_length} by {length_mismatch} samples, which exceeds the "
            f"allowed STFT padding mismatch of {max_length_mismatch} samples."
        )

    if current_length > target_length:
        return audio[..., :target_length]
    if current_length < target_length:
        return torch.nn.functional.pad(audio, (0, target_length - current_length))
    return audio


def _prepare_output_audio(audio, predictor):
    if getattr(predictor, "is_separation_predictor", False):
        if audio.dim() == 3 and audio.shape[0] == 1:
            return audio.squeeze(0)
        if audio.dim() == 1:
            return audio.unsqueeze(0)
        return audio

    while audio.dim() > 1 and audio.shape[0] == 1:
        audio = audio.squeeze(0)
    return audio


def _write_enhanced(out_path: str, audio, sr: int, predictor):
    if getattr(predictor, "is_separation_predictor", False):
        for speaker_index, speaker_audio in enumerate(audio):
            speaker_path = os.path.join(
                os.path.dirname(out_path),
                f"s{speaker_index + 1}",
                os.path.basename(out_path),
            )
            os.makedirs(os.path.dirname(speaker_path), exist_ok=True)
            sf.write(speaker_path, speaker_audio, sr)
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, audio, sr)


def _load_mono_wav(path: str):
    return load_mono_wav(path, warn_multichannel_once=True, stacklevel=3)


def _output_path(in_path: str, in_dir: str, out_dir: str | None):
    if out_dir is None:
        return None

    rel_path = os.path.relpath(in_path, in_dir)
    return os.path.join(out_dir, rel_path)
