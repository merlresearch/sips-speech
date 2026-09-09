# Copyright (c) 2026 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

import sys
import types


class FakeDevice:
    def __init__(self, spec="cpu"):
        spec = str(spec)
        device_type, _, index = spec.partition(":")
        self.type = device_type
        self.index = int(index) if index else None


def install_torch_stub(monkeypatch):
    torch = types.ModuleType("torch")
    dist = types.ModuleType("torch.distributed")

    dist.is_initialized = _false
    dist.get_rank = _zero
    dist.get_world_size = _one
    dist.barrier = _noop
    dist.destroy_process_group = _noop

    torch.distributed = dist
    torch.device = FakeDevice
    torch.no_grad = _no_grad
    torch.from_numpy = _identity
    torch.cuda = types.SimpleNamespace(
        is_available=_false,
        set_device=_noop,
        current_device=_zero,
    )
    torch.nn = types.SimpleNamespace(
        functional=types.SimpleNamespace(pad=_pad),
    )

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    return torch, dist


def install_pipeline_import_stubs(monkeypatch):
    soundfile = types.ModuleType("soundfile")
    sampler = types.ModuleType("sips_speech.inference.sampler")

    sampler.RandomGenerator = object
    sampler.euler_sampler = _noop

    monkeypatch.setitem(sys.modules, "soundfile", soundfile)
    monkeypatch.setitem(sys.modules, "sips_speech.inference.sampler", sampler)


def install_infer_import_stubs(monkeypatch):
    tqdm = types.ModuleType("tqdm")
    pipeline = types.ModuleType("sips_speech.inference.pipeline")
    model = types.ModuleType("sips_speech.models.model")
    registry = types.ModuleType("sips_speech.predictors.registry")

    tqdm.tqdm = _tqdm
    pipeline.get_inference_iterable = _empty_iterable
    model.SIPS = object
    registry.build_predictor = _object

    monkeypatch.setitem(sys.modules, "tqdm", tqdm)
    monkeypatch.setitem(sys.modules, "sips_speech.inference.pipeline", pipeline)
    monkeypatch.setitem(sys.modules, "sips_speech.models.model", model)
    monkeypatch.setitem(sys.modules, "sips_speech.predictors.registry", registry)


def _no_grad(func=None):
    if func is None:
        return lambda wrapped: wrapped
    return func


def _false():
    return False


def _zero():
    return 0


def _one():
    return 1


def _noop(*args, **kwargs):
    return None


def _pad(audio, padding):
    return audio


def _identity(value):
    return value


def _tqdm(iterable, **kwargs):
    return iterable


def _empty_iterable(**kwargs):
    return ()


def _object(*args, **kwargs):
    return object()
