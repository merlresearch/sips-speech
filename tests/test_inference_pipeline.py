# Copyright (c) 2026 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

import importlib
import sys
import types
import warnings

from tests.fakes import install_pipeline_import_stubs, install_torch_stub


class _Model:
    def __init__(self, device):
        self.device = device

    def parameters(self):
        yield types.SimpleNamespace(device=self.device)


def test_distributed_inference_iterable_handles_uneven_shards_without_barrier(monkeypatch):
    torch, _ = install_torch_stub(monkeypatch)
    install_pipeline_import_stubs(monkeypatch)
    monkeypatch.delitem(sys.modules, "sips_speech.inference.pipeline", raising=False)
    pipeline = importlib.import_module("sips_speech.inference.pipeline")

    in_paths = [f"in/file_{index}.wav" for index in range(5)]
    calls_by_rank = []

    monkeypatch.setattr(pipeline, "glob", lambda *args, **kwargs: in_paths)
    monkeypatch.setattr(pipeline.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(pipeline.dist, "get_world_size", lambda: 3)
    monkeypatch.setattr(
        pipeline.dist,
        "barrier",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("barrier should not be called")),
    )

    def enhance_file(**kwargs):
        current_calls.append(kwargs["in_path"])
        return kwargs["in_path"]

    monkeypatch.setattr(pipeline, "enhance_file", enhance_file)

    for rank in range(3):
        current_calls = []
        monkeypatch.setattr(pipeline.dist, "get_rank", lambda rank=rank: rank)

        inference_iterable = pipeline.get_inference_iterable(
            model=_Model(torch.device("cpu")),
            in_dir="in",
            predictor=object(),
        )

        assert list(inference_iterable) == in_paths[rank::3]
        assert len(inference_iterable) == len(in_paths[rank::3])
        calls_by_rank.append(current_calls)

    assert calls_by_rank == [in_paths[0::3], in_paths[1::3], in_paths[2::3]]


def test_load_mono_wav_warns_once_for_multichannel_input(monkeypatch):
    torch, _ = install_torch_stub(monkeypatch)
    install_pipeline_import_stubs(monkeypatch)
    monkeypatch.delitem(sys.modules, "sips_speech.io", raising=False)
    monkeypatch.delitem(sys.modules, "sips_speech.inference.pipeline", raising=False)
    pipeline = importlib.import_module("sips_speech.inference.pipeline")
    audio_io = importlib.import_module("sips_speech.io")

    class FakeAudio:
        shape = (100, 2)

        def __getitem__(self, key):
            return self

        @property
        def T(self):
            return self

        def copy(self):
            return self

    class FakeTensor:
        pass

    fake_tensor = FakeTensor()

    monkeypatch.setattr(audio_io.sf, "read", lambda *args, **kwargs: (FakeAudio(), 16000), raising=False)
    monkeypatch.setattr(torch, "from_numpy", lambda audio: fake_tensor)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        audio, sr = pipeline._load_mono_wav("stereo.wav")

    assert len(caught) == 1
    assert "only the first channel" in str(caught[0].message)
    assert audio is fake_tensor
    assert sr == 16000

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline._load_mono_wav("stereo.wav")

    assert caught == []


def test_match_audio_length_rejects_unexplained_mismatch(monkeypatch):
    install_torch_stub(monkeypatch)
    install_pipeline_import_stubs(monkeypatch)
    monkeypatch.delitem(sys.modules, "sips_speech.inference.pipeline", raising=False)
    pipeline = importlib.import_module("sips_speech.inference.pipeline")

    class FakeTensor:
        shape = (1, 16000)

    try:
        pipeline._match_audio_length(FakeTensor(), target_length=12000, max_length_mismatch=512)
    except ValueError as error:
        assert "exceeds the allowed STFT padding mismatch" in str(error)
    else:
        raise AssertionError("Expected ValueError for unexplained predictor length mismatch")


def test_prepare_output_audio_preserves_single_sample_time_dimension(monkeypatch):
    install_torch_stub(monkeypatch)
    install_pipeline_import_stubs(monkeypatch)
    monkeypatch.delitem(sys.modules, "sips_speech.inference.pipeline", raising=False)
    pipeline = importlib.import_module("sips_speech.inference.pipeline")

    class FakeTensor:
        def __init__(self, shape):
            self.shape = shape

        def dim(self):
            return len(self.shape)

        def squeeze(self, dim):
            return FakeTensor(self.shape[:dim] + self.shape[dim + 1 :])

    output = pipeline._prepare_output_audio(FakeTensor((1, 1)), predictor=None)

    assert output.shape == (1,)


def test_prepare_output_audio_removes_singleton_leading_dimensions(monkeypatch):
    install_torch_stub(monkeypatch)
    install_pipeline_import_stubs(monkeypatch)
    monkeypatch.delitem(sys.modules, "sips_speech.inference.pipeline", raising=False)
    pipeline = importlib.import_module("sips_speech.inference.pipeline")

    class FakeTensor:
        def __init__(self, shape):
            self.shape = shape

        def dim(self):
            return len(self.shape)

        def squeeze(self, dim):
            return FakeTensor(self.shape[:dim] + self.shape[dim + 1 :])

    output = pipeline._prepare_output_audio(FakeTensor((1, 1, 16000)), predictor=None)

    assert output.shape == (16000,)


def test_prepare_output_audio_preserves_separation_source_dimension(monkeypatch):
    install_torch_stub(monkeypatch)
    install_pipeline_import_stubs(monkeypatch)
    monkeypatch.delitem(sys.modules, "sips_speech.inference.pipeline", raising=False)
    pipeline = importlib.import_module("sips_speech.inference.pipeline")

    class FakeTensor:
        def __init__(self, shape):
            self.shape = shape

        def dim(self):
            return len(self.shape)

        def squeeze(self, dim):
            return FakeTensor(self.shape[:dim] + self.shape[dim + 1 :])

    predictor = types.SimpleNamespace(is_separation_predictor=True)

    output = pipeline._prepare_output_audio(FakeTensor((1, 1, 16000)), predictor=predictor)

    assert output.shape == (1, 16000)


def test_postprocess_uses_predictor_waveform_output_directly(monkeypatch):
    torch, _ = install_torch_stub(monkeypatch)
    install_pipeline_import_stubs(monkeypatch)
    monkeypatch.delitem(sys.modules, "sips_speech.inference.pipeline", raising=False)
    pipeline = importlib.import_module("sips_speech.inference.pipeline")

    encode_inputs = []
    decode_inputs = []

    class FakeArray:
        def __init__(self, values):
            self.values = values

        def __getitem__(self, index):
            if isinstance(index, tuple):
                index = index[-1]
            return FakeArray(self.values[index])

        def tolist(self):
            return self.values

    class FakeScalar:
        def __init__(self, value):
            self.value = value

        def clamp_min(self, minimum):
            return FakeScalar(max(self.value, minimum))

    class FakeTensor:
        def __init__(self, values, shape=None):
            self.values = values
            self.shape = shape or (1, len(values))
            self.device = torch.device("cpu")

        def size(self, dim):
            return self.shape[dim]

        def abs(self):
            return FakeTensor([abs(value) for value in self.values], self.shape)

        def max(self):
            return FakeScalar(max(self.values))

        def to(self, device):
            return self

        def dim(self):
            return len(self.shape)

        def squeeze(self, dim=None):
            if dim is None:
                return self
            return FakeTensor(self.values, self.shape[:dim] + self.shape[dim + 1 :])

        def cpu(self):
            return self

        def numpy(self):
            return FakeArray(self.values)

        def __truediv__(self, scalar):
            return FakeTensor([value / scalar.value for value in self.values], self.shape)

        def __mul__(self, scalar):
            return FakeTensor([value * scalar.value for value in self.values], self.shape)

        def __sub__(self, other):
            return FakeTensor(self.values, self.shape)

        def __getitem__(self, index):
            return self

    class Encoder:
        sr = 16000

        def encode(self, audio):
            encode_inputs.append(audio)
            return FakeTensor([float(len(encode_inputs))], (1, 2, 3, 4))

        def decode(self, spec):
            decode_inputs.append(spec)
            return FakeTensor([1.0] * 6)

    class Model:
        encoder = Encoder()

        channel_mult = [1]

        def parameters(self):
            yield types.SimpleNamespace(device=torch.device("cpu"))

        def pad_spec(self, spec):
            return spec, spec.shape[-1]

    class Predictor:
        def enhance(self, audio):
            assert audio.values == [2.0] * 6
            return FakeTensor([5.0] * 6)

    def load_mono_wav(path):
        if path == "in.wav":
            return FakeTensor([0.5, -2.0, 1.0, 0.0, 0.25, -0.5]), 16000
        return FakeTensor([0.25, -1.0, 0.5, 0.0, 0.125, -0.25]), 16000

    monkeypatch.setattr(pipeline, "_load_mono_wav", load_mono_wav)
    monkeypatch.setattr(pipeline, "RandomGenerator", lambda *args, **kwargs: object())
    monkeypatch.setattr(pipeline, "euler_sampler", lambda **kwargs: FakeTensor([3.0], (1, 2, 3, 4)))

    x_est = pipeline.enhance_file(
        model=Model(),
        in_path="in.wav",
        proc_path="proc/in.wav",
        predictor=Predictor(),
        postprocess=True,
    )

    assert len(encode_inputs) == 2
    assert len(decode_inputs) == 1
    assert x_est.tolist() == [5.0] * 6
