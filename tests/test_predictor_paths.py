# Copyright (c) 2026 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import sips_speech
from tests.fakes import install_torch_stub


def test_flexio_default_paths_are_package_root_relative(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    from sips_speech.predictors.flexio import loader

    repo_root = Path(sips_speech.__file__).resolve().parent.parent
    assert loader.DEFAULT_CONFIG_PATH == repo_root / "checkpoints" / "flexio_hparams.yaml"
    assert loader.DEFAULT_CHECKPOINT_PATH == repo_root / "checkpoints" / "flexio_avg.ckpt"


def test_registry_checkpoint_dir_is_package_root_relative(monkeypatch, tmp_path):
    install_torch_stub(monkeypatch)
    serialization = types.ModuleType("torch.serialization")
    yaml = types.ModuleType("yaml")
    serialization.add_safe_globals = lambda *args, **kwargs: None
    yaml.safe_load = lambda *args, **kwargs: {}
    sys.modules["torch"].serialization = serialization
    monkeypatch.setitem(sys.modules, "torch.serialization", serialization)
    monkeypatch.setitem(sys.modules, "yaml", yaml)
    monkeypatch.delitem(sys.modules, "sips_speech.predictors.registry", raising=False)
    monkeypatch.chdir(tmp_path)

    registry = importlib.import_module("sips_speech.predictors.registry")

    repo_root = Path(sips_speech.__file__).resolve().parent.parent
    assert registry.CHECKPOINTS_DIR == repo_root / "checkpoints"


def test_standalone_predictor_sample_rate_prefers_model_encoder_sr():
    from sips_speech.predictors import standalone

    predictor = SimpleNamespace(model=SimpleNamespace(encoder=SimpleNamespace(sr=22050)), sample_rate=16000, sr=8000)

    assert standalone._predictor_sample_rate(predictor) == 22050


def test_standalone_predictor_sample_rate_falls_back_to_predictor_attrs():
    from sips_speech.predictors import standalone

    assert standalone._predictor_sample_rate(SimpleNamespace(sample_rate=22050)) == 22050
    assert standalone._predictor_sample_rate(SimpleNamespace(sr=8000)) == 8000
    assert standalone._predictor_sample_rate(SimpleNamespace()) == 16000
