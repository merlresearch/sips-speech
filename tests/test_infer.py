# Copyright (c) 2026 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

import importlib
import sys

from tests.fakes import install_infer_import_stubs, install_torch_stub


class _Args:
    model = "model.pt"
    predictor = None
    num_speakers = 2
    in_dir = "in"
    out_dir = "out"
    proc_dir = "proc"
    seed = 0
    kappa = 0.0
    postprocess = False
    num_steps = 15


def test_infer_main_destroys_initialized_process_group(monkeypatch):
    install_torch_stub(monkeypatch)
    install_infer_import_stubs(monkeypatch)
    monkeypatch.delitem(sys.modules, "sips_speech.infer", raising=False)
    monkeypatch.delitem(sys.modules, "sips_speech.utils.distributed", raising=False)
    infer = importlib.import_module("sips_speech.infer")

    calls = []

    monkeypatch.setattr(infer, "parse_args", lambda: _Args())
    monkeypatch.setattr(infer, "validate_args", lambda args: None)
    monkeypatch.setattr(infer, "setup_distributed", lambda: 0)
    monkeypatch.setattr(infer.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(infer, "load_model", lambda model, device: object())
    monkeypatch.setattr(infer, "build_predictor", lambda *args, **kwargs: object())
    monkeypatch.setattr(infer, "get_inference_iterable", lambda **kwargs: ())
    monkeypatch.setattr(infer, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.setattr(infer, "is_rank0", lambda: True)
    monkeypatch.setattr(infer.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(infer.dist, "destroy_process_group", lambda: calls.append("destroyed"))

    infer.main()

    assert calls == ["destroyed"]
