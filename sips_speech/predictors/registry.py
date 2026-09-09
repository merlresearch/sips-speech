# Copyright (c) 2026 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Predictor registry and builder function."""

from pathlib import Path

import torch
from torch.serialization import add_safe_globals

import sips_speech
from sips_speech.utils.config import read_config
from sips_speech.utils.download import ensure_file

from .checkpoints import NCSNPP_CKPT_FILE_ID, SEMAMBA_CKPT_URL, SEMAMBA_CONFIG_URL

REPO_ROOT = Path(sips_speech.__file__).resolve().parent.parent
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"


class SpecsDataModule:
    pass


SpecsDataModule.__module__ = "sgmse.data_module"


def build_predictor(predictor_name, local_rank, device, **kwargs):
    """Build and initialize a predictor model."""

    if predictor_name is None:
        return None

    predictor_name = predictor_name.lower()

    if predictor_name == "semamba":
        from sips_speech.third_party.semamba import SEMamba

        predictor_path = CHECKPOINTS_DIR / "SEMamba_advanced.pth"
        config_path = CHECKPOINTS_DIR / "SEMamba_advanced.yaml"

        ensure_file(predictor_path, SEMAMBA_CKPT_URL, None, local_rank)
        ensure_file(config_path, SEMAMBA_CONFIG_URL, None, local_rank)

        predictor_cfg = read_config(str(config_path))
        predictor = SEMamba(predictor_cfg)

        checkpoint = torch.load(predictor_path, map_location=device)
        predictor.load_state_dict(checkpoint["generator"])

        predictor.to(device)
        predictor.eval()
        return predictor

    if predictor_name == "ncsnpp":
        from sips_speech.third_party.ncsnpp import NCSNppPredictor

        predictor_path = CHECKPOINTS_DIR / "NCSNpp_VB-DMD.ckpt"

        ensure_file(predictor_path, None, NCSNPP_CKPT_FILE_ID, local_rank)

        add_safe_globals([SpecsDataModule])
        checkpoint = torch.load(
            predictor_path,
            map_location="cpu",
            weights_only=True,
        )

        predictor = NCSNppPredictor()
        predictor.load_state_dict(checkpoint["state_dict"])
        predictor.ema.load_state_dict(checkpoint["ema"])
        predictor.to(device)
        predictor.eval(no_ema=False)
        return predictor

    if predictor_name == "convtasnet":
        from sips_speech.third_party.convtasnet import ConvTasNet

        predictor_path = CHECKPOINTS_DIR / "Conv-TasNet.pt"

        predictor = ConvTasNet.load_model(predictor_path)
        predictor.to(device)
        predictor.eval()
        return predictor

    if predictor_name == "flexio":
        from sips_speech.predictors.flexio.loader import DEFAULT_CHECKPOINT_PATH, DEFAULT_CONFIG_PATH, load_model

        predictor = load_model(
            kwargs.get("config_path", DEFAULT_CONFIG_PATH),
            kwargs.get("checkpoint_path", DEFAULT_CHECKPOINT_PATH),
            device,
            num_speakers=kwargs.get("num_speakers", 1),
        )
        return predictor

    supported = ["semamba", "ncsnpp", "convtasnet", "flexio"]
    raise ValueError(f"Unsupported predictor: {predictor_name}. " f"Supported predictors are: {', '.join(supported)}")
