# Copyright (C) 2024-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import sips_speech

REPO_ROOT = Path(sips_speech.__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "checkpoints" / "flexio_hparams.yaml"
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "flexio_avg.ckpt"


def unwrap_state_dict(checkpoint):
    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            checkpoint = checkpoint[key]
            break
    if isinstance(checkpoint, dict) and all(key.startswith("model.") for key in checkpoint):
        return {key.removeprefix("model."): value for key, value in checkpoint.items()}
    return checkpoint


def load_model(config_path, checkpoint_path, device, num_speakers=1):
    import torch

    from sips_speech.predictors.flexio.config_utils import yaml_to_parser
    from sips_speech.predictors.flexio.flexio_wrapper import SeparationModel

    hparams = yaml_to_parser(config_path)
    hparams = hparams.parse_args([])

    model = SeparationModel(
        hparams.encoder_name,
        hparams.encoder_conf,
        hparams.decoder_name,
        hparams.decoder_conf,
        hparams.model_name,
        hparams.model_conf,
    )

    checkpoint = unwrap_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.load_state_dict(checkpoint, strict=True)
    model.num_speakers = num_speakers
    model.to(device)
    model.eval()
    return model
