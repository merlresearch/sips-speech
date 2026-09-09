# Copyright (c) 2026 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Checkpoint and configuration URLs for predictors."""

# --------------------------------------------------------------------------------------
# SEMamba (VoiceBank-DEMAND)

SEMAMBA_BASE_URL = "https://github.com/RoyChao19477/SEMamba/raw/750ff5d8f0e112557ca039c268cdcc51e267f342/"

SEMAMBA_CKPT_URL = f"{SEMAMBA_BASE_URL}ckpts/SEMamba_advanced.pth"
SEMAMBA_CONFIG_URL = f"{SEMAMBA_BASE_URL}recipes/SEMamba_advanced/SEMamba_advanced.yaml"

# --------------------------------------------------------------------------------------
# NCSN++ (VoiceBank-DEMAND)
#
# Source:
# https://drive.google.com/drive/folders/1ExFm97obaXTYFoBApWjbK_ypxTP-Cgdq
# Path:
# NCSN++M > VoiceBank/DEMAND > epoch=96-pesq=0.00.ckpt

NCSNPP_CKPT_FILE_ID = "1AptunZUsDoglP8ME-VkTulOksh1_78p9"
