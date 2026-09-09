# Copyright (c) 2026 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

from sips_speech.predictors.standalone import main as run_standalone


def main():
    run_standalone("convtasnet")


if __name__ == "__main__":
    main()
