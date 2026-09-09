# Copyright (c) 2026 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print("Usage: python -m sips_speech [train|infer]")
        sys.exit(0 if len(sys.argv) >= 2 else 1)

    cmd = sys.argv[1]
    sys.argv = [sys.argv[0], *sys.argv[2:]]

    if cmd == "train":
        from .train import main as train_main

        train_main()
    elif cmd == "infer":
        from .infer import main as infer_main

        infer_main()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
