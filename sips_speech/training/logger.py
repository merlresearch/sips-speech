# Copyright (c) 2026 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Utilities for logging training output to a file."""

import sys


class Logger:
    """Mirror stdout and stderr to a file."""

    def __init__(self, file_name=None, file_mode="w", should_flush=True, encoding="utf-8"):
        self.file = None
        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        if file_name is not None:
            self.file = open(file_name, file_mode, encoding=encoding)

        sys.stdout = self
        sys.stderr = self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def write(self, text):
        if isinstance(text, bytes):
            text = text.decode("utf-8")

        # Workaround for VSCode debugger issue with empty writes.
        if not text:
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self):
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self):
        self.flush()

        if sys.stdout is self:
            sys.stdout = self.stdout

        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()
            self.file = None
