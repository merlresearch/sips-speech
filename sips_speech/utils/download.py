# Copyright (c) 2026 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Utility functions for distributed processing and file handling in SIPS."""

import os
import subprocess
import tempfile
import urllib.request
from html.parser import HTMLParser
from urllib.parse import urlencode

import torch
import torch.distributed as dist

from .distributed import is_rank0


class GoogleDriveDownloadParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_form = False
        self.action = None
        self.data = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form" and attrs.get("id") == "download-form":
            self.in_form = True
            self.action = attrs.get("action")
        elif tag == "input" and self.in_form and "name" in attrs:
            self.data[attrs["name"]] = attrs.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self.in_form = False


def gdrive_download(file_id, out):
    c = tempfile.NamedTemporaryFile(delete=False)
    h = tempfile.NamedTemporaryFile(delete=False)
    c.close()
    h.close()

    try:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        subprocess.run(["curl", "-L", "-s", "-c", c.name, url, "-o", h.name], check=True)

        text = open(h.name, encoding="utf-8", errors="ignore").read()
        p = GoogleDriveDownloadParser()
        p.feed(text)

        if p.action:
            p.data["id"] = file_id
            url = p.action + "?" + urlencode(p.data)

        subprocess.run(["curl", "-L", "-b", c.name, url, "-o", out], check=True)
    finally:
        os.unlink(c.name)
        os.unlink(h.name)


def ensure_file(
    file_path: str,
    url: str = None,
    google_id: str = None,
    local_rank: int = None,
):
    # Download only on rank 0
    if is_rank0():
        if not os.path.isfile(file_path):
            os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
            if url:
                print(f"[Rank 0] Download {file_path} from {url}...")
                urllib.request.urlretrieve(url, file_path)
                print("[Rank 0] Download complete.")
            elif google_id:
                print(f"[Rank 0] Download {file_path} from Google Drive with ID {google_id}...")
                gdrive_download(google_id, file_path)
                print("[Rank 0] Download complete.")
            else:
                raise ValueError("Either url or google_id must be provided to download the file.")
    # Synchronize all processes
    if dist.is_initialized():
        if torch.cuda.is_available() and local_rank is not None:
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()
