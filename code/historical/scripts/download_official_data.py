#!/usr/bin/env python3
"""Download the official THUML Time-Series-Library CSVs required by BR-CP.

Source repository:
https://huggingface.co/datasets/thuml/Time-Series-Library

The script creates exactly the directory structure expected by brcp/realdata.py
and writes SHA256SUMS.txt for reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import Request, urlopen

BASE = "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/"
FILES = [
    "ETT-small/ETTh1.csv",
    "ETT-small/ETTh2.csv",
    "ETT-small/ETTm1.csv",
    "ETT-small/ETTm2.csv",
    "electricity/electricity.csv",
    "traffic/traffic.csv",
    "weather/weather.csv",
    "exchange_rate/exchange_rate.csv",
    "illness/national_illness.csv",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "BRCP-reproducibility-downloader/1.0"})
    with urlopen(req, timeout=120) as r, dst.open("wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    if dst.stat().st_size < 100:
        raise RuntimeError(f"Downloaded file is unexpectedly small: {dst}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="Class-1",
        help="Output root. Default: ./Class-1 (the layout expected by BR-CP)",
    )
    ap.add_argument("--force", action="store_true", help="Redownload existing files")
    args = ap.parse_args()

    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = []

    print(f"Official source: https://huggingface.co/datasets/thuml/Time-Series-Library")
    print(f"Output root: {root}")

    for rel in FILES:
        dst = root / rel
        url = BASE + rel + "?download=true"
        if dst.exists() and not args.force:
            print(f"[exists]   {rel}")
        else:
            print(f"[download] {rel}")
            download(url, dst)
        digest = sha256(dst)
        manifest.append(f"{digest}  {rel}")
        print(f"           sha256={digest}")

    (root / "SHA256SUMS.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    (root / "SOURCE.txt").write_text(
        "Official dataset repository:\n"
        "https://huggingface.co/datasets/thuml/Time-Series-Library\n\n"
        "Dataset card license at download time should be independently verified "
        "and recorded in the paper/reproducibility package.\n",
        encoding="utf-8",
    )

    print("\nDone.")
    print(f"Set BRCP_DATA to: {root}")
    print(f"Hashes written to: {root / 'SHA256SUMS.txt'}")


if __name__ == "__main__":
    main()
