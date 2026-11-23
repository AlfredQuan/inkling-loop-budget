"""Download the pinned public ARC release after checking SHA-256."""

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

from arc import FILES, REVISION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--hub", default="https://huggingface.co")
    a = p.parse_args()
    root = f"{a.hub.rstrip('/')}/datasets/allenai/ai2_arc/resolve/{REVISION}"
    for key, expected in FILES.items():
        relative = f"{key}-00000-of-00001.parquet"
        file = a.directory / relative
        if file.exists():
            payload = file.read_bytes()
        else:
            with urlopen(f"{root}/{relative}?download=true", timeout=30) as response:
                payload = response.read(2_000_000)
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"Incorrect file hash: {file}")
        if not file.exists():
            file.parent.mkdir(parents=True, exist_ok=True)
            partial = file.with_suffix(".part")
            partial.write_bytes(payload)
            partial.replace(file)
        print(file)


if __name__ == "__main__":
    main()
