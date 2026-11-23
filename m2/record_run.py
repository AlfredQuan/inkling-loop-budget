"""Record immutable inputs, observed layer counts, and checksums for a GPU run."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from arc import REVISION


BACKEND_REVISION = "7d9b931df9bb7c689f067a7fed23e7f5253951f6"
MODEL_REVISION = "7af7f5e17b57395455bd8a37d8cfefa98824ca6e"
MODEL_SHA256 = "e9f70a167e54794bebce2d456b2118730b74fa7c35e23cb1abd4378f66b17bb2"
PUBLIC = ("paired.jsonl.gz", "metrics.json", "heads.json", "decisions.json", "accuracy-cost.svg")


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def record(work, artifacts, binary, model, hardware, driver, cuda):
    if sha(model) != MODEL_SHA256:
        raise ValueError("Unexpected model file hash")
    settings = {}
    for d in (1, 2, 3):
        log = (work / f"depth-{d}.log").read_text(encoding="utf-8")
        count = 12 + 12 * d
        if not re.search(rf"^.*Using metadata override.*inkling2_loop\.loop\.count.* = {d}$", log, flags=re.M):
            raise ValueError(f"Missing depth {d} metadata override")
        if not re.search(rf"^.*n_layer_all\s*= {count}$", log, flags=re.M):
            raise ValueError(f"Missing depth {d} layer count")
        if f"offloaded {count+1}/{count+1} layers to GPU" not in log:
            raise ValueError(f"Incomplete GPU offload at depth {d}")
        predictions = work / f"depth-{d}.jsonl"
        settings[str(d)] = {
            "effective_layers": count, "gpu_offloaded_including_output": count + 1,
            "output_rows_including_canaries": len(predictions.read_bytes().splitlines()),
            "raw_trace_sha256": sha(predictions),
        }
    return {
        "build": {"backend_revision": BACKEND_REVISION, "probe_sha256": sha(binary)},
        "environment": {"hardware": hardware, "driver": driver, "cuda": cuda, "batch_slots": 4},
        "model": {"revision": MODEL_REVISION, "format": "IQ4_NL", "sha256": MODEL_SHA256},
        "dataset_revision": REVISION, "case_manifest_sha256": sha(work / "manifest.jsonl"),
        "per_depth": settings,
        "public_sha256": {k: sha(artifacts / k) for k in PUBLIC},
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("artifacts", type=Path)
    p.add_argument("binary", type=Path)
    p.add_argument("model", type=Path)
    p.add_argument("--hardware", required=True)
    p.add_argument("--driver", required=True)
    p.add_argument("--cuda", required=True)
    a = p.parse_args()
    result = record(a.work, a.artifacts, a.binary, a.model, a.hardware, a.driver, a.cuda)
    (a.artifacts / "run.json").write_bytes((json.dumps(result, indent=2) + "\n").encode())
