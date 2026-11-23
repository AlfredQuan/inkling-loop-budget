"""Validate the shipped evidence and replay both studies on CPU."""

import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare(a, b, key="root"):
    if isinstance(a, dict):
        require(isinstance(b, dict) and a.keys() == b.keys(), f"Keys diverge: {key}")
        for k in a:
            compare(a[k], b[k], f"{key}.{k}")
    elif isinstance(a, list):
        require(isinstance(b, list) and len(a) == len(b), f"List length diverges: {key}")
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{key}[{i}]")
    elif isinstance(a, float):
        require(isinstance(b, (int, float)) and math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-8), f"Value diverges: {key}")
    else:
        require(type(a) is type(b) and a == b, f"Value diverges: {key}")


def call(*args):
    return subprocess.check_output([sys.executable, *map(str, args)], cwd=ROOT, text=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    m1 = ROOT / "evidence"
    m2 = ROOT / "m2" / "evidence"
    manifests = (load(m1 / "run-evidence.json"), load(m2 / "run.json"))
    for d, record in manifests[0].items():
        require(record["sha256"] == sha(m1 / f"depth-{d}.jsonl"), f"M1 depth {d} checksum mismatch")
    for name, expected in manifests[1]["public_sha256"].items():
        require(sha(m2 / name) == expected, f"M2 checksum mismatch: {name}")
    with tempfile.TemporaryDirectory(prefix="inkling-replay-") as scratch:
        work = Path(scratch)
        call("controller.py", m1, work / "m1")
        call("transfer.py", m1, work / "transfer.json")
        call("m2/fit.py", m2 / "paired.jsonl.gz", work / "m2")
        for prefix, directory, files in (
            ("m1", m1, ("metrics.json", "controller.json", "family-controller.json", "decisions.json")),
            ("m2", m2, ("metrics.json", "heads.json", "decisions.json")),
        ):
            for file in files:
                compare(load(directory / file), load(work / prefix / file), f"{prefix}/{file}")
        compare(load(m1 / "transfer.json"), load(work / "transfer.json"), "m1/transfer")
    with gzip.open(m2 / "paired.jsonl.gz", "rt", encoding="utf-8") as file:
        rows = {r["id"]: r for r in map(json.loads, file)}
    require(len(rows) == 1088, "Unexpected ARC case count")
    # Use the stdlib inference implementation to check export parity independently
    # of the numpy/scikit-learn evaluation path.
    from m2.route import route

    heads = load(m2 / "heads.json")
    decisions = load(m2 / "decisions.json")
    require(len(decisions) == 384, "Unexpected test decision count")
    for d in decisions:
        for name, head in heads.items():
            picks_3 = route(head, rows[d["id"]])["passes"] == 3
            require(picks_3 == d["raise"][name], f"{name} export mismatch: {d['id']}")
    print("Verified file hashes, M1 and M2 replay, and 768 exported M2 head decisions.")


if __name__ == "__main__":
    verify()
