"""检查发布用实验记录完整性，并复现导出权重的逐题决策。"""

import argparse
import hashlib
import json
import re
from pathlib import Path

from route import predict
from sweep import read_jsonl


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(work, results):
    cases = read_jsonl(work / "cases.jsonl")
    metrics = json.loads((results / "metrics.json").read_text(encoding="utf-8"))
    model = json.loads((results / "controller.json").read_text(encoding="utf-8"))
    group_model = json.loads((results / "family-controller.json").read_text(encoding="utf-8"))
    decisions = json.loads((results / "decisions.json").read_text(encoding="utf-8"))
    assert metrics["cases_sha256"] == file_hash(work / "cases.jsonl")
    assert len(cases) == len(decisions) == len({c["id"] for c in cases})
    evidence = {}
    for d in range(1, 6):
        text = (work / f"depth-{d}.server.log").read_text(encoding="utf-8")
        key = re.search(rf"^.*metadata override.*inkling2_loop\.loop\.count.*= (?:[0-9.]+ I )?{d}\s*$", text, re.M)
        layers = re.search(rf"^.*n_layer_all\s*=\s*{12 + 12*d}\s*$", text, re.M)
        offload = re.search(r"^.*offloaded .*layers to GPU.*$", text, re.M)
        assert key and layers and offload, f"{d} 圈缺少后端确认"
        rows_file = work / f"depth-{d}.jsonl"
        rows = read_jsonl(rows_file)
        assert len(rows) == len(cases)
        assert {r["id"] for r in rows} == {c["id"] for c in cases}
        evidence[str(d)] = {
            "metadata": key[0].strip(), "layer_count": layers[0].strip(),
            "offload": offload[0].strip(), "rows": len(rows),
            "sha256": file_hash(rows_file),
        }
    tokens = {r["id"]: r["prompt_tokens"] for r in read_jsonl(work / "depth-3.jsonl")}
    for case, decision in zip(cases, decisions):
        assert case["id"] == decision["id"]
        out = predict(model, case["prompt"], tokens[case["id"]])
        assert out["depth"] == decision["depth"], case["id"]
        assert all(abs(a - b) < 1e-7 for a, b in zip(out["p_correct"].values(), decision["p_correct"]))
        assert predict(group_model, case["prompt"], tokens[case["id"]])["depth"] == decision["family_depth"]
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.work, args.results), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
