"""查看题目与五个真实答案，或按相对 3 圈的得失查找例子。"""

import argparse
import json
from pathlib import Path

from sweep import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="包含 cases.jsonl 和五份推理记录的目录")
    parser.add_argument("decisions", type=Path)
    parser.add_argument("--id", help="只查看该题")
    parser.add_argument("--split", default="test", choices=("train", "val", "test", "ood"))
    parser.add_argument("--change", choices=("lost", "gained", "any"), default="any")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    decisions = {d["id"]: d["depth"] for d in json.loads(args.decisions.read_text(encoding="utf-8"))}
    answers = {
        depth: {r["id"]: r for r in read_jsonl(args.run / f"depth-{depth}.jsonl")}
        for depth in range(1, 6)
    }
    shown = 0
    for case in read_jsonl(args.run / "cases.jsonl"):
        case_id = case["id"]
        if (args.id and case_id != args.id) or (not args.id and case["split"] != args.split):
            continue
        selected = decisions[case_id]
        delta = int(answers[selected][case_id]["correct"]) - int(answers[3][case_id]["correct"])
        if args.change != "any" and delta != {"lost": -1, "gained": 1}[args.change]:
            continue
        print(f"{case_id} · {case['split']} · {case['family']}")
        print(case["prompt"])
        print(f"正确答案 {case['answer']}；控制器选 {selected} 圈")
        for depth, by_id in answers.items():
            row = by_id[case_id]
            print(f"{depth} 圈：{row['choice']}（{'对' if row['correct'] else '错'}） {row['latency_ms']:.1f} ms")
        print()
        shown += 1
        if shown >= args.limit:
            break
    if not shown:
        print("没有匹配的记录。")


if __name__ == "__main__":
    main()
