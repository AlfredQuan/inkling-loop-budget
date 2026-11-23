"""对真实推理后端做逐题配对扫圈，失败时保留完整的已成功记录。"""

import argparse
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from urllib.request import Request, urlopen


GRAMMAR = 'root ::= "A" | "B" | "C" | "D"'
ANSWER_PREFIX = "<|message_assistant|><|final|><|content_text|>"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def request(base, path, data=None):
    body = json.dumps(data).encode() if data is not None else None
    req = Request(base + path, data=body, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=120) as res:
        return json.load(res)


def score(case, base, depth):
    rendered = request(
        base, "/apply-template",
        {
            "messages": [{"role": "user", "content": case["prompt"]}],
            "reasoning_effort": "none", "enable_thinking": False,
        },
    )["prompt"]
    if not rendered.endswith("<|end_message|>"):
        raise RuntimeError("研究版的对话模板边界已变，请先检查答案前缀")
    rendered += ANSWER_PREFIX
    start = time.perf_counter()
    response = request(
        base, "/completion",
        {
            "prompt": rendered, "temperature": 0.0, "n_predict": 1,
            "grammar": GRAMMAR, "seed": 17, "cache_prompt": False,
        },
    )
    elapsed = 1000 * (time.perf_counter() - start)
    choice = response["content"].strip()
    timings = response["timings"]
    prompt_tokens = timings["prompt_n"]
    if (
        choice not in ("A", "B", "C", "D")
        or response.get("truncated")
        or timings["predicted_n"] != 1
        or prompt_tokens < 2
        or response.get("tokens_cached", 0) > prompt_tokens + 1
    ):
        raise RuntimeError(f"推理结果不满足受控单选协议：{case['id']} {response}")
    return {
        "id": case["id"], "depth": depth, "choice": choice,
        "correct": choice == case["answer"], "prompt_tokens": prompt_tokens,
        "generated_tokens": 1, "latency_ms": round(elapsed, 4),
        "prompt_ms": timings["prompt_ms"], "predicted_ms": timings["predicted_ms"],
        "render_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
    }


def run_canaries(base, depth):
    letters = "ABCD"
    for idx, letter in enumerate(letters):
        choices = [300, 100, 200, 400]
        choices[idx] = 2
        prompt = "What is 1+1? Reply with just the option letter.\n" + "\n".join(
            f"{label}. {n}" for label, n in zip(letters, choices)
        )
        out = score({"id": f"format-control-{letter}", "prompt": prompt, "answer": letter}, base, depth)
        if not out["correct"]:
            raise RuntimeError(f"格式阳性对照失败：{depth} 圈，预期 {letter}，实得 {out['choice']}")
    print(f"{depth} 圈：4 个格式阳性对照通过", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--depth", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8397")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    cases = read_jsonl(args.cases)
    random.Random(8835).shuffle(cases)
    cases = cases[:args.limit]
    if not cases or args.workers < 1:
        raise ValueError("没有可运行题目或并发数非法")
    run_canaries(args.url, args.depth)
    signature = {
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "depth": args.depth, "limit": args.limit, "grammar": GRAMMAR,
        "answer_prefix": ANSWER_PREFIX, "workers": args.workers,
    }
    signature_path = args.out.with_suffix(".signature.json")
    existing = set()
    if args.out.exists():
        if not signature_path.exists() or json.loads(signature_path.read_text(encoding="utf-8")) != signature:
            raise ValueError("拒绝续跑缺少签名或设置已更改的文件")
        prior = read_jsonl(args.out)
        if any(row["depth"] != args.depth for row in prior):
            raise ValueError("拒绝续跑不同圈数的文件")
        existing = {row["id"] for row in prior}
        if len(existing) != len(prior) or not existing.issubset({c["id"] for c in cases}):
            raise ValueError("输出文件存在重复或不同题集的记录")
    else:
        signature_path.write_text(json.dumps(signature) + "\n", encoding="utf-8")
    pending = [c for c in cases if c["id"] not in existing]
    with args.out.open("a", encoding="utf-8") as out, ThreadPoolExecutor(args.workers) as pool:
        for i, row in enumerate(pool.map(partial(score, base=args.url, depth=args.depth), pending), len(existing) + 1):
            out.write(json.dumps(row) + "\n")
            out.flush()
            if i % 40 == 0 or i == len(cases):
                print(f"{args.depth} 圈：{i}/{len(cases)}", flush=True)


if __name__ == "__main__":
    main()
