"""用导出的控制器权重为一个提示选择循环预算，无需导入训练依赖。"""

import argparse
import json
import math
import re
import sys
from pathlib import Path


def question_text(prompt):
    return prompt.split("\nA. ", 1)[0].split("\n\n", 1)[-1]


def question_prefix(prompt):
    return " ".join(re.findall(r"[A-Za-z]+", question_text(prompt).lower())[:4])


def features(prompt):
    """廉价特征：只看问题文本，不读取选项、真值、元数据或未来输出。"""
    question = question_text(prompt)
    numbers = re.findall(r"\d+", question)
    chars = {
        "question_chars": len(question) / 150,
        "number_count": len(numbers) / 10,
        "two_digit_count": sum(len(n) > 1 for n in numbers) / 8,
    }
    for symbol in (",", ";", "*", "-", "+", "=", "->", "("):
        chars[f"count:{symbol}"] = question.count(symbol) / 10
    prefix = question_prefix(prompt)
    chars[f"prefix:{prefix}"] = 1.0
    for key in list(chars):
        if not key.startswith("prefix:"):
            chars[f"{prefix}/{key}"] = chars[key]
    return chars


def predict(model, prompt, tokens):
    if tokens <= 0:
        raise ValueError("tokens 必须是渲染后提示的正整数 token 数")
    groups = model.get("groups")
    if groups is not None:
        scores = groups.get(question_prefix(prompt))
        if scores is None:
            return {"depth": 3, "p_correct": None}
    else:
        values = features(prompt)
        x = [values.get(key, 0) for key in model["features"]]
        scores = []
        for head in model["heads"]:
            logits = head["intercept"] + sum(a * b for a, b in zip(x, head["coefficients"]))
            scores.append(1 / (1 + math.exp(-max(-40, min(40, logits)))))
    selected = 3
    if model["policy"] == "learned":
        selected = max(
            zip(model["depths"], scores),
            key=lambda item: item[1]
            - model["penalty"] * tokens * (12 + 12 * item[0]) / model["cost_normalizer"],
        )[0]
    return {"depth": selected, "p_correct": dict(zip(model["depths"], scores))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--tokens", type=int, required=True)
    args = parser.parse_args()
    model = json.loads(args.model.read_text(encoding="utf-8"))
    print(json.dumps(predict(model, sys.stdin.read(), args.tokens), indent=2))


if __name__ == "__main__":
    main()
