"""生成有确定判分规则的受控任务；不包含真实用户数据。"""

import argparse
import hashlib
import json
import random
from collections import deque
from pathlib import Path


def arithmetic(rng, size):
    value = start = rng.randint(1, 24)
    operations = []
    for _ in range(size):
        op = rng.choice(("+", "-", "*"))
        number = rng.randint(2, 9)
        value = value + number if op == "+" else value - number if op == "-" else value * number
        operations.append(f"{op} {number}")
    question = (
        f"Start at {start}. Apply these operations from left to right, ignoring "
        f"arithmetic precedence: {', '.join(operations)}. What is the final value?"
    )
    return question, value, [value + delta for delta in (-9, -2, 1, 4, 8)]


def program(rng, size):
    a, b = initial = rng.sample(range(1, 12), 2)
    code = []
    for _ in range(size):
        line = rng.choice(("a,b=b,a", "a=a+b", "b=b-a", "b=a+2*b", "a=a-2*b"))
        if line == "a,b=b,a":
            a, b = b, a
        elif line == "a=a+b":
            a += b
        elif line == "b=b-a":
            b -= a
        elif line == "b=a+2*b":
            b = a + 2 * b
        else:
            a -= 2 * b
        code.append(line)
    target = rng.choice((0, 1))
    answer = (a, b)[target]
    return (
        f"In Python, a={initial[0]} and b={initial[1]}. Execute these statements "
        f"in order: {'; '.join(code)}. What is the final value of {'ab'[target]}?",
        answer,
        [a, b, answer - 3, answer + 1, answer * 2, answer + 5],
    )


def graph(rng, size):
    nodes = rng.sample(list("JKLMNOPQRSTUVWXYZ"), size + 3)
    edges = list(zip(nodes[:size], nodes[1 : size + 1]))
    for x, y in rng.sample([(x, y) for x in nodes for y in nodes if x != y], size):
        if (x, y) not in edges:
            edges.append((x, y))
    rng.shuffle(edges)
    start, end = nodes[0], nodes[size]
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        node, distance = queue.popleft()
        if node == end:
            break
        for x, y in edges:
            if x == node and y not in seen:
                seen.add(y)
                queue.append((y, distance + 1))
    return (
        "All edges are directed and have unit length: "
        + ", ".join(f"{x}->{y}" for x, y in edges)
        + f". What is the minimum number of edges from {start} to {end}?",
        distance,
        [n for n in range(1, size + 4) if abs(distance - n) <= 3],
    )


def table(rng, size):
    values = [(rng.choice("XYZ"), rng.choice("PQ"), rng.randint(1, 23)) for _ in range(size + 3)]
    group, kind, _ = rng.choice(values)
    answer = sum(n for g, k, n in values if (g, k) == (group, kind))
    return (
        "Rows are (group, kind, amount): " + ", ".join(f"({g},{k},{n})" for g, k, n in values)
        + f". Sum amount only for rows with group={group} and kind={kind}.",
        answer,
        [
            sum(n for g, _, n in values if g == group),
            sum(n for _, k, n in values if k == kind),
            answer + 2,
            answer + 7,
            max(0, answer - 1),
        ],
    )


FAMILIES = (arithmetic, program, graph, table)
SPLITS = {"train": 72, "val": 24, "test": 32, "ood": 24}
SEEDS = {"train": 123457, "val": 112323, "test": 837402, "ood": 434103}
LETTERS = "ABCD"


def make_cases(counts=None):
    used = set()
    for split, count in (counts or SPLITS).items():
        rng = random.Random(SEEDS[split])
        option_rng = random.Random(SEEDS[split] ^ 0x6D47)
        split_cases = []
        for task in FAMILIES:
            # 题目长度与正确选项位置分别产生，不能用同一个 idx 取模。
            labels = [idx % len(LETTERS) for idx in range(count)]
            option_rng.shuffle(labels)
            for idx in range(count):
                size = (7 if split == "ood" else 2) + idx % (4 if split == "ood" else 5)
                while True:
                    question, answer, alternatives = task(rng, size)
                    if question not in used:
                        used.add(question)
                        break
                alternatives = sorted(set(alternatives) - {answer})
                wrong = rng.sample(alternatives, 3)
                label = labels[idx]
                wrong.insert(label, answer)
                choices = "\n".join(f"{letter}. {choice}" for letter, choice in zip(LETTERS, wrong))
                prompt = (
                    "Solve the following question. Output exactly one option letter "
                    "(A, B, C, or D).\n\n" + question + "\n" + choices
                )
                split_cases.append(
                    {
                        "id": hashlib.sha256(question.encode()).hexdigest()[:20],
                        "split": split, "family": task.__name__, "size": size,
                        "prompt": prompt, "answer": LETTERS[label],
                    }
                )
        rng.shuffle(split_cases)
        yield from split_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    counts = {s: 4 for s in SPLITS} if args.smoke else None
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in make_cases(counts)),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
