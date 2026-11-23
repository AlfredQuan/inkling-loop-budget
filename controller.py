"""从扫圈真值训练请求级控制器，并在未参与选参的题上比较。"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from route import features, predict, question_prefix
from sweep import read_jsonl


DEPTHS = np.arange(1, 6)
BASE = 2  # 3 圈对应的列
PENALTIES = (0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 2.0)
VAL_MAX_LOSS = 0.02
GATE_SAVINGS = 0.10
GATE_LOWER_CI = -0.05


def load(work):
    cases = read_jsonl(work / "cases.jsonl")
    ids = [c["id"] for c in cases]
    if not cases or len(set(ids)) != len(ids):
        raise ValueError("题集为空或题目 ID 重复")
    outputs = []
    for depth in DEPTHS:
        rows = read_jsonl(work / f"depth-{depth}.jsonl")
        by_id = {r["id"]: r for r in rows}
        if len(rows) != len(by_id) or set(ids) != set(by_id):
            raise ValueError(f"{depth} 圈结果缺失、重复或不属于题集")
        aligned = [by_id[i] for i in ids]
        for case, row in zip(cases, aligned):
            if (
                row["depth"] != depth or row["choice"] not in "ABCD"
                or len(row["choice"]) != 1 or row["generated_tokens"] != 1
                or row["prompt_tokens"] < 2 or not math.isfinite(row["latency_ms"])
                or row["latency_ms"] <= 0
                or row["correct"] != (case["answer"] == row["choice"])
            ):
                raise ValueError(f"无效评测记录：{row}")
        outputs.append(aligned)
    for group in zip(*outputs):
        if len({(r["render_sha256"], r["prompt_tokens"]) for r in group}) != 1:
            raise ValueError(f"跨圈提示渲染或 token 数不一致：{group[0]['id']}")
    correct = np.array([[r["correct"] for r in column] for column in outputs], dtype=int).T
    ms = np.array([[r["latency_ms"] for r in column] for column in outputs]).T
    tokens = np.array([r["prompt_tokens"] for r in outputs[0]])
    return cases, correct, ms, tokens


def fit(cases, correct, train):
    vectorizer = DictVectorizer()
    rows = [features(c["prompt"]) for c in cases]
    vectorizer.fit([rows[i] for i in train])
    x = vectorizer.transform(rows)
    probabilities, heads = [], []
    for labels in correct.T:
        y = labels[train]
        if np.unique(y).size == 1:
            heads.append({"intercept": 20.0 if y[0] else -20.0, "coefficients": []})
            probabilities.append(np.full(len(cases), y[0], dtype=float))
            continue
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        model.fit(x[train], y)
        probabilities.append(model.predict_proba(x)[:, 1])
        heads.append({
            "intercept": float(model.intercept_[0]), "coefficients": model.coef_[0].tolist(),
        })
    return np.array(probabilities).T, {
        "features": vectorizer.get_feature_names_out().tolist(), "heads": heads,
        "depths": DEPTHS.tolist(),
    }


def select(prob, work, correct, val):
    """只在验证集选价格；收益相同时偏向观察到的更高准确率。"""
    baseline = correct[val, BASE].mean()
    options = [(work[val, BASE].sum(), -baseline, "fixed-3", np.full(len(prob), BASE), None)]
    for price in PENALTIES:
        picks = np.argmax(prob - price * (work / work[val, BASE].mean()), axis=1)
        acc = correct[val, picks[val]].mean()
        if acc >= baseline - VAL_MAX_LOSS - 1e-12:
            options.append((work[val, picks[val]].sum(), -acc, "learned", picks, price))
    _, _, policy, picks, price = min(options, key=lambda x: x[:2])
    return policy, picks, price


def bootstrap(delta, families):
    """按题型分层，保持题内配对；区间只反映题集采样不确定性。"""
    rng = np.random.default_rng(30284)
    samples = np.zeros(4000)
    for family in np.unique(families):
        slice_delta = delta[families == family]
        index = rng.integers(0, len(slice_delta), (len(samples), len(slice_delta)))
        samples += slice_delta[index].sum(axis=1)
    return np.quantile(samples / len(delta), [0.025, 0.975]).tolist()


def stats(ix, picks, correct, work, ms, families):
    outcomes = correct[ix, picks[ix]]
    reference = correct[ix, BASE]
    cost = work[ix, picks[ix]].sum()
    return {
        "n": int(len(ix)),
        "accuracy": float(outcomes.mean()),
        "delta_vs_3": float((outcomes - reference).mean()),
        "ci95_delta_vs_3": bootstrap(outcomes - reference, families[ix]),
        "lost_vs_3": int(((reference == 1) & (outcomes == 0)).sum()),
        "gained_vs_3": int(((reference == 0) & (outcomes == 1)).sum()),
        "mean_depth": float(DEPTHS[picks[ix]].mean()),
        "work_savings_vs_3": float(1 - cost / work[ix, BASE].sum()),
        "mean_remote_ms": float(ms[ix, picks[ix]].mean()),
        "remote_time_savings_vs_3": float(1 - ms[ix, picks[ix]].sum() / ms[ix, BASE].sum()),
        "p95_remote_ms": float(np.quantile(ms[ix, picks[ix]], 0.95)),
        "depth_counts": np.bincount(picks[ix], minlength=len(DEPTHS)).tolist(),
    }


def analyze(workdir):
    cases, correct, ms, tokens = load(workdir)
    families = np.array([c["family"] for c in cases])
    split_field = np.array([c["split"] for c in cases])
    splits = {s: np.flatnonzero(split_field == s) for s in ("train", "val", "test", "ood")}
    if not all(len(ix) for ix in splits.values()):
        raise ValueError("缺少 train、val、test 或 ood 分区")
    # 仅作结构成本代理：固定 12 层 + 12 层 × 循环次数，不等同 FLOPs。
    work = tokens[:, None] * (12 + 12 * DEPTHS)
    train, val = splits["train"], splits["val"]
    group_prob = np.full(correct.shape, correct[train].mean(axis=0))
    group_heads = {}
    for f in np.unique(families):
        fi = train[families[train] == f]
        if len(fi):
            averages = correct[fi].mean(axis=0)
            group_prob[families == f] = averages
            for i in fi:
                prefix = question_prefix(cases[i]["prompt"])
                if prefix in group_heads and not np.array_equal(group_heads[prefix], averages):
                    raise ValueError("题族前缀冲突，无法导出同等规则")
                group_heads[prefix] = averages.tolist()
    prob, weights = fit(cases, correct, train)
    policy, picks, price = select(prob, work, correct, val)
    group_policy, group_picks, group_price = select(group_prob, work, correct, val)
    weights.update({
        "penalty": price, "policy": policy,
        "cost_normalizer": float(work[val, BASE].mean()),
        "note": "当前成本归一化使用后端 tokenizer 的 prompt token 数；仅支持同配置。",
    })
    group_model = {
        "groups": group_heads, "depths": DEPTHS.tolist(), "policy": group_policy,
        "penalty": group_price, "cost_normalizer": float(work[val, BASE].mean()),
        "note": "仅对已知题目模板估计训练集正确率；遇到未知前缀则回退固定 3 圈。",
    }
    for i, case in enumerate(cases):
        if predict(group_model, case["prompt"], int(tokens[i]))["depth"] != DEPTHS[group_picks[i]]:
            raise ValueError(f"导出的题族规则与评估不一致：{case['id']}")
    # Oracle 可以读到真值；无正确出口时取最便宜的一圈。只用来量出空间。
    oracle = np.argmax(correct, axis=1)
    configs = {f"fixed-{d}": np.full(len(cases), d - 1) for d in DEPTHS}
    configs.update(controller=picks, family_baseline=group_picks, oracle=oracle)
    results = {}
    for split in ("val", "test", "ood"):
        ix = splits[split]
        results[split] = {label: stats(ix, p, correct, work, ms, families) for label, p in configs.items()}
        mix = np.bincount(picks[ix], minlength=len(DEPTHS)) / len(ix)
        results[split]["random_same_depth_mix"] = {
            "accuracy": float(correct[ix].mean(axis=0) @ mix),
            "work_savings_vs_3": float(1 - work[ix].mean(axis=0) @ mix / work[ix, BASE].mean()),
        }
        results[split]["families"] = {
            f: {
                label: stats(sub, p, correct, work, ms, families)
                for label, p in configs.items()
            }
            for f in np.unique(families[ix])
            for sub in [ix[families[ix] == f]]
        }
        results[split]["transitions"] = [
            {
                "from": int(d), "to": int(d + 1),
                "lost": int(((correct[ix, d - 1] == 1) & (correct[ix, d] == 0)).sum()),
                "gained": int(((correct[ix, d - 1] == 0) & (correct[ix, d] == 1)).sum()),
            }
            for d in DEPTHS[:-1]
        ]
    return {
        "cases_sha256": hashlib.sha256((workdir / "cases.jsonl").read_bytes()).hexdigest(),
        "train_n": len(train), "selected_policy": policy, "penalty": price,
        "family_penalty": group_price, "results": results,
        "gate": {
            "required_work_savings": GATE_SAVINGS, "min_lower_accuracy_ci": GATE_LOWER_CI,
            "pass": all(
                results[s]["controller"]["work_savings_vs_3"] >= GATE_SAVINGS
                and results[s]["controller"]["ci95_delta_vs_3"][0] >= GATE_LOWER_CI
                for s in ("test", "ood")
            ),
        },
    }, weights, group_model, [
        {"id": c["id"], "split": c["split"], "depth": int(DEPTHS[p]), "family": c["family"],
         "p_correct": prob[i].tolist(), "family_depth": int(DEPTHS[group_picks[i]])}
        for i, (c, p) in enumerate(zip(cases, picks))
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    result, model, group_model, decisions = analyze(args.work)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("metrics.json", result), ("controller.json", model), ("family-controller.json", group_model),
        ("decisions.json", decisions),
    ):
        (args.out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
