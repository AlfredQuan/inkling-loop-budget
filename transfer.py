"""补充诊断：每次把一个题族从训练和选参中完全拿掉。"""

import argparse
import json
from pathlib import Path

import numpy as np

from controller import BASE, DEPTHS, fit, load, select, stats


def evaluate(work):
    cases, correct, ms, tokens = load(work)
    families = np.array([c["family"] for c in cases])
    splits = np.array([c["split"] for c in cases])
    cost = tokens[:, None] * (12 + 12 * DEPTHS)
    result = {}
    for family in sorted(set(families)):
        train = np.flatnonzero((families != family) & (splits == "train"))
        val = np.flatnonzero((families != family) & (splits == "val"))
        probabilities, _ = fit(cases, correct, train)
        policy, picks, penalty = select(probabilities, cost, correct, val)
        row = {"train_n": len(train), "val_n": len(val), "policy": policy, "penalty": penalty}
        for split in ("test", "ood"):
            ix = np.flatnonzero((families == family) & (splits == split))
            row[split] = {
                "controller": stats(ix, picks, correct, cost, ms, families),
                "fixed-3": stats(ix, np.full(len(cases), BASE), correct, cost, ms, families),
            }
        result[family] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.write_text(json.dumps(evaluate(args.work), indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
