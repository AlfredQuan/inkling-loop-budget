"""Score an already computed first-pass record with a released M2 head."""

import argparse
import json
import math
from pathlib import Path
import sys


def route(head, record):
    logits, n = record["logits"], record["tokens"]
    if len(logits) != 4 or not all(math.isfinite(x) for x in logits) or not isinstance(n, int) or n <= 0:
        raise ValueError("Expected four finite A–D logits and a positive token count")
    s = sorted(logits, reverse=True)
    exps = [math.exp(x - s[0]) for x in s]
    mass = sum(exps)
    probs = [x / mass for x in exps]
    features = [s[0] - s[1], probs[0], -sum(p * math.log(max(p, 1e-300)) for p in probs), math.log(n)]
    weights = head["coefficients"]
    if len(weights) == 100:
        state = record["state"]
        if len(state) != 96 or not all(math.isfinite(x) for x in state):
            raise ValueError("Expected a finite 96-component projected state")
        features += state
    if len(weights) != len(features):
        raise ValueError("Incompatible head")
    score = head["intercept"] + sum(a * b for a, b in zip(features, weights))
    threshold = head["threshold"]
    return {"score": score, "passes": 3 if threshold is None or score >= threshold else 1}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("heads", type=Path)
    p.add_argument("--head", choices=("confidence_head", "state_head"), default="state_head")
    a = p.parse_args()
    heads = json.loads(a.heads.read_text(encoding="utf-8"))
    print(json.dumps(route(heads[a.head], json.load(sys.stdin))))
