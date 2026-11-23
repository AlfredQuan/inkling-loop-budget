"""Fit and evaluate a signed-value head for buying two more Inkling passes."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


LETTERS = "ABCD"
BUDGETS = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90, 1.0)
PRIMARY_BUDGET = 0.40
MAX_LOSS = 0.05
MIN_SAVING = 0.10


def read(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def index(rows):
    result = {r["id"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate IDs")
    return result


def winning_choice(logits):
    if len(logits) != 4 or not np.isfinite(logits).all():
        raise ValueError("Invalid choice logits")
    return LETTERS[int(np.argmax(logits))]


def join(work, out):
    manifest = read(work / "manifest.jsonl")
    ids = index(manifest)
    runs = [index(read(work / f"depth-{depth}.jsonl")) for depth in (1, 2, 3)]
    if any(set(run) != set(ids) for run in runs):
        raise ValueError("Incomplete paired sweep")
    rows = []
    canaries = set()
    for case in manifest:
        examples = [run[case["id"]] for run in runs]
        if [r["depth"] for r in examples] != [1, 2, 3] or len({r["tokens"] for r in examples}) != 1:
            raise ValueError(f"Depth/token mismatch: {case['id']}")
        choices = [winning_choice(r["logits"]) for r in examples]
        if case["split"] == "canary":
            canaries.add(case["label"])
            if choices != [case["label"]] * 3:
                raise ValueError(f"Canary failed: {case['id']} {choices}")
            continue
        state = examples[0]["state"]
        if len(state) != 96 or not np.isfinite(state).all():
            raise ValueError("Invalid first-pass state")
        rows.append({
            **case, "tokens": examples[0]["tokens"], "choices": choices,
            "logits": examples[0]["logits"], "state": state,
        })
    if canaries != set(LETTERS):
        raise ValueError("Missing format canaries")
    payload = "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in rows)
    out.write_bytes(gzip.compress(payload.encode(), mtime=0))


def confidence_features(logits, sizes):
    ordered = np.sort(logits, axis=1)[:, ::-1]
    mass = np.exp(ordered - ordered[:, :1])
    probs = mass / mass.sum(axis=1, keepdims=True)
    return np.column_stack((
        ordered[:, 0] - ordered[:, 1],
        probs[:, 0], -(probs * np.log(np.maximum(probs, 1e-300))).sum(axis=1),
        np.log(sizes),
    ))


def weighted_threshold(scores, sizes, budget):
    """Inclusive threshold with whole ties and a cost cap, fit only on validation."""
    if budget == 1.0:
        return None
    total = 0
    limit = budget * sizes.sum()
    last = None
    levels, inverse = np.unique(scores, return_inverse=True)
    token_mass = np.bincount(inverse, weights=sizes)
    for value, weight in zip(levels[::-1], token_mass[::-1]):
        total += weight
        if total > limit + 1e-12:
            # Keep the boundary away from a validation example's exact score.
            return float(value + 1e-9 if last is None else value + (last - value) / 2)
        last = value
    return None


def policy(scores, threshold):
    return np.ones(len(scores), dtype=bool) if threshold is None else scores >= threshold


def paired_ci(values, groups):
    rng = np.random.default_rng(5317)
    draws = np.zeros(6000)
    for group in np.unique(groups):
        sub = values[groups == group]
        draws += sub[rng.integers(0, len(sub), (len(draws), len(sub)))].sum(axis=1)
    return np.quantile(draws / len(values), [0.025, 0.975]).tolist()


def summarize(correct, raises, sizes, groups):
    c1, _, c3 = correct.T
    rescued = (~c1) & c3
    reversed = c1 & (~c3)
    output = np.where(raises, c3, c1)
    delta = output.astype(int) - c3.astype(int)
    tokens_raised = float(sizes[raises].sum() / sizes.sum())
    return {
        "n": len(sizes), "accuracy": float(output.mean()),
        "accuracy_delta_vs_3": float(delta.mean()), "ci95_delta_vs_3": paired_ci(delta, groups),
        "correct_vs_fixed_3_lost": int((delta == -1).sum()),
        "correct_vs_fixed_3_gained": int((delta == 1).sum()),
        "rescued": int((rescued & raises).sum()), "rescue_total": int(rescued.sum()),
        "rescue_recall": float((rescued & raises).sum() / rescued.sum()) if rescued.any() else None,
        "missed_rescue": int((rescued & (~raises)).sum()),
        "harmed_by_extra_passes": int((reversed & raises).sum()),
        "correct_kept_by_early_exit": int((reversed & (~raises)).sum()),
        "request_raise_rate": float(raises.mean()), "token_raise_share": tokens_raised,
        "replay_work_saving": 0.5 - tokens_raised,
        "reuse_proxy_saving": 0.5 - 0.625 * tokens_raised,
    }


def modeled_scores(x, signed_delta, train):
    fitted = make_pipeline(StandardScaler(), Ridge(alpha=32.0)).fit(x[train], signed_delta[train])
    scaler, ridge = fitted.steps[0][1], fitted.steps[1][1]
    # Store an affine transform so inference does not depend on a pickle or sklearn.
    coef = ridge.coef_ / scaler.scale_
    intercept = ridge.intercept_ - scaler.mean_ @ coef
    return x @ coef + intercept, {"coefficients": coef.tolist(), "intercept": float(intercept)}


def evaluate(rows):
    index(rows)
    groups = np.array([r["subset"] for r in rows])
    parts = np.array([r["split"] for r in rows])
    if set(parts) != {"train", "validation", "test"} or set(groups) != {"ARC-Easy", "ARC-Challenge"}:
        raise ValueError("Unexpected split or subset")
    train, val, test = (parts == name for name in ("train", "validation", "test"))
    sizes = np.array([r["tokens"] for r in rows])
    if (
        (sizes <= 0).any() or (sizes != sizes.astype(int)).any()
        or any(len(r["choices"]) != 3 or any(c not in tuple(LETTERS) for c in r["choices"]) for r in rows)
        or any(r["label"] not in tuple(LETTERS) for r in rows)
    ):
        raise ValueError("Invalid token counts or choices")
    correct = np.array([[s == r["label"] for s in r["choices"]] for r in rows])
    signed_delta = correct[:, 2].astype(int) - correct[:, 0].astype(int)
    baseline_x = confidence_features(np.array([r["logits"] for r in rows]), sizes)
    state_x = np.column_stack((baseline_x, np.array([r["state"] for r in rows])))
    if baseline_x.shape[1] != 4 or state_x.shape[1] != 100 or not np.isfinite(state_x).all():
        raise ValueError("Invalid feature shape or non-finite features")
    table = {
        "margin": -baseline_x[:, 0],
        "subset": np.array([signed_delta[train & (groups == group)].mean() for group in groups]),
    }
    weights = {}
    for name, x in (("confidence_head", baseline_x), ("state_head", state_x)):
        table[name], weights[name] = modeled_scores(x, signed_delta, train)
    result = {
        "split_sizes": {name: int((parts == name).sum()) for name in ("train", "validation", "test")},
        "reward": "correct_at_3 - correct_at_1",
        "primary_validation_token_raise_cap": PRIMARY_BUDGET,
        "acceptance": {"min_replay_work_saving": MIN_SAVING, "min_accuracy_ci_lower_vs_3": -MAX_LOSS},
        "train_outcomes": {str(k): int((signed_delta[train] == k).sum()) for k in (-1, 0, 1)},
        "fixed": {}, "curves": {},
    }
    for label, ix in [("validation", val), ("test", test)] + [(g, test & (groups == g)) for g in np.unique(groups)]:
        result["fixed"][label] = {
            f"depth_{d}": float(correct[ix, d - 1].mean()) for d in (1, 2, 3)
        }
        result["fixed"][label]["oracle"] = summarize(correct[ix], signed_delta[ix] > 0, sizes[ix], groups[ix])
    decisions = {}
    for name, scores in table.items():
        curve = []
        for budget in BUDGETS:
            threshold = weighted_threshold(scores[val], sizes[val], budget)
            reached = policy(scores, threshold)
            row = {
                "validation_cap": budget, "threshold": threshold,
                "validation": summarize(correct[val], reached[val], sizes[val], groups[val]),
                "test": summarize(correct[test], reached[test], sizes[test], groups[test]),
            }
            curve.append(row)
            if budget == PRIMARY_BUDGET:
                decisions[name] = reached
                row["subsets"] = {
                    group: summarize(correct[ix], reached[ix], sizes[ix], groups[ix])
                    for group in np.unique(groups)
                    for ix in [test & (groups == group)]
                }
                row["passes_replay_gate"] = (
                    row["test"]["replay_work_saving"] >= MIN_SAVING
                    and row["test"]["ci95_delta_vs_3"][0] >= -MAX_LOSS
                )
                if name in weights:
                    weights[name]["threshold"] = threshold
        result["curves"][name] = curve
    traces = [
        {
            "id": r["id"], "arc_id": r["arc_id"], "subset": r["subset"],
            "correct_1": bool(correct[i, 0]), "correct_3": bool(correct[i, 2]),
            "raise": {name: bool(values[i]) for name, values in decisions.items()},
        }
        for i, r in enumerate(rows) if test[i]
    ]
    return result, weights, traces


def write(path, obj):
    path.write_bytes((json.dumps(obj, indent=2, allow_nan=False) + "\n").encode())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("evidence", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--join", type=Path, metavar="RAW_SWEEP_DIR")
    a = p.parse_args()
    if a.join:
        join(a.join, a.evidence)
    a.out.mkdir(parents=True, exist_ok=True)
    result, weights, traces = evaluate(read(a.evidence))
    result["paired_sha256"] = hashlib.sha256(a.evidence.read_bytes()).hexdigest()
    write(a.out / "metrics.json", result)
    write(a.out / "heads.json", weights)
    write(a.out / "decisions.json", traces)


if __name__ == "__main__":
    main()
