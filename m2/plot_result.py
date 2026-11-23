"""Render hold-out accuracy vs charged layer visits to a standalone SVG."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def draw(data, out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "inkling-loop-budget-v1"})
    fig, ax = plt.subplots(figsize=(8.6, 4.6), layout="constrained")
    ax.set(title="Which ARC questions benefit from two more passes?",
           xlabel="Charged layer visits vs. fixed 3 passes (restart proxy)",
           ylabel="Held-out accuracy")
    styles = {
        "margin": ("Choice margin", "#747e89", "--", "o"),
        "confidence_head": ("Confidence head", "#2867a6", "-", "s"),
        "state_head": ("State head", "#0b926f", "-", "D"),
        "subset": ("Subset mean", "#9e6020", ":", "^"),
    }
    for key, (label, color, line, marker) in styles.items():
        rows = data["curves"][key]
        xs = [1 - r["test"]["replay_work_saving"] for r in rows]
        ys = [r["test"]["accuracy"] for r in rows]
        ax.plot(xs, ys, label=label, color=color, ls=line, marker=marker, ms=4)
        i = next(i for i, r in enumerate(rows) if r["validation_cap"] == data["primary_validation_token_raise_cap"])
        ax.scatter(xs[i], ys[i], s=60, color=color, marker=marker, edgecolor="white", linewidth=1, zorder=4)
    fixed = data["fixed"]["test"]
    ax.scatter([0.5, 0.75, 1.0], [fixed[f"depth_{d}"] for d in (1, 2, 3)],
               color="#1f2937", marker="x", s=60, label="Fixed 1 / 2 / 3", zorder=5)
    ax.axvline(1.0, color="#b4bdc6", ls="--", lw=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.legend(loc="lower right", fontsize=8, frameon=False, ncols=2)
    fig.savefig(out, metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("metrics", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    draw(json.loads(a.metrics.read_text(encoding="utf-8")), a.out)
