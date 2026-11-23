"""由 metrics.json 生成准确率—计算成本图；绘图额外依赖 matplotlib。"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))["results"]
    fig, panels = plt.subplots(1, 2, figsize=(9.8, 4.4), layout="constrained", sharey=True)
    for ax, split, label in zip(panels, ("test", "ood"), ("Held-out test", "Longer chains")):
        result = metrics[split]
        fixed = [result[f"fixed-{i}"] for i in range(1, 6)]
        xs = [1 - m["work_savings_vs_3"] for m in fixed]
        ys = [m["accuracy"] for m in fixed]
        ax.plot(xs, ys, "o-", color="#3f5268", label="Fixed passes")
        for i, (x, y) in enumerate(zip(xs, ys), 1):
            ax.annotate(str(i), (x, y), xytext=(0, 7), textcoords="offset points", ha="center")
        for key, name, marker, color in (
            ("controller", "Trained controller", "*", "#146c56"),
            ("family_baseline", "Task-family rule", "s", "#846012"),
        ):
            m = result[key]
            ax.scatter(
                1 - m["work_savings_vs_3"], m["accuracy"], label=name,
                marker=marker, facecolors="none", edgecolors=color,
                s=154 if key == "controller" else 65, zorder=4,
            )
        ax.axvline(1, alpha=0.24, color="#485365", linestyle="--")
        ax.set(
            title=f"{label} (n={fixed[0]['n']})",
            xlabel="Layer visits × prompt tokens (fixed-3 = 100%)",
            xlim=(0.46, 1.57),
        )
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.grid(axis="y", alpha=0.20)
        ax.margins(y=0.21)
        ax.spines[["top", "right"]].set_visible(False)
    panels[0].set_ylabel("Exact accuracy")
    handles, labels = panels[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, fontsize=8, frameon=False)
    fig.suptitle("Inkling research checkpoint · one-token answers", fontsize=13)
    fig.savefig(args.out, bbox_inches="tight")


if __name__ == "__main__":
    main()
