# Inkling Loop Budget

[English](README.md)

这是一个在冻结的 Inkling 可变循环研究版上训练停止头的项目。**M2** 让模型先跑一圈，从答案分数和内部状态预测再算两圈的净收益。它输出继续或停止的决策；实验离线回放固定 1/3 圈结果，尚未实现同一请求内部的续算。

首次使用时可以在 CPU 上安装 `requirements.txt`，然后运行 `python verify_release.py`。该命令会验证两轮实验的收录数据、重训两轮控制器并核对 M2 导出权重的逐题决策。详见 [M2 结果](m2/RESULTS.md)、[协议](m2/PROTOCOL.md) 和 [从头复现](m2/REPRODUCE.md)。

## M1 基线

M1 采用 608 道可精确判分的合成题，让控制器在模型生成前从 1–5 圈中选一个深度。完整结果见 [M1 实验报告](M1-results.md)。

需要 Linux、Python 3.11+、`curl` 和支持 `inkling2_loop` 的 llama.cpp。本次使用 H20、CUDA 12.6、[llama.cpp 源码修订 `7d9b931`](https://github.com/ggml-org/llama.cpp/tree/7d9b931df9bb7c689f067a7fed23e7f5253951f6)，以及 [ggml-org 研究版权重](https://huggingface.co/ggml-org/TML-d24-varloop-research-GGUF/tree/7af7f5e17b57395455bd8a37d8cfefa98824ca6e) 的 `TML-d24-varloop-research-IQ4_NL.gguf`：

```text
GGUF SHA-256  e9f70a167e54794bebce2d456b2118730b74fa7c35e23cb1abd4378f66b17bb2
GGUF bytes    17627940800
```

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v
mkdir -p work
python cases.py work/cases.jsonl
export MODEL=/absolute/path/TML-d24-varloop-research-IQ4_NL.gguf
export BIN=/absolute/path/llama
export PYTHON_BIN="$PWD/.venv/bin/python"
for depth in 1 3 2 5 4; do bash run-depth.sh "$depth" work; done
python controller.py work work/results
```

`BIN` 也可指向 `llama-server`。这台 H20（驱动 535.161.08）验证通过的配置是以 CUDA 12.6 编译 SM90；试用的预编译包可以枚举设备，但在运算时失败。[后端复现说明](BACKEND.md)给出构建参数和 CPU 冒烟路径。`run-depth.sh` 每次只在回环地址启动一个实例，结束后清理进程。重跑同一轮可以从匹配签名的完整 JSONL 记录继续。更换模型、后端、选项、模板或测量设置时，使用新结果目录。

导出的 `controller.json` 包含五个分类头及验证集选出的策略。本轮选参回退到固定 3 圈。`family-controller.json` 包含题族的训练集均值与预算价格；对陌生前缀回退到固定 3 圈。离线推断可用 `python route.py evidence/family-controller.json --tokens <渲染后提示的token数> < prompt.txt`。这两份权重未经跨任务、跨模板的上线验证。

[实验协议](PROTOCOL.md) 定义了配对基线、切片、判分规则和推进门槛。[M1 实验报告](M1-results.md) 汇总实测结论。`evidence/` 保存 608 道题、3,040 条真实预测、权重和统计结果。下载项目后可直接跑 `python controller.py evidence work/replayed` 从这些记录重训，无需 GPU。`decisions.json` 保存每题选择和五个正确率估计；用 `python inspect_case.py evidence evidence/decisions.json --id 23cd539d20ebc0e884e9` 可以观察单题在不同圈数下的答案；`transfer.py` 可独立复跑留一题族验证。

官方来源：[可变循环研究版模型卡](https://huggingface.co/thinkingmachines/TML-d24-varloop-research)、[Hugging Face 对架构与 llama.cpp 支持的说明](https://huggingface.co/blog/inkling2mini)。
