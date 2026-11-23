# Reproduce the ARC run

The CPU replay in the root README is sufficient to reproduce the tables and
policy decisions. This page describes the separate GPU run that generated
the input evidence.

| Input | Pin |
| --- | --- |
| Dataset | `allenai/ai2_arc` revision `210d026faf9955653af8916fad021475a3f00453`; six source SHA-256 values in [`arc.py`](arc.py) |
| Base weights | `ggml-org/TML-d24-varloop-research-GGUF` revision `7af7f5e17b57395455bd8a37d8cfefa98824ca6e`, `TML-d24-varloop-research-IQ4_NL.gguf`, SHA-256 `e9f70a167e54794bebce2d456b2118730b74fa7c35e23cb1abd4378f66b17bb2` |
| Native backend | `ggml-org/llama.cpp` commit `7d9b931df9bb7c689f067a7fed23e7f5253951f6` |
| Verified GPU environment | Linux x86-64, GCC 12.2, single NVIDIA H20, driver 535.161.08, CUDA 12.6, SM90 |
| Classifier | Python 3.11, [`requirements.txt`](../requirements.txt) |

On Linux, build llama.cpp's shared libraries using [the pinned build
instructions](../BACKEND.md). Set `LLAMA_SRC` to the source directory and
`LLAMA_LIB` to the build's `bin/` directory. Ensure the CUDA 12 runtime
libraries are accessible via `LD_LIBRARY_PATH`. Run from the project root
in a clean working directory:

```bash
python -m pip install -r requirements.txt pyarrow==23.0.1
python m2/download_arc.py work/arc-source
python m2/arc.py work/arc-source work/arc-run
bash m2/build.sh work/loop-probe
export PROBE="$PWD/work/loop-probe"
export MODEL=/absolute/path/TML-d24-varloop-research-IQ4_NL.gguf
bash m2/run.sh work/arc-run
```

Only the upstream source download and dependency installation need network
access. The GPU probe batches four independent prompts, right pads each with
unused causal suffix tokens, and reads logits only at the last real prompt
token. At one pass it reads `result_norm` using the backend callback and
projects the vector to 96 dimensions with a fixed seed. The readout comes
after the six final layers, not from the loop core itself. Batching, hardware
and quantized kernels can change numeric outputs. A 24-item smoke comparison
between one and four slots agreed on all answer choices; the numeric outputs
were not bitwise identical. Reproduce the experiment with four slots, and
treat the shipped trace as the reference for CPU replay.

`work/arc-run/results/` contains the model coefficients and metrics. For each
pass the runner checks the effective layer count and full GPU offload.
`fit.py --join` also checks that every case appears at all three depths with
the same prompt token count, and that four format controls pass at every
depth. Preserve raw logs locally when diagnosing a discrepancy. The public
[run record](evidence/run.json) records the observed settings and artifact
checksums without local paths.
