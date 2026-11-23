# 推理后端复现

H20 实验以 llama.cpp [`7d9b931df9bb7c689f067a7fed23e7f5253951f6`](https://github.com/ggml-org/llama.cpp/commit/7d9b931df9bb7c689f067a7fed23e7f5253951f6) 的源码编译。已有系统级 CUDA 工具链时，无需额外安装；本次把 NVIDIA Debian 12 官方包解到远程任务目录，未切换驱动或系统 CUDA。

依赖：CMake 3.31.10、Ninja、GCC 12.2、CUDA NVCC 12.6.85、CUDA 12.6 runtime、cuBLAS 12.6.4.1，以及 NVML、CUDA driver、CCCL 开发头文件。`CUDA_ROOT` 指向 CUDA 12.6 根目录。

```bash
cmake -S /path/to/llama.cpp -B build-cu126 -G Ninja \
  -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DGGML_CUDA_GRAPHS=OFF -DGGML_NATIVE=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_OPENSSL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cu126 --parallel 16 --target llama-server
export BIN="$PWD/build-cu126/bin/llama-server"
export LD_LIBRARY_PATH="$CUDA_ROOT/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$BIN" --list-devices
```

设备枚举不负责验证运算。用 `run-depth.sh` 检查真实请求，日志应同时出现循环元数据覆盖、`n_layer_all = 12 + 12 × 圈数` 和 CUDA 层卸载。GPU 不可用时可以设 `GPU_LAYERS=0` 完全关闭 GPU 路径做冒烟测试；CPU 与 GPU 的输出与耗时应分别记录。

现场记录：发布包 `b12607`（`1fdda89a4`）直接在此驱动上运行报 `device kernel image is invalid`；临时加 CUDA 13 兼容库后枚举仍成功，但首次 cuBLAS matmul 失败。正式测量使用上面给出的 CUDA 12.6 构建。已验证的 CPU 冒烟路径使用该发布包的 `--device none --no-op-offload`，只证明基本推理路径，不计入 GPU 结果。
