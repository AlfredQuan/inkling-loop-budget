# Sources and licenses

- **AI2 Reasoning Challenge (ARC)**: Peter Clark, Isaac Cowhey, Oren Etzioni,
  Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord.
  [Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge](https://arxiv.org/abs/1803.05457)
  (2018). The [pinned dataset card](https://huggingface.co/datasets/allenai/ai2_arc/tree/210d026faf9955653af8916fad021475a3f00453)
  lists [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
  `m2/evidence/paired.jsonl.gz` and `m2/evidence/decisions.json` identify selected
  items by their original IDs. The paired evidence retains remapped answer keys,
  option predictions and projected activations. We distribute these ARC-derived
  files under CC BY-SA 4.0. We filter out non-four-option items and normalized
  duplicates, sample official splits, and reorder options. The repository does
  not ship question or option text.
- **Thinking Machines Lab's TML-d24-varloop-research**: obtain the model from
  the [original release](https://huggingface.co/thinkingmachines/TML-d24-varloop-research)
  or use the [pinned ggml-org conversion](https://huggingface.co/ggml-org/TML-d24-varloop-research-GGUF/tree/7af7f5e17b57395455bd8a37d8cfefa98824ca6e).
  The original model is Apache-2.0. No base checkpoint or GGUF is redistributed.
- **llama.cpp**: the native probe links against an external
  [pinned revision](https://github.com/ggml-org/llama.cpp/tree/7d9b931df9bb7c689f067a7fed23e7f5253951f6)
  under its upstream MIT terms. No copy of its code or binaries is bundled.

The repository's MIT license covers the project's own source, documentation,
M1 synthetic benchmark, and its own learned heads. Third-party terms still
apply to the data and dependencies listed above. This project is independent
of Thinking Machines Lab, Allen Institute for AI, and the llama.cpp project.
