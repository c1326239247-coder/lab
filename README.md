# Cache-Bridging Cross-Modal Understanding

基于缓存桥接的跨模态理解系统 (Cache-Bridging Cross-Modal Understanding System)。

将视觉层级 KV cache 通过可学习的 bridge adapter 注入语言模型，用无损的
"image -> KV -> LLM" 路径取代传统的 "image -> text -> LLM" 有损流程，
保留空间、语义与属性信息。

## 目录结构

- `cache_bridge/` - NumPy-only 实现（bridge adapter、hierarchical KV、visual encoder、language model、训练管线）
- `cache_bridge_torch/` - PyTorch 实现与端到端测试

## 快速开始

```bash
# NumPy 版本
python cache_bridge/tests/run_all.py

# PyTorch 版本
python cache_bridge_torch/tests/test_shapes.py
python cache_bridge_torch/tests/test_bridge_real.py
```

详见各子目录下的 `README.md`。
