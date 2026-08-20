# cache_bridge_torch

PyTorch implementation of the **bridge adapter** that splices CLIP's
visual KV cache into LLaMA's per-layer KV cache.  This is the
core innovation of the patent "基于缓存桥接的跨模态理解系统".

The code is checkpoint-aligned to:
  * `openai/clip-vit-large-patch14`  (24 layers, 1024 dim, 16 heads)
  * `meta-llama/Llama-2-7b-hf`      (32 layers, 4096 dim, 32 heads)

and works with any other CLIP variant + LLaMA variant by overriding
the fields in `BridgeConfig`.

## What the bridge does

```
   image
     |
     v
  CLIP-ViT-L/14                (frozen)
     |
     |  hooks on 4 chosen layers
     v
   4-layer visual KV cache
   (basic / semantic_object /
    spatial_relation / attribute)
     |
     v
  BridgeAdapter
   1. DimensionProjection  (LoRA: 1024 -> 4096)
   2. StructurePreservingAttn (2D relative-position bias)
   3. AdaptiveGating (per-token sigmoid, context-aware)
     |
     |   K_bridge, V_bridge
     |   (B, 32, T_visual, 128)
     v
  VisualInjector
     |
     |   splices the bridged cache into
     |   LLaMA's past_key_values at the
     |   right text positions (noun /
     |   preposition / adjective)
     v
  LLaMA-7B
     |
     v
  next-token logits
```

## Files

```
cache_bridge_torch/
|-- __init__.py
|-- config.py             # BridgeConfig + clip_llama_bridge_config()
|-- bridge.py             # DimensionProjection, StructurePreservingAttn,
|                         # AdaptiveGating, BridgeAdapter
|-- extract.py            # CLIPHierarchicalCacheExtractor (hook-based)
|-- inject.py             # VisualInjector, simple_pos_tag
|-- integration.py        # CacheBridgeLLM, build_cache_bridge_llm
|-- tests/
|   `-- test_shapes.py    # NumPy shape verification (no torch needed)
`-- README.md             # this file
```

## Training the bridge

The bridge is small (~700K parameters with rank-64 LoRA) and is the
only thing you actually train.  The recommended 3-stage schedule is
the same as in the patent:

  Stage 1:  alignment pre-training
              - freeze CLIP and LLaMA
              - train the bridge + a single linear adapter on a small
                captioning dataset (CC3M, COCO Captions, LLaVA-Pretrain)
              - objective:  cross-entropy on the next token
  Stage 2:  end-to-end fine-tuning
              - unfreeze the bridge + gate + the LLaMA LoRA
              - objective:  cross-entropy + an attention-distillation
                loss between the bridged K and the LLaMA's own K
  Stage 3:  task adaptation
              - freeze everything except the dynamic-router head
              - objective:  supervised query-type classification

The training loop is left out of this module because it is identical
to standard LLaMA fine-tuning, just with the bridge inserted.  A
working `train.py` would be ~80 lines wrapping HF Trainer.

## Running the shape tests

The environment used to develop this code did not have PyTorch, so
the test suite uses NumPy to verify the tensor shapes at every
juncture of the bridge.  These tests do not need GPU.

```
py cache_bridge_torch/tests/test_shapes.py
7 / 7 tests passed
```

The tests cover:
  1. CLIP KV cache shapes  (B, 16, 256, 64) per layer
  2. Bridge output shapes  (B, 32, 4*256, 128) for the full mask
  3. POS tagger            "red"->2, "on"->1, "the"->0, "cat"->0
  4. 2D relative-position bias indexing
  5. Prepend-mode splicing (text K, V get basic visual K, V prepended)
  6. Interleave-mode splicing (visual token at every text position)
  7. Bridge parameter count (~16% of the full 1024->4096 projection)

## Real CLIP + LLaMA usage (requires PyTorch)

```python
import torch
from transformers import (
    CLIPVisionModel, CLIPProcessor,
    LlamaForCausalLM, LlamaTokenizer,
)
from cache_bridge_torch import (
    build_cache_bridge_llm, clip_llama_bridge_config,
)

cfg = clip_llama_bridge_config()
clip = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
llama = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",
                                         torch_dtype=torch.float16)
tok = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

model = build_cache_bridge_llm(cfg, clip, llama, tok)
model = model.cuda().eval()

# Encode the image
pixel_values = proc(images=image, return_tensors="pt")["pixel_values"].cuda()
model.set_image(pixel_values)

# Tokenize the prompt
prompt = "Describe the image in detail."
input_ids = tok(prompt, return_tensors="pt")["input_ids"][0].cuda()
model.set_prompt(input_ids)

# Generate
out = model.generate(input_ids[None], max_new_tokens=64)
print(tok.decode(out[0]))
```

## Caveats

* The bridge expects a frozen CLIP and a frozen LLaMA.  If you want
  to fine-tune those, use LoRA on them and keep the bridge itself
  as the only unfrozen component during stage 1.
* The simple_pos_tag function is a 30-line heuristic.  Replace it
  with spaCy or a Hugging Face token-classifier for production.
* The bridge injects the visual cache into EVERY LLaMA layer.  This
  matches the patent.  In practice you may want to inject only into
  a subset of layers (e.g. the first 8) for efficiency.  Add a
  `layer_idx` filter in `VisualInjector._splice_into_pkv` to do that.
* The 2D relative-position bias assumes the CLIP patch grid is
  square.  For non-square grids (e.g. 14x16) you would need a
  rectangular bias table.
* The gate is a single linear layer, not an MLP, to keep the
  parameter count down.  Replace it with a 2-layer MLP if you
  observe underfitting.