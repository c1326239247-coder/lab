# Cache-Bridging Cross-Modal Understanding System

A NumPy-only reproduction of the architecture described in the patent
"基于缓存桥接的跨模态理解系统" (cache-bridging cross-modal
understanding system).  The system routes a hierarchical visual
key-value cache through a learned bridge adapter into a language
model, replacing the lossy "image → text → LLM" pipeline with a
lossless "image → KV → LLM" pipeline that preserves spatial,
semantic, and attribute information.

## Why this is interesting

Traditional multimodal systems serialize the vision and language
pipelines: a vision model first produces a textual description of the
image, and a language model then consumes that description.  This
discards the spatial, object-level, and attribute structure that the
vision model spent compute to build.

This project keeps the entire visual KV cache alive and injects it
into the language model at the right positions, gated by a learned
controller, so the language model can attend to the original visual
structure rather than a lossy description.

## Architecture

```
                    image
                      |
                      v
        +-------------------------+
        |     VisualEncoder       |    small ViT-style backbone
        |  (patch tokens + 2 tx)  |
        +-------------------------+
                      |
                      v
        +-------------------------+
        |  HierarchicalKVCache    |    4 layers:
        |   basic, semantic_obj,  |     * basic            (8)
        |   spatial_relation,     |     * semantic_object  (6)
        |   attribute             |     * spatial_relation (4)
        +-------------------------+     * attribute         (6)
                      |
                      v
        +-------------------------+
        |  CompressionModule      |    optional Top-K + similarity merge
        +-------------------------+
                      |
                      v
        +-------------------------+
        |  DynamicRouter          |    picks which layers to inject
        |  (per query type)       |    per text position injection slot
        +-------------------------+
                      |
                      v
        +-------------------------+
        |  BridgeAdapter          |    3 sub-modules:
        |   - DimensionProjection |      lift to language model dim
        |   - StructurePreserving |      inject spatial-relative bias
        |   - AdaptiveGating      |      learn when to use the cache
        +-------------------------+
                      |
                K, V
                      v
        +-------------------------+
        |     LanguageModel       |    GPT-style decoder with
        |  (causal self-attention  |    cross-attention to the
        |   + cross-attn to cache)|    injected K, V
        +-------------------------+
                      |
                      v
                  next-token logits
```

The four cache layers (basic, semantic_object, spatial_relation,
attribute) match the four-level decomposition listed in the manual:

* **basic** -- pixel textures, local edges (early-layer features)
* **semantic_object** -- object classes, instance bounding boxes
* **spatial_relation** -- object-to-object positional relations
* **attribute** -- colour, material, state (global attribute features)

The **DynamicRouter** decides which layers to inject for the current
query:

* `detailed_description` → all four layers
* `spatial_reasoning` → `semantic_object` + `spatial_relation`
* `subjective` → `attribute` only

The cross-attention in the language model uses a different scale
factor at each text position depending on the position-type tag
(noun, preposition, adjective) of the prompt token, which lets the
visual cache land precisely where the language model needs it.

## Project layout

```
cache_bridge/
├── __init__.py
├── config.py              ModelConfig namedtuple, three presets
├── hierarchical_kv.py     4-layer (basic, sem_obj, spatial, attr) cache
├── visual_encoder.py      ViT-style encoder + 4 layer heads
├── bridge_adapter.py      DimensionProjection + StructurePreservingAttn
│                          + AdaptiveGating
├── language_model.py      GPT-style decoder with cross-attention
├── dynamic_router.py      Query-type classifier, per-position scales
├── compression.py         Top-K + similarity-merge compressor
├── data.py                Byte-level synthetic multimodal dataset
├── pipeline.py            End-to-end image -> logits wrapper
├── training.py            3-stage training with finite-difference grad
├── train.py               CLI: --demo | --tiny | --full
└── tests/
    └── run_all.py         Smoke tests for every module
```

## Running

The project is intentionally NumPy-only so it runs on any machine
that has at least NumPy 1.9.  No GPU, no PyTorch, no network access
required.

```bash
# 1) Forward-pass demo, no training
py -m cache_bridge.train --demo --examples 4

# 2) Tiny training run (CPU friendly)
py -m cache_bridge.train --tiny --stage1 4 --stage2 4 --stage3 4 --examples 6

# 3) Bigger forward-pass demo
py -m cache_bridge.train --full --demo --examples 4

# 4) Run the unit tests
py cache_bridge/tests/run_all.py
```

The output of `--demo` looks like:

```
--- Example 0 ---
  object  : ball
  type    : spatial_reasoning
  query   : where is the ball located
  response: the ball appears in the upper right corner
  layer mask (basic, obj, rel, attr): [1, 0, 0, 1]
  pos weights (noun, prep, adj)    : [0.331 0.335 0.335]
  gate bias                         : 0.5
  forward time                      : 0.014 s
  logits shape                      : (1, 25, 256)
  greedy argmax                     : ...
```

The layer mask `[1, 0, 0, 1]` is the router's output for a
`spatial_reasoning` query, which should normally favour the
`semantic_object` and `spatial_relation` layers.  With an untrained
model the router is essentially random -- the model needs training
to produce meaningful masks.  Use `--tiny --stage1 20 --stage2 20
--stage3 20` for a few hundred iterations of training to see the
router lock onto the correct query types.

## Three-stage training

The training loop follows the manual's three-stage schedule:

1. **Alignment pre-training** -- freeze everything except the bridge
   adapter, train the adapter with a cross-entropy loss on the
   next-token prediction.  This teaches the adapter how to project
   visual caches into the language model's input space.
2. **End-to-end fine-tuning** -- unfreeze the bridge adapter and the
   adaptive gating; train the joint system end-to-end with
   cross-entropy.
3. **Task adaptation** -- freeze the bridge adapter, train only the
   dynamic router with a supervised loss matching the query-type
   label.

The default training uses finite-difference gradients because the
project is NumPy-only.  For a real research run you would replace
`cache_bridge/training.py` with a PyTorch / JAX implementation of
the same loss functions; the architecture stays the same.

## Limitations of the reproduction

* The visual encoder is intentionally small (a 2-block ViT on 32×32
  images with patch size 8).  It is enough to exercise the pipeline
  end-to-end but obviously does not match the quality of a
  production CLIP / ViT-L.
* The language model is a 1-2 layer decoder; production models are
  7B+ parameters.
* The byte-level vocabulary is used so we do not depend on a real
  tokenizer.  Replace `cache_bridge/data.py` with a Hugging Face
  tokenizer to use a real LLM.
* The training loop uses finite-difference gradients, which is
  correct but very slow.  Replace it with autograd for real work.
* No image I/O.  Replace the synthetic dataset with a real one
  (COCO, Visual Genome, etc.) by subclassing
  `cache_bridge.data.SyntheticMultimodalDataset`.

## License

This is a research reproduction of a published patent description.
It is provided as-is for educational use.