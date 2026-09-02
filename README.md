# Cache-Bridging: Hierarchical Visual KV Injection into a Frozen LLM

Reference implementation and full experimental record for the paper
*Cache-Bridging: Injecting a Hierarchical Visual Key-Value Cache into a Frozen
Language Model - A Negative Result and a Diagnosis*
(Luyang Ji et al., Shaoxing University).

Instead of compressing an image into a handful of soft prompt tokens, the
vision encoder is read at four depths to form a hierarchical key-value cache,
a small learned bridge lifts that cache into the language model's own per-head
KV space, and the result is spliced into every decoder layer as a cache prefix.
Both backbones stay frozen; only the bridge (821,635 parameters,
0.059% of the 1,403,228,160 frozen backbone) is trained.

**Headline result, reported honestly.** Under an identical training budget the
bridge does *not* beat the much simpler caption-style projection connector it
was designed to replace, and on the relation-heavy GQA subset it does not beat
a question-only control that never sees the image:

| System | Trainable params | VQAv2 | OK-VQA | GQA-spatial |
| --- | --- | --- | --- | --- |
| Question only (blind) | 51,200 | 32.75% | 24.87% | 32.35% |
| Caption-style projection | 6,299,648 | 32.95% | 28.02% | 38.15% |
| Cache-bridging | 821,635 | 14.00% | 2.23% | 25.25% |

Accuracies are VQA accuracy on 2000 held-out questions per dataset;
95% bootstrap intervals are about +/-1.93 points.

Visual information does reach the language model through the KV pathway -
pairing every question with the wrong image costs +9.40 points (95% CI +7.40 to +11.40) on
GQA-spatial - so this is not the trivial failure of a connector that transmits
nothing. Three measurements locate the cause instead:

- **Capacity is not the constraint.** A sweep over 821,635 to 3,279,651 trainable
  parameters, bracketing the baseline's 6,299,648, moves accuracy by
  6.75 points.
- **The frozen decoder barely reads the injected prefix.** Text tokens assign
  0.966 of their attention probability to the injected KV prefix against
  0.279 for the baseline's embedding-space prefix.
- **Sharing one cache across all layers is not the whole story.** Giving each
  of the 22 decoder layers its own learned K/V scale changes accuracy
  by +1.55 points.

**On reproducibility.** All runs are deterministic. Reaching that required
replacing `F.adaptive_avg_pool2d` (non-deterministic CUDA backward) with an
explicit block mean and disabling the fused attention backward, both in the
released code. At a fixed seed on the same machine a configuration reproduces
its per-question scores exactly. Every number, table, and figure in the paper
is regenerated from `results/`.

## Repository layout

| Path | Paper section | Contents |
| --- | --- | --- |
| `cache_bridge_torch/` | 3.2, 3.3 | PyTorch bridge modules: low-rank KV projection, 2D-geometry-preserving attention, adaptive gate, prefix/interleaved injectors, unit tests |
| `cache_bridge/` | Appendix C | Dependency-free NumPy reference implementation and its module-level smoke tests |
| `experiments/` | 4 | Every script that produced a number in the paper |
| `results/` | 4, Appendices A-B | Raw per-run JSON: configuration, hardware report, host KV statistics, accuracies and bootstrap intervals under all control conditions, per-question scores, sample predictions |
| `paper_figures/` | Figures 5-9 | Figures regenerated from `results/` by `experiments/report.py` |
| `rewrite/` | - | Scripts that build the manuscript and this README, reading all numbers from `results/` so no figure in the text can drift from the data |

### `experiments/`

| File | Role |
| --- | --- |
| `common.py` | Environment setup, dataset loaders (VQAv2, OK-VQA, GQA-spatial), official VQA accuracy metric, bootstrap and paired-bootstrap intervals, deterministic seeding, device report |
| `model.py` | All three systems: the cache-bridging connector, the caption-style projection baseline, and the question-only blind control |
| `run_vqa.py` | Train and evaluate one configuration end to end; writes one `results/run_<tag>.json` and one checkpoint |
| `run_all.py` | Runs every stage below in order, skipping stages whose results already exist |
| `run_main.py` | Main comparison: 3 datasets x 3 systems, 1500 steps |
| `run_seeds.py` | Seed repeats of the GQA-spatial comparison, both systems |
| `run_ablations.py` | Single-component ablations, 800 steps each |
| `run_capacity.py` | Capacity sweep from 821,635 to 3,279,651 trainable parameters |
| `run_cost.py` | Inference latency and peak GPU memory per variant |
| `probe.py` | Diagnostics on a trained checkpoint: per-layer attention mass on the visual prefix, per-layer host/bridged KV statistics |
| `backbone_params.py` | Frozen-parameter accounting for CLIP and TinyLlama |
| `summarize.py`, `report.py` | Build the paper tables and figures from `results/` |
| `smoke.py` | Fast end-to-end sanity check before a long run |

## Setup

Python 3.11 and one CUDA GPU with >=12 GB. The reported runs used an
NVIDIA GeForce RTX 5070 Ti (15.9 GB, compute capability 12.0) with PyTorch 2.11.0+cu128 and
CUDA 12.8.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Checkpoints (`openai/clip-vit-large-patch14`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
and the three datasets are downloaded from Hugging Face on first use into `.hf/`.
Behind a slow connection set `HF_ENDPOINT` to a mirror; `common.setup_env()`
already defaults to one.

## Reproducing the paper

One command runs everything and skips whatever is already in `results/`:

```powershell
.\.venv\Scripts\python.exe experiments\run_all.py
```

That is about 3.3 GPU-hours on one NVIDIA GeForce RTX 5070 Ti. The individual stages,
if you want them separately:

```powershell
# Tables 2-3, Figure 5: main comparison (3 datasets x blind/caption/bridge)
.\.venv\Scripts\python.exe experiments\run_main.py

# Section 4.2: seed repeats for both systems
.\.venv\Scripts\python.exe experiments\run_seeds.py

# Table 5, Figure 6: ablations, including the matched-parameter gate control
# and the per-layer-scale variant
.\.venv\Scripts\python.exe experiments\run_ablations.py

# Table 6, Figure 7: capacity sweep
.\.venv\Scripts\python.exe experiments\run_capacity.py

# Table 4: inference cost
.\.venv\Scripts\python.exe experiments\run_cost.py --repeats 60

# Table 7, Figures 8-9: attention-mass and KV-scale diagnostics
.\.venv\Scripts\python.exe experiments\probe.py --checkpoint checkpoints/GQA_bridge.pt
.\.venv\Scripts\python.exe experiments\probe.py --checkpoint checkpoints/GQA_caption.pt

# tables and figures from whatever is in results/
.\.venv\Scripts\python.exe experiments\summarize.py
.\.venv\Scripts\python.exe experiments\report.py
```

A single main configuration trains in about 5 minutes.

## Tests

No GPU and no checkpoint download required:

```powershell
.\.venv\Scripts\python.exe cache_bridge_torch\tests\test_shapes.py      # 7 shape invariants, no PyTorch needed
.\.venv\Scripts\python.exe cache_bridge_torch\tests\test_bridge_real.py # 10 forward/backward properties
.\.venv\Scripts\python.exe cache_bridge\tests\run_all.py                # 9 NumPy reference tests
```

These verify the implementation only; none of the accuracy claims in the paper
rests on them.

## Implementation notes worth knowing

Discussed in Sections 3.5, 5.1 and 5.2 of the paper:

- **Keys and values must be normalised separately.** The host cache has
  k_std 2.31 and v_std 0.21, about 11x apart; normalising the two
  jointly destroys the visual signal completely and the failure is invisible
  without the mismatched-image control.
- **Ablating the gate also removes its parameters.** `--no-gate` drops the
  bridge from 821,635 to 296,812 trainable parameters, so it
  confounds the mechanism with capacity. Use `--gate-mode fixed` for the
  matched-parameter control (821,100 parameters, learned
  coefficient replaced by a fixed blend).
- **The low-rank projections are shared across the four hierarchy levels.** A
  level is distinguished by its tap depth and one scalar, not by its own
  weights.
- **The 2D bias table must be sized for the pooled grid.** Pooling to
  36 tokens gives a 6x6 grid, so offsets never
  exceed +/-5; a table built for the raw 16x16 grid is mostly
  dead weight.
- **Position ids must be `cumsum(mask) - 1`**, not a plain `arange`, or
  variable-length batches get positions that do not match the tokens.
- **Prompts must be left-padded at generation time**, otherwise the model
  continues from a pad token.
- **The EOS token is down-weighted** during training so the model does not
  learn to answer with an immediate stop.

## Data

- VQAv2 (`merve/vqav2-small`), OK-VQA (`lmms-lab/OK-VQA`), GQA-spatial
  (relation-heavy subset of GQA testdev-balanced), all public and used as
  distributed.
- 3000 training and 2000 held-out questions per dataset; the
  held-out split is never used for tuning.

## Citation

```bibtex
@article{ji2026cachebridging,
  title  = {Cache-Bridging: Injecting a Hierarchical Visual Key-Value Cache
            into a Frozen Language Model - A Negative Result and a Diagnosis},
  author = {Ji, Luyang and others},
  year   = {2026}
}
```

## License

MIT, see `LICENSE`. The CLIP and TinyLlama checkpoints and the three datasets
remain under their own licenses.
