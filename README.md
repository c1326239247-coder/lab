# Cache-Bridging: Hierarchical Visual KV Injection into a Frozen LLM

Reference implementation and full experimental record for the paper
*Cache-Bridging: Hierarchical Visual KV Injection into a Frozen Language Model*
(Luyang Ji et al., Shaoxing University).

Instead of compressing an image into a handful of soft prompt tokens, the
vision encoder is read at four depths to form a hierarchical key-value cache,
a small learned bridge lifts that cache into the language model's own per-head
KV space, and the result is spliced into every decoder layer as a cache prefix.
Both backbones stay frozen; only the bridge (825,507 parameters, 0.059% of the
1.40B frozen backbone) is trained.

**Headline result, reported honestly.** Under an identical training budget the
bridge does *not* beat the much simpler caption-style projection connector it
was designed to replace (GQA-spatial 19.50% against 34.25%). Visual information
does reach the language model through the KV pathway - accuracy falls to 14.00%
with mismatched images and 15.25% with the cache disabled - and the shortfall
is not a capacity effect, since a 2.39x larger bridge scores identically. Every
number in the paper can be regenerated from this repository.

## Repository layout

| Path | Paper section | Contents |
| --- | --- | --- |
| `cache_bridge_torch/` | 3.2, 3.3 | PyTorch bridge modules: low-rank KV projection, 2D-geometry-preserving attention, adaptive gate, prefix/interleaved injectors, unit tests |
| `cache_bridge/` | Appendix C | Dependency-free NumPy reference implementation and its module-level smoke tests |
| `experiments/` | 4 | Every script that produced a number in the paper |
| `results/` | 4, Appendices A-B | Raw per-run JSON: configuration, hardware report, host KV statistics, accuracies under all control conditions, sample predictions |
| `paper_figures/` | Figures 5-6 | Figures regenerated from `results/` by `experiments/report.py` |
| `rewrite/` | - | Scripts that build the manuscript, reading all numbers from `results/` so no figure in the text can drift from the data |

### `experiments/`

| File | Role |
| --- | --- |
| `common.py` | Environment setup, dataset loaders (VQAv2, OK-VQA, GQA-spatial), official VQA accuracy metric, seeding, device report |
| `model.py` | Both systems: the cache-bridging connector and the caption-style projection baseline |
| `run_vqa.py` | Train and evaluate one configuration end to end; writes one `results/run_<tag>.json` |
| `run_main.py` | Main comparison: 3 datasets x 2 connectors, 1500 steps |
| `run_seeds.py` | Seed repeats of the GQA-spatial comparison |
| `run_ablations.py` | 13 single-component ablations, 800 steps each |
| `run_cost.py` | Inference latency and peak GPU memory, 60 questions per variant |
| `backbone_params.py` | Frozen-parameter accounting for CLIP and TinyLlama |
| `summarize.py`, `report.py` | Build the paper tables and figures from `results/` |
| `smoke.py` | Fast end-to-end sanity check before a long run |

## Setup

Python 3.11 and one CUDA GPU with >=12 GB. The reported runs used an
NVIDIA GeForce RTX 5070 Ti (15.9 GB, compute capability 12.0) with PyTorch
2.11.0+cu128.

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

```powershell
# one configuration (~10 min on one RTX 5070 Ti)
.\.venv\Scripts\python.exe experiments\run_vqa.py --dataset GQA-spatial --system bridge `
    --train-size 3000 --eval-size 400 --steps 1500 --batch-size 8 --lr 1e-3 --tag GQA_bridge

# Tables 2-3, Figure 5: main comparison
.\.venv\Scripts\python.exe experiments\run_main.py

# Section 4.7: seed repeats
.\.venv\Scripts\python.exe experiments\run_seeds.py

# Table 5, Figure 6: ablations
.\.venv\Scripts\python.exe experiments\run_ablations.py

# Table 4: inference cost
.\.venv\Scripts\python.exe experiments\run_cost.py --repeats 60

# Section 4.6: capacity control
.\.venv\Scripts\python.exe experiments\run_vqa.py --dataset GQA-spatial --system bridge `
    --rank 512 --n-visual 64 --steps 1500 --tag GQA_bridge_big

# tables and figures from whatever is in results/
.\.venv\Scripts\python.exe experiments\summarize.py
.\.venv\Scripts\python.exe experiments\report.py
```

The whole set is about 1.8 GPU-hours on one RTX 5070 Ti.

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

Four details cost real debugging time and are discussed in Section 5 of the paper:

- **Keys and values must be normalised separately.** The host cache has
  k_std 2.29 and v_std 0.22; normalising the two jointly destroys the visual
  signal completely.
- **Position ids must be `cumsum(mask) - 1`**, not a plain `arange`, or
  variable-length batches get positions that do not match the tokens.
- **Prompts must be left-padded at generation time**, otherwise the model
  continues from a pad token.
- **The EOS token is down-weighted** during training so the model does not
  learn to answer with an immediate stop.

## Data

- VQAv2 (`merve/vqav2-small`), OK-VQA (`lmms-lab/OK-VQA`), GQA-spatial
  (spatial-question subset of GQA), all public.
- 3000 training and 400 held-out questions per dataset; the held-out split is
  never used for tuning.

## Citation

```bibtex
@article{ji2026cachebridging,
  title  = {Cache-Bridging: Hierarchical Visual KV Injection into a Frozen Language Model},
  author = {Ji, Luyang and others},
  year   = {2026}
}
```

## License

MIT, see `LICENSE`. The CLIP and TinyLlama checkpoints and the three datasets
remain under their own licenses.
