# -*- coding: utf-8 -*-
"""Generate README.md from the same numbers the manuscript uses.

The README quoted hand-copied accuracies before, which drifted as soon as
the experiments were rerun.  It is now generated from results/*.json by
the same module that feeds the manuscript, so the two cannot disagree.
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numbers as facts_mod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEMPLATE = """# Cache-Bridging: Hierarchical Visual KV Injection into a Frozen LLM

Reference implementation and full experimental record for the paper
*Cache bridging of hierarchical visual key values does not consistently improve
visual question answering with a frozen language model*
(Luyang Ji et al., Shaoxing University).

Instead of compressing an image into a handful of soft prompt tokens, the
vision encoder is read at four depths to form a hierarchical key-value cache,
a small learned bridge lifts that cache into the language model's own per-head
KV space, and the result is spliced into every decoder layer as a cache prefix.
Both backbones stay frozen; only the bridge ({small_params} parameters,
{bridge_pct} of the {total_backbone} frozen backbone) is trained.

**Headline result, reported honestly.** Under an identical training budget the
bridge does not establish a consistent advantage over the much simpler
caption-style projection connector or a question-only control. It approaches
the caption baseline on GQA-spatial but remains behind on VQAv2 and OK-VQA:

| System | Trainable params | VQAv2 | OK-VQA | GQA-spatial |
| --- | --- | --- | --- | --- |
| Question only (blind) | {blind_params} | {vqa_blind} | {ok_blind} | {gqa_blind} |
| Caption-style projection | {caption_params} | {vqa_caption} | {ok_caption} | {gqa_caption} |
| Cache-bridging | {small_params} | {vqa_bridge} | {ok_bridge} | {gqa_bridge} |

Accuracies are measured on {eval_size} held-out questions per dataset; 95%
bootstrap intervals are about +/-{ci_half_typical} points. OK-VQA uses its ten
human references with the VQA-style soft score. The VQAv2 source used here has
only its majority answer and GQA has one reference, so those two scores are
normalised exact match rather than full ten-annotator VQA soft accuracy.

Visual information does reach the language model through the KV pathway -
pairing every question with the wrong image costs {delta_gqa_shuf} on
GQA-spatial - so this is not the trivial failure of a connector that transmits
nothing. The revised controls and diagnostics include:

- **Capacity affects the outcome.** A sweep over {cap_range} trainable
  parameters brackets the baseline's {caption_params}; the best bridge is
  {cap_best} at {cap_best_params}, and the largest point over the baseline's
  budget reaches {cap_largest}. The best row is exploratory and not
  seed-repeated.
- **The injected prefix dominates attention.** Text tokens assign {probe_mass}
  of their attention probability to the injected KV prefix against
  {probe_caption_mass} for the baseline's embedding-space prefix.
- **Matched prefix comparison.** {prefix_sentence}.
- **Joint key/value scaling.** {kv_sentence}. This control also changes value
  magnitudes, so it does not isolate attention crowding.
- **Larger-decoder sensitivity check.** {mistral_sentence}
- **Sharing one cache across all layers is not the whole story.** Giving each
  of the {llm_layers} decoder layers its own learned K/V scale changes accuracy
  by {abl_pls_delta} points.

**On reproducibility.** The repeated TinyLlama configurations are deterministic.
Reaching that required
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
| `paper_figures/` | Figures 1, 5-9 | Reviewer-labelled Figure 1 plus figures regenerated from `results/` by `experiments/report.py` |
| `rewrite/` | - | Scripts that build the manuscript and this README, reading all numbers from `results/` so no figure in the text can drift from the data |

### `experiments/`

| File | Role |
| --- | --- |
| `common.py` | Environment setup, dataset loaders (VQAv2, OK-VQA, GQA-spatial), VQA-style scoring, bootstrap and paired-bootstrap intervals, deterministic seeding, device report |
| `model.py` | All three systems: the cache-bridging connector, the caption-style projection baseline, and the question-only blind control |
| `run_vqa.py` | Train and evaluate one configuration end to end; writes one `results/run_<tag>.json` and one checkpoint |
| `run_all.py` | Runs every stage below in order, skipping stages whose results already exist |
| `run_main.py` | Main comparison: 3 datasets x 3 systems, {steps} steps |
| `run_seeds.py` | Seed repeats of the GQA-spatial comparison, both systems |
| `run_ablations.py` | Single-component ablations, {abl_steps} steps each |
| `run_capacity.py` | Capacity sweep from {cap_range} trainable parameters |
| `run_prefix_sweep.py` | Matched total-prefix-length comparison |
| `run_kv_scale_sweep.py` | Direct K/V attention-allocation sweep |
| `run_mistral.py` | Larger-decoder blind/caption/bridge sensitivity check |
| `run_bridge_revision.py` | Rerun the affected results and finish the corrected comparison record |
| `compare_systems.py` | Paired bootstrap differences between systems on the same questions |
| `audit_splits.py` | Image-overlap audit for the question-level train/evaluation split |
| `verify_references.py` | Checks every bibliography entry against arXiv, Crossref and OpenAlex |
| `run_cost.py` | Inference latency and peak GPU memory per variant |
| `probe.py` | Diagnostics on a trained checkpoint: per-layer attention mass on the visual prefix, per-layer host/bridged KV statistics |
| `backbone_params.py` | Frozen-parameter accounting for CLIP and TinyLlama |
| `summarize.py`, `report.py` | Build the paper tables and figures from `results/` |
| `smoke.py` | Fast end-to-end sanity check before a long run |

## Setup

Python 3.11 and one CUDA GPU with >=12 GB. The reported runs used an
{gpu} ({vram} GB, compute capability {cc}) with PyTorch {torch_ver} and
CUDA {cuda_ver}.

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install --upgrade pip
.\\.venv\\Scripts\\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
```

Checkpoints (`openai/clip-vit-large-patch14`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
and the three datasets are downloaded from Hugging Face on first use into `.hf/`.
Behind a slow connection set `HF_ENDPOINT` to a mirror; `common.setup_env()`
already defaults to one.

The larger-decoder check uses `unsloth/mistral-7b-instruct-v0.2-bnb-4bit`
in `models/mistral-7b-instruct-v0.2-bnb-4bit`. Run bridge with
`--llm-id models/mistral-7b-instruct-v0.2-bnb-4bit`; the caption and blind
controls use the same decoder and training protocol.

## Reproducing the paper

One command runs everything and skips whatever is already in `results/`:

```powershell
.\\.venv\\Scripts\\python.exe experiments\\run_all.py
```

The recorded training and evaluation passes account for about
{total_gpu_hours} GPU-hours on one {gpu}. Attention probes, the separate cost
measurement, downloads, and document generation are additional. The individual
stages, if you want them separately:

```powershell
# Tables 2-3, Figure 5: main comparison (3 datasets x blind/caption/bridge)
.\\.venv\\Scripts\\python.exe experiments\\run_main.py

# Section 4.2: seed repeats for both systems
.\\.venv\\Scripts\\python.exe experiments\\run_seeds.py

# Table 5, Figure 6: ablations, including the matched-parameter gate control
# and the per-layer-scale variant
.\\.venv\\Scripts\\python.exe experiments\\run_ablations.py

# Table 6, Figure 7: capacity sweep
.\\.venv\\Scripts\\python.exe experiments\\run_capacity.py

# Matched prefix-length and K/V attention-allocation sweeps
.\\.venv\\Scripts\\python.exe experiments\\run_prefix_sweep.py
.\\.venv\\Scripts\\python.exe experiments\\run_kv_scale_sweep.py

# Table 4: inference cost
.\\.venv\\Scripts\\python.exe experiments\\run_cost.py --repeats 60

# Paired differences between caption, bridge, and the controls
.\\.venv\\Scripts\\python.exe experiments\\compare_systems.py

# Table 7, Figures 8-9: attention-mass and KV-scale diagnostics
.\\.venv\\Scripts\\python.exe experiments\\probe.py --checkpoint checkpoints/GQA_bridge.pt
.\\.venv\\Scripts\\python.exe experiments\\probe.py --checkpoint checkpoints/GQA_caption.pt

# tables and figures from whatever is in results/
.\\.venv\\Scripts\\python.exe experiments\\summarize.py
.\\.venv\\Scripts\\python.exe experiments\\report.py

# bibliography audit against arXiv, Crossref and OpenAlex (needs network)
.\\.venv\\Scripts\\python.exe experiments\\verify_references.py

# manuscript and reviewer response in dist/, then regenerate this README
.\\.venv\\Scripts\\python.exe rewrite\\annotate_figure1.py
.\\.venv\\Scripts\\python.exe rewrite\\build_docx.py
.\\.venv\\Scripts\\python.exe rewrite\\build_response.py
.\\.venv\\Scripts\\python.exe rewrite\\build_readme.py
```

A single main configuration trains in about {bridge_train_s_min} minutes.

## Tests

No GPU and no checkpoint download required:

```powershell
.\\.venv\\Scripts\\python.exe cache_bridge_torch\\tests\\test_shapes.py      # {n_shape_tests} shape invariants, no PyTorch needed
.\\.venv\\Scripts\\python.exe cache_bridge_torch\\tests\\test_bridge_real.py # {n_torch_tests} forward/backward properties
.\\.venv\\Scripts\\python.exe cache_bridge_torch\\tests\\test_prompt_summary.py # {n_prompt_tests} prompt-summary regression checks
.\\.venv\\Scripts\\python.exe cache_bridge\\tests\\run_all.py                # {n_numpy_tests} NumPy reference tests
```

These verify the implementation only; none of the accuracy claims in the paper
rests on them.

## Implementation notes worth knowing

Discussed in Sections 3.5, 5.1 and 5.2 of the paper:

- **Keys and values must be normalised separately.** The host cache has
  k_std {k_std} and v_std {v_std}, about {kv_ratio}x apart; normalising the two
  jointly can destroy usable visual signal. Mismatched-image and disabled-cache
  controls make that failure visible.
- **Ablating the gate also removes its parameters.** `--no-gate` drops the
  bridge from {small_params} to {abl_gate_params} trainable parameters, so it
  confounds the mechanism with capacity. Use `--gate-mode fixed` for the
  matched-parameter control ({abl_gate_fixed_params} parameters, learned
  coefficient replaced by a fixed blend).
- **The low-rank projections are shared across the four hierarchy levels.** A
  level is distinguished by its tap depth and one scalar, not by its own
  weights.
- **The 2D bias table must be sized for the pooled grid.** Pooling to
  {n_visual} tokens gives a {pool_side}x{pool_side} grid, so offsets never
  exceed +/-{max_offset}; a table built for the raw {grid}x{grid} grid is mostly
  dead weight.
- **Position ids must be `cumsum(mask) - 1`**, not a plain `arange`, or
  variable-length batches get positions that do not match the tokens.
- **Prompts must be left-padded at generation time**, otherwise the model
  continues from a pad token.
- **The prompt summary must use the same masked mean in training and
  inference.** Only prompt tokens are averaged; gold answers and left padding
  are excluded.
- **The EOS token is down-weighted** during training so the model does not
  learn to answer with an immediate stop.

## Data

- VQAv2 (`merve/vqav2-small`), OK-VQA (`lmms-lab/OK-VQA`), GQA-spatial
  (relation-heavy subset of GQA testdev-balanced), all public and used as
  distributed.
- {train_size} training and {eval_size} held-out questions per dataset; the
  split is by question-answer pair, not by image. Capacity-best selection on the
  reported evaluation set is exploratory, not independent test confirmation.
  Bootstrap intervals resample questions, not image clusters.

## Citation

```bibtex
@article{{ji2026cachebridging,
  title  = {{Cache bridging of hierarchical visual key values does not
            consistently improve visual question answering with a frozen
            language model}},
  author = {{Ji, Luyang and others}},
  year   = {{2026}}
}}
```

The archived releases are deposited in Zenodo and citable through the concept
DOI [10.5281/zenodo.23974489](https://doi.org/10.5281/zenodo.23974489), which
always resolves to the latest version. Each release is the GitHub zip-ball of
its tag.

## License

MIT, see `LICENSE`. The CLIP and TinyLlama checkpoints and the three datasets
remain under their own licenses.
"""


def build():
    values, _ = facts_mod.facts()
    try:
        text = TEMPLATE.format(**values)
    except KeyError as exc:
        raise SystemExit("README placeholder not provided by numbers.py: %s"
                         % exc)
    path = os.path.join(ROOT, "README.md")
    io.open(path, "w", encoding="utf-8", newline="\n").write(text)
    return path


if __name__ == "__main__":
    print("written ->", build())
