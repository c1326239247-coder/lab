# -*- coding: utf-8 -*-
"""Shared experiment utilities: environment, data loading, VQA scoring.

All experiments in this directory run on a single consumer GPU
(NVIDIA GeForce RTX 5070 Ti, 16 GB) with CLIP-ViT-L/14 as the frozen
vision encoder and TinyLlama-1.1B-Chat as the frozen language model.
"""

import io
import json
import os
import random
import re
import time

import numpy as np
import torch

CLIP_ID = "openai/clip-vit-large-patch14"
LLM_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


def setup_env():
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", os.path.join(REPO_ROOT, ".hf"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def set_seed(seed, deterministic=True):
    """Seed every RNG and, by default, force deterministic kernels.

    Without this the 2D pooling in the bridge has a non-deterministic CUDA
    backward and an identical configuration does not reproduce; with it the
    same seed gives the same number on the same machine.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # The fused attention kernels have a non-deterministic backward.
        # Sequences here are short (a few dozen tokens), so the math
        # kernel costs little and makes training bit-for-bit repeatable.
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def device_report():
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    return {
        "gpu": name,
        "compute_capability": "%d.%d" % cap,
        "vram_gb": round(total, 1),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


# ---------------------------------------------------------------- data

def _pil_from_bytes(raw):
    from PIL import Image
    return Image.open(io.BytesIO(raw)).convert("RGB")


def load_okvqa(limit):
    """Real OK-VQA validation split (lmms-lab/OK-VQA)."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    rows = []
    for shard in ["data/val2014-00000-of-00002.parquet",
                  "data/val2014-00001-of-00002.parquet"]:
        path = hf_hub_download("lmms-lab/OK-VQA", shard, repo_type="dataset")
        table = pq.read_table(path, columns=["question", "answers", "image"])
        data = table.to_pylist()
        for record in data:
            rows.append({
                "question": record["question"],
                "answers": list(record["answers"]),
                "image": record["image"]["bytes"],
            })
        if len(rows) >= limit:
            break
    random.Random(0).shuffle(rows)
    return rows[:limit]


def load_vqav2(limit):
    """Real VQAv2 validation subset (merve/vqav2-small).

    The split is sharded; shards are pulled in order until `limit` rows
    are available, so the evaluation set can be enlarged without changing
    the sampling procedure. This source provides only the majority answer,
    so the VQA-style score below reduces to normalised exact match.
    """
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    rows = []
    for shard in range(7):
        name = "data/validation-0000%d-of-00007.parquet" % shard
        try:
            path = hf_hub_download("merve/vqav2-small", name,
                                   repo_type="dataset")
        except Exception:
            # Shards beyond the ones already fetched are optional; stop
            # early rather than fail when the hub is unreachable.
            break
        table = pq.read_table(
            path, columns=["question", "multiple_choice_answer", "image"])
        for record in table.to_pylist():
            rows.append({
                "question": record["question"],
                "answers": [record["multiple_choice_answer"]] * 10,
                "image": record["image"]["bytes"],
            })
        if len(rows) >= limit:
            break
    random.Random(0).shuffle(rows)
    return rows[:limit]


def load_gqa(limit, spatial_only=True):
    """Real GQA testdev-balanced split (lmms-lab/GQA).

    GQA ships questions and images in separate parquet files, joined on
    the image id.  When `spatial_only` is set we keep only the questions
    whose semantic annotation involves a relational or positional step,
    which is the spatial subset used for the relation analysis.
    """
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    q_path = hf_hub_download(
        "lmms-lab/GQA",
        "testdev_balanced_instructions/testdev-00000-of-00001.parquet",
        repo_type="dataset")
    i_path = hf_hub_download(
        "lmms-lab/GQA",
        "testdev_balanced_images/testdev-00000-of-00001.parquet",
        repo_type="dataset")

    images = {}
    for record in pq.read_table(i_path).to_pylist():
        images[record["id"]] = record["image"]["bytes"]

    questions = pq.read_table(
        q_path, columns=["imageId", "question", "answer", "types", "semanticStr"]
    ).to_pylist()

    rows = []
    for record in questions:
        img = images.get(record["imageId"])
        if img is None:
            continue
        if spatial_only:
            types = record.get("types") or {}
            semantic = (record.get("semanticStr") or "").lower()
            detailed = str(types.get("detailed", "")).lower()
            is_spatial = ("relation" in detailed or "rel" in semantic
                          or "position" in semantic)
            if not is_spatial:
                continue
        rows.append({
            "question": record["question"],
            "answers": [record["answer"]] * 10,
            "image": img,
        })
    random.Random(0).shuffle(rows)
    return rows[:limit]


DATASETS = {
    "VQAv2": load_vqav2,
    "OK-VQA": load_okvqa,
    "GQA-spatial": load_gqa,
}


# -------------------------------------------------------------- metric

_ARTICLES = {"a", "an", "the"}
_PUNCT = re.compile(r"[^a-z0-9 ]")


def normalize_answer(text):
    """The standard VQA answer normalisation (lowercase, strip articles
    and punctuation, collapse whitespace)."""
    text = text.lower().strip()
    text = _PUNCT.sub(" ", text)
    tokens = [t for t in text.split() if t and t not in _ARTICLES]
    return " ".join(tokens)


def vqa_accuracy(prediction, ground_truths):
    """VQA-style soft score: min(#matching references / 3, 1).

    With the single-reference VQAv2 and GQA loaders this is exact match after
    normalisation; OK-VQA retains its ten human references.
    """
    pred = normalize_answer(prediction)
    if not pred:
        return 0.0
    golds = [normalize_answer(g) for g in ground_truths]
    matches = sum(1 for g in golds if g == pred)
    return min(matches / 3.0, 1.0)


def bootstrap_ci(scores, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap confidence interval for a mean VQA score.

    `scores` are the per-question VQA scores in [0, 1].  Returned values
    are in percent, matching the accuracies reported elsewhere.
    """
    if not scores:
        return None
    data = np.asarray(scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(data), size=(n_boot, len(data)))
    means = data[idx].mean(axis=1)
    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": round(100.0 * float(data.mean()), 2),
            "low": round(100.0 * float(low), 2),
            "high": round(100.0 * float(high), 2),
            "half_width": round(100.0 * float(high - low) / 2.0, 2),
            "n": int(len(data))}


def paired_delta_ci(scores_a, scores_b, n_boot=10000, alpha=0.05, seed=0):
    """Bootstrap CI for the paired difference mean(a) - mean(b).

    The two conditions are scored on the same questions with the same
    weights, so pairing removes the question-difficulty variance and gives
    a much tighter interval than comparing two independent intervals.
    """
    if not scores_a or not scores_b or len(scores_a) != len(scores_b):
        return None
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"delta": round(100.0 * float(diff.mean()), 2),
            "low": round(100.0 * float(low), 2),
            "high": round(100.0 * float(high), 2),
            "significant": bool(low > 0 or high < 0),
            "n": int(len(diff))}


def save_json(name, payload):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    with io.open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


class Timer(object):
    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time.time()
        return self

    def __exit__(self, *args):
        torch.cuda.synchronize()
        self.elapsed = time.time() - self.start
