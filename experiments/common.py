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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    """Real VQAv2 validation subset (merve/vqav2-small)."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download("merve/vqav2-small",
                           "data/validation-00000-of-00007.parquet",
                           repo_type="dataset")
    table = pq.read_table(path, columns=["question", "multiple_choice_answer", "image"])
    rows = []
    for record in table.to_pylist():
        rows.append({
            "question": record["question"],
            "answers": [record["multiple_choice_answer"]] * 10,
            "image": record["image"]["bytes"],
        })
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
    """Official VQA accuracy: min(#humans that said it / 3, 1)."""
    pred = normalize_answer(prediction)
    if not pred:
        return 0.0
    golds = [normalize_answer(g) for g in ground_truths]
    matches = sum(1 for g in golds if g == pred)
    return min(matches / 3.0, 1.0)


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
