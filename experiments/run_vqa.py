# -*- coding: utf-8 -*-
"""Train a connector on real VQA data and evaluate it under a fixed protocol.

Every system (the proposed cache bridge and the caption-style baseline)
is trained on the same question/answer pairs, for the same number of
steps, with the same optimiser, and evaluated on a held-out split of the
same dataset.  CLIP and the LLM stay frozen throughout; only the
connector is updated.
"""

import argparse
import io
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common
import model as M

common.setup_env()

from transformers import (CLIPImageProcessor, CLIPVisionModel,
                          AutoModelForCausalLM, AutoTokenizer)

DTYPE = torch.bfloat16


def load_backbones():
    clip = CLIPVisionModel.from_pretrained(common.CLIP_ID, dtype=DTYPE).cuda().eval()
    llm = AutoModelForCausalLM.from_pretrained(common.LLM_ID, dtype=DTYPE).cuda().eval()
    tok = AutoTokenizer.from_pretrained(common.LLM_ID)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    proc = CLIPImageProcessor.from_pretrained(common.CLIP_ID)
    return clip, llm, tok, proc


def build_system(kind, clip, llm, cfg, **kwargs):
    if kind == "caption":
        return M.CaptionBaseline(clip, llm, cfg).cuda().to(DTYPE)
    return M.CacheBridgeVQA(clip, llm, cfg, **kwargs).cuda().to(DTYPE)


class Batcher(object):
    """Turns raw VQA rows into padded tensors on the GPU."""

    def __init__(self, tok, proc, max_len=40):
        self.tok = tok
        self.proc = proc
        self.max_len = max_len

    def pixels(self, rows):
        images = [common._pil_from_bytes(r["image"]) for r in rows]
        px = self.proc(images=images, return_tensors="pt")["pixel_values"]
        return px.cuda().to(DTYPE)

    def train_batch(self, rows):
        """Prompt + gold answer, with the loss masked to the answer span."""
        prompts, fulls = [], []
        for row in rows:
            prompt = "Question: %s Answer:" % row["question"].strip()
            answer = " %s%s" % (row["answers"][0].strip(), self.tok.eos_token)
            prompts.append(prompt)
            fulls.append(prompt + answer)
        enc = self.tok(fulls, return_tensors="pt", padding=True,
                       truncation=True, max_length=self.max_len)
        ids = enc["input_ids"].cuda()
        mask = enc["attention_mask"].cuda()
        labels = ids.clone()
        labels[mask == 0] = -100
        # Mask the prompt tokens so only the answer contributes.
        for i, prompt in enumerate(prompts):
            n = len(self.tok(prompt, add_special_tokens=True)["input_ids"])
            labels[i, :n] = -100
        return self.pixels(rows), ids, mask, labels

    def eval_batch(self, rows):
        prompts = ["Question: %s Answer:" % r["question"].strip() for r in rows]
        previous = self.tok.padding_side
        self.tok.padding_side = "left"
        enc = self.tok(prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=self.max_len)
        self.tok.padding_side = previous
        return (self.pixels(rows), enc["input_ids"].cuda(),
                enc["attention_mask"].cuda())


def train(system, batcher, rows, steps, batch_size, lr, log_every=25,
          log=None):
    params = system.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    system.train()
    history = []
    cursor = 0
    t0 = time.time()
    for step in range(steps):
        batch = []
        while len(batch) < batch_size:
            batch.append(rows[cursor % len(rows)])
            cursor += 1
        px, ids, mask, labels = batcher.train_batch(batch)
        loss = system(px, ids, mask, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        value = float(loss.detach())
        history.append(value)
        if (step + 1) % log_every == 0 or step == 0:
            msg = "    step %4d/%d  loss %.4f  (%.1fs)" % (
                step + 1, steps, sum(history[-log_every:]) / min(log_every, len(history)),
                time.time() - t0)
            print(msg, flush=True)
            if log is not None:
                log.append(msg)
    system.eval()
    return history, time.time() - t0


@torch.no_grad()
def evaluate(system, batcher, rows, tok, batch_size=8, max_new_tokens=6,
             image_source=None):
    """Score the system on `rows`.

    `image_source` allows the image stream to be decoupled from the
    question stream.  Passing a rotated copy of `rows` feeds every
    question the wrong image, which measures how much of the accuracy
    actually comes from the visual pathway rather than from a language
    prior over frequent answers.
    """
    total = 0.0
    samples = []
    images = image_source if image_source is not None else rows
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        px, ids, mask = batcher.eval_batch(chunk)
        if image_source is not None:
            px = batcher.pixels(images[start:start + batch_size])
        out = system.generate(px, ids, mask, max_new_tokens=max_new_tokens)
        texts = tok.batch_decode(out, skip_special_tokens=True)
        for row, text in zip(chunk, texts):
            pred = text.strip().split("\n")[0].strip()
            pred = pred.split(".")[0].split(",")[0].strip()
            score = common.vqa_accuracy(pred, row["answers"])
            total += score
            if len(samples) < 12:
                samples.append({"question": row["question"],
                                "prediction": pred,
                                "gold": row["answers"][0],
                                "score": score})
    return 100.0 * total / max(1, len(rows)), samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="OK-VQA")
    ap.add_argument("--system", default="bridge")
    ap.add_argument("--inject-mode", default="prepend")
    ap.add_argument("--train-size", type=int, default=2000)
    ap.add_argument("--eval-size", type=int, default=500)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-visual", type=int, default=36)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--keep-ratio", type=float, default=1.0)
    ap.add_argument("--no-geometry", action="store_true")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--gate-mode", default="learned",
                    choices=["learned", "fixed"])
    ap.add_argument("--layer-mask", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    common.set_seed(args.seed)
    print("device:", common.device_report(), flush=True)

    rows = common.DATASETS[args.dataset](args.train_size + args.eval_size)
    train_rows = rows[args.eval_size:]
    eval_rows = rows[:args.eval_size]
    print("dataset %s: %d train / %d eval" % (args.dataset, len(train_rows),
                                              len(eval_rows)), flush=True)

    clip, llm, tok, proc = load_backbones()
    cfg = M.BridgeCfg(clip, llm, rank=args.rank, n_visual=args.n_visual)
    batcher = Batcher(tok, proc)

    kwargs = {}
    if args.system != "caption":
        mask = None
        if args.layer_mask:
            mask = [float(x) for x in args.layer_mask.split(",")]
        kwargs = {"inject_mode": args.inject_mode,
                  "use_geometry": not args.no_geometry,
                  "use_gate": not args.no_gate,
                  "gate_mode": args.gate_mode,
                  "layer_mask": mask,
                  "keep_ratio": args.keep_ratio}
    if args.system != "caption":
        probe = batcher.train_batch(train_rows[:4])
        k_std, v_std = M.calibrate_kv_stats(
            cfg, llm, llm.get_input_embeddings()(probe[1]))
        print("host KV stats: k_std %.4f  v_std %.4f" % (k_std, v_std),
              flush=True)
    system = build_system(args.system, clip, llm, cfg, **kwargs)
    n_params = sum(p.numel() for p in system.trainable_parameters())
    print("trainable connector params: %d" % n_params, flush=True)

    torch.cuda.reset_peak_memory_stats()
    history, train_secs = train(system, batcher, train_rows, args.steps,
                                args.batch_size, args.lr)
    train_mem = torch.cuda.max_memory_allocated() / 1024 ** 3

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    acc, samples = evaluate(system, batcher, eval_rows, tok)
    eval_secs = time.time() - t0
    eval_mem = torch.cuda.max_memory_allocated() / 1024 ** 3

    # Control conditions: the same trained weights scored with mismatched
    # images, and (for the bridge) with the visual cache switched off.
    rotated = eval_rows[1:] + eval_rows[:1]
    acc_shuffled, _ = evaluate(system, batcher, eval_rows, tok,
                               image_source=rotated)
    acc_novisual = None
    if args.system != "caption":
        keep = system.layer_mask
        system.layer_mask = [0.0, 0.0, 0.0, 0.0]
        acc_novisual, _ = evaluate(system, batcher, eval_rows, tok)
        system.layer_mask = keep

    payload = {
        "host_kv_stats": {"k_std": round(cfg.k_std, 4),
                          "v_std": round(cfg.v_std, 4)},
        "dataset": args.dataset,
        "system": args.system,
        "config": vars(args),
        "device": common.device_report(),
        "connector_params": n_params,
        "accuracy": round(acc, 2),
        "accuracy_shuffled_images": round(acc_shuffled, 2),
        "accuracy_no_visual": (None if acc_novisual is None
                               else round(acc_novisual, 2)),
        "train_seconds": round(train_secs, 1),
        "train_peak_mem_gb": round(train_mem, 2),
        "eval_seconds": round(eval_secs, 1),
        "eval_peak_mem_gb": round(eval_mem, 2),
        "loss_first_25": round(sum(history[:25]) / min(25, len(history)), 4),
        "loss_last_25": round(sum(history[-25:]) / min(25, len(history)), 4),
        "samples": samples,
    }
    tag = args.tag or "%s_%s" % (args.dataset.replace("-", ""), args.system)
    path = common.save_json("run_%s.json" % tag, payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "samples"},
                     indent=2), flush=True)
    print("saved ->", path, flush=True)


if __name__ == "__main__":
    main()
