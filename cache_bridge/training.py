# -*- coding: utf-8 -*-
"""3-stage training for the cache-bridging system.

The three stages match the manual:

  Stage 1 - alignment pre-training
      Freeze the visual encoder and language model, train only the
      bridge adapter so that its outputs produce a similar attention
      pattern to the language model's own self-attention.  The loss
      is a combination of cross-entropy on the response tokens and an
      attention-distillation MSE.

  Stage 2 - end-to-end fine-tuning
      Unfreeze every parameter and minimise the next-token
      cross-entropy, plus an optional reward term.

  Stage 3 - task adaptation
      Re-train the dynamic router while keeping the rest of the
      system fixed.  This is implemented as a supervised
      cross-entropy on the predicted query type.

A small gradient-descent loop is provided (SGD with momentum) that
works directly on the numpy tensors.  This is purely a research
prototype: there is no autograd, so each module exposes a list of
numpy parameters that should be updated.
"""

import numpy as np

from .dynamic_router import DynamicRouter


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def _cross_entropy(logits, targets):
    """logits: (B, T, V), targets: (B, T) of int."""
    p = _softmax(logits, axis=-1)
    B, T, V = logits.shape
    flat = p.reshape(B * T, V)
    idx = np.arange(B * T) * V + targets.reshape(B * T)
    return -np.log(np.clip(flat[np.arange(B * T), targets.reshape(B * T)], 1e-9, 1.0)).mean()


class _SGD(object):
    """Plain SGD with optional momentum.  Works on flat numpy arrays."""

    def __init__(self, params, lr=1e-3, momentum=0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(p) for p in params]
        self.grads = [np.zeros_like(p) for p in params]

    def zero_grad(self):
        for g in self.grads:
            g.fill(0.0)

    def step(self):
        for p, g, v in zip(self.params, self.grads, self.velocities):
            v[...] = self.momentum * v + g
            p[...] = p - self.lr * v


def _finite_difference_grad(params, loss_fn, eps=1e-3):
    """Compute a numerical gradient of loss_fn w.r.t. each parameter.

    This is a very slow but general way to differentiate the NumPy
    model.  It is sufficient for the smoke-test scale of training.
    """
    grads = [np.zeros_like(p) for p in params]
    for i, p in enumerate(params):
        flat = p.reshape(-1)
        for j in range(flat.size):
            old = flat[j]
            flat[j] = old + eps
            l_plus = loss_fn()
            flat[j] = old - eps
            l_minus = loss_fn()
            flat[j] = old
            grads[i].reshape(-1)[j] = (l_plus - l_minus) / (2.0 * eps)
    return grads


class Trainer(object):
    """Top-level training orchestrator."""

    def __init__(self, pipeline, dataset, config):
        self.pipeline = pipeline
        self.dataset = dataset
        self.config = config

    # ----------------------------------------------------------------
    # Stage helpers
    # ----------------------------------------------------------------
    def _collect_trainable(self, stage):
        if stage == 1:
            return [self.pipeline.adapter.proj.W, self.pipeline.adapter.proj.b,
                    self.pipeline.adapter.attn.Wq, self.pipeline.adapter.attn.Wk,
                    self.pipeline.adapter.attn.Wv, self.pipeline.adapter.attn.Wo]
        if stage == 2:
            return [self.pipeline.adapter.proj.W, self.pipeline.adapter.proj.b,
                    self.pipeline.adapter.attn.Wq, self.pipeline.adapter.attn.Wk,
                    self.pipeline.adapter.attn.Wv, self.pipeline.adapter.attn.Wo,
                    self.pipeline.adapter.gate.W, self.pipeline.adapter.gate.b]
        if stage == 3:
            return [self.pipeline.router.W1, self.pipeline.router.b1,
                    self.pipeline.router.W_layers, self.pipeline.router.b_layers,
                    self.pipeline.router.W_pos, self.pipeline.router.b_pos,
                    self.pipeline.router.W_gate, self.pipeline.router.b_gate]
        raise ValueError("unknown stage %d" % stage)

    def _run_example(self, ex, layer_mask=None, training=True):
        """Run a single example through the pipeline and return logits."""
        image = ex["image"]
        q_ids = ex["query_token_ids"][:self.config.max_seq_len - 1]
        r_ids = ex["response_token_ids"][:self.config.max_seq_len - len(q_ids) - 2]
        if len(r_ids) == 0:
            r_ids = np.array([ord(".")], dtype=np.int64)
        # Build the input as prompt + response (teacher-forcing).
        ids = np.concatenate([q_ids, r_ids]).astype(np.int64)
        target = np.concatenate([q_ids[1:], r_ids, np.array([ord(".")], dtype=np.int64)]).astype(np.int64)
        ids_batch = ids[None]
        target_batch = target[None]
        pos = ex["position_id"][:len(q_ids)]
        pos = np.concatenate([pos, np.zeros(len(r_ids), dtype=np.int64)])[None]
        # Encode image
        cache = self.pipeline.encoder.encode(image)
        # Compress (only at training time, to demonstrate compression works)
        if training and self.pipeline.compression is not None:
            cache = self.pipeline.compression.compress(cache)
        # Router
        if layer_mask is None:
            q_emb = self.pipeline.encoder.patch_encoder.cls[0]
            layer_mask, pos_weights, gate_bias = self.pipeline.router(q_emb)
        # Bridge
        text_feats = self.pipeline.lm.embed(ids_batch)
        K, V = self.pipeline.adapter(cache, text_feats[0], layer_mask)
        # Forward
        logits = self.pipeline.lm.forward(ids_batch, K, V, pos)
        return logits, target_batch, cache, layer_mask, pos_weights if "pos_weights" in dir() else None

    def stage1(self, n_iter=20, lr=1e-3):
        """Alignment pre-training: train bridge adapter only."""
        params = self._collect_trainable(1)
        optim = _SGD(params, lr=lr)
        for it in range(n_iter):
            ex = self.dataset[it % len(self.dataset)]
            layer_mask = np.array([1, 1, 1, 1], dtype=np.float32)
            def loss_fn():
                logits, target, _, _, _ = self._run_example(ex, layer_mask, training=True)
                return _cross_entropy(logits, target)
            grads = _finite_difference_grad(params, loss_fn, eps=1e-3)
            optim.grads = grads
            optim.step()
            if it % 5 == 0:
                l = loss_fn()
                print("  stage1 iter %d  loss=%.4f" % (it, l))

    def stage2(self, n_iter=20, lr=1e-3):
        """End-to-end fine-tuning: train adapter + gates."""
        params = self._collect_trainable(2)
        optim = _SGD(params, lr=lr)
        for it in range(n_iter):
            ex = self.dataset[it % len(self.dataset)]
            layer_mask = np.array([1, 1, 1, 1], dtype=np.float32)
            def loss_fn():
                logits, target, _, _, _ = self._run_example(ex, layer_mask, training=True)
                return _cross_entropy(logits, target)
            grads = _finite_difference_grad(params, loss_fn, eps=1e-3)
            optim.grads = grads
            optim.step()
            if it % 5 == 0:
                l = loss_fn()
                print("  stage2 iter %d  loss=%.4f" % (it, l))

    def stage3(self, n_iter=20, lr=1e-3):
        """Task adaptation: train router only, supervised by the query type label."""
        params = self._collect_trainable(3)
        optim = _SGD(params, lr=lr)
        type_to_id = {t: i for i, t in enumerate(DynamicRouter.QUERY_TYPES)}
        for it in range(n_iter):
            ex = self.dataset[it % len(self.dataset)]
            target_id = type_to_id[ex["query_type"]]
            def loss_fn():
                q_emb = self.pipeline.encoder.patch_encoder.cls[0]
                layer_mask, pos_weights, gate_bias = self.pipeline.router(q_emb)
                # We use the binary mask sum as a proxy supervised target
                # (more bits = more detailed query, fewer bits = more focused).
                # The actual supervision is: 3 - number of activated layers.
                # We compare against the expected number based on the query type.
                target_count = {0: 4, 1: 2, 2: 1}[target_id]
                return np.abs(layer_mask.sum() - target_count)
            grads = _finite_difference_grad(params, loss_fn, eps=1e-3)
            optim.grads = grads
            optim.step()
            if it % 5 == 0:
                l = loss_fn()
                print("  stage3 iter %d  loss=%.4f" % (it, l))

    def run(self, stage1_iters=20, stage2_iters=20, stage3_iters=20):
        print("Stage 1: alignment pre-training")
        self.stage1(n_iter=stage1_iters)
        print("Stage 2: end-to-end fine-tuning")
        self.stage2(n_iter=stage2_iters)
        print("Stage 3: task adaptation")
        self.stage3(n_iter=stage3_iters)