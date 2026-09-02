# -*- coding: utf-8 -*-
"""Cache-bridging model wired to real CLIP and TinyLlama checkpoints.

Two systems are defined here so that they can be compared under an
identical protocol:

  CaptionBaseline   the conventional serial pipeline.  CLIP produces a
                    pooled description of the image, that description
                    is turned into text-space tokens by a projection
                    (the LLaVA-style single MLP), and the LLM consumes
                    the resulting soft prompt.

  CacheBridgeModel  the proposed system.  A hierarchical four-level KV
                    cache is tapped from four CLIP encoder layers,
                    lifted into the LLM's per-head KV space by the
                    bridge adapter, and spliced into every decoder
                    layer's past-key-values.

Both systems keep CLIP and the LLM frozen and train only their own
connector, so the parameter budget is directly comparable.
"""

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache_bridge_torch.bridge import (
    DimensionProjection, StructurePreservingAttn, AdaptiveGating,
)

LAYER_NAMES = ["basic", "semantic_object", "spatial_relation", "attribute"]

EOS_WEIGHT = 0.2


def answer_loss(logits, labels, eos_id):
    """Cross-entropy over the answer span with a down-weighted EOS."""
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    per_token = F.cross_entropy(flat_logits, flat_labels,
                                ignore_index=-100, reduction="none")
    valid = (flat_labels != -100).float()
    weight = torch.where(flat_labels == eos_id,
                         torch.full_like(valid, EOS_WEIGHT),
                         torch.ones_like(valid)) * valid
    return (per_token * weight).sum() / weight.sum().clamp(min=1.0)


class BridgeCfg(object):
    """Shape configuration resolved from the real checkpoints."""

    def __init__(self, clip, llm, rank=64, n_visual=32):
        vcfg = clip.config
        lcfg = llm.config
        self.vision_dim = vcfg.hidden_size
        self.vision_num_heads = vcfg.num_attention_heads
        self.vision_head_dim = vcfg.hidden_size // vcfg.num_attention_heads
        self.vision_num_layers = vcfg.num_hidden_layers
        self.image_size = vcfg.image_size
        self.patch_size = vcfg.patch_size
        self.grid = vcfg.image_size // vcfg.patch_size
        # Four taps spread over the encoder: early / mid / late / final.
        n = vcfg.num_hidden_layers
        self.layer_idx = {
            "basic": max(1, n // 6),
            "semantic_object": n // 2,
            "spatial_relation": int(n * 0.75),
            "attribute": n - 2,
        }
        # The LLM uses grouped-query attention, so its KV width is
        # num_key_value_heads * head_dim, which is narrower than the
        # hidden size.  The bridge writes into the KV space, therefore
        # `language_dim` is the KV width rather than the hidden size.
        self.hidden_size = lcfg.hidden_size
        self.language_num_heads = lcfg.num_key_value_heads
        self.language_head_dim = lcfg.hidden_size // lcfg.num_attention_heads
        self.language_dim = self.language_num_heads * self.language_head_dim
        self.language_num_layers = lcfg.num_hidden_layers
        self.bridge_rank = rank
        self.bridge_dropout = 0.0
        self.gate_hidden = 32
        self.max_seq_len = 2048
        # Visual tokens kept per cache level after pooling.
        self.n_visual = n_visual
        # Statistics of the host model's own KV cache.  These are measured
        # from the real checkpoint by `calibrate_kv_stats` so the bridge
        # starts in the same numeric range as the cache it is spliced into.
        self.k_std = 1.0
        self.v_std = 0.02


@torch.no_grad()
def calibrate_kv_stats(cfg, llm, embed):
    """Measure the host model's own K/V spread on a real batch.

    The bridged cache is spliced next to these tensors, so it has to live
    on the same scale.  Keys and values differ by almost two orders of
    magnitude in LLaMA-style models, which is why they are calibrated
    separately.
    """
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    llm(inputs_embeds=embed, past_key_values=cache, use_cache=True)
    k_stds, v_stds = [], []
    for layer in cache.layers:
        k_stds.append(layer.keys.float().std().item())
        v_stds.append(layer.values.float().std().item())
    cfg.k_std = sum(k_stds) / len(k_stds)
    cfg.v_std = sum(v_stds) / len(v_stds)
    return cfg.k_std, cfg.v_std


class HierarchicalCacheExtractor(nn.Module):
    """Taps four CLIP encoder layers and returns a per-level K/V cache."""

    def __init__(self, cfg, clip):
        super(HierarchicalCacheExtractor, self).__init__()
        self.cfg = cfg
        self.clip = clip
        self._captures = {}
        self._handles = []
        grid = cfg.grid
        rows = torch.arange(grid).repeat_interleave(grid).float()
        cols = torch.arange(grid).repeat(grid).float()
        self.register_buffer("patch_coords", torch.stack([rows, cols], dim=1))

    def _hook(self, name):
        def fn(module, args, kwargs):
            hidden = args[0] if args else kwargs.get("hidden_states")
            self._captures[name] = hidden
        return fn

    def __enter__(self):
        for name, idx in self.cfg.layer_idx.items():
            layer = self.clip.encoder.layers[idx]
            self._handles.append(
                layer.self_attn.register_forward_pre_hook(self._hook(name),
                                                          with_kwargs=True))
        return self

    def __exit__(self, *args):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def forward(self, pixel_values):
        self._captures = {}
        with self:
            with torch.no_grad():
                self.clip(pixel_values=pixel_values)
        cache = {}
        coords = {}
        pool = self.cfg.n_visual
        grid = self.cfg.grid
        for name in LAYER_NAMES:
            hidden = self._captures[name]
            layer = self.clip.encoder.layers[self.cfg.layer_idx[name]].self_attn
            batch, seq, _ = hidden.shape
            with torch.no_grad():
                k = layer.k_proj(hidden)
                v = layer.v_proj(hidden)
            # Drop the CLS row, keep the 2D patch grid.
            k = k[:, 1:, :]
            v = v[:, 1:, :]
            coord = self.patch_coords.unsqueeze(0).expand(batch, -1, -1)
            # Average-pool the patch grid down to `pool` tokens so the
            # injected cache stays short.  Pooling is done on the 2D
            # grid so the spatial layout survives.
            side = int(math.sqrt(pool))
            k = self._pool2d(k, grid, side)
            v = self._pool2d(v, grid, side)
            coord = self._pool_coords(coord, grid, side)
            cache[name] = {"K": k, "V": v}
            coords[name] = coord
        return cache, coords

    @staticmethod
    def _grid_pool(x, grid, side):
        """Average-pool a (B, C, grid, grid) map down to (B, C, side, side).

        F.adaptive_avg_pool2d has a non-deterministic CUDA backward, which
        made identical configurations irreproducible.  When grid is an exact
        multiple of side the same result is obtained by reshaping into blocks
        and taking a mean, which is deterministic; otherwise we fall back to
        an explicit index-based block mean.  Both paths are differentiable.
        """
        batch, channels = x.shape[0], x.shape[1]
        if grid % side == 0:
            factor = grid // side
            x = x.reshape(batch, channels, side, factor, side, factor)
            return x.mean(dim=(3, 5))
        # Unequal split: build the same block boundaries adaptive pooling uses
        # and average each block explicitly.
        bounds = [(i * grid) // side for i in range(side + 1)]
        rows = torch.stack([x[:, :, bounds[i]:bounds[i + 1], :].mean(dim=2)
                            for i in range(side)], dim=2)
        return torch.stack([rows[:, :, :, bounds[j]:bounds[j + 1]].mean(dim=3)
                            for j in range(side)], dim=3)

    @classmethod
    def _pool2d(cls, x, grid, side):
        batch, _, dim = x.shape
        x = x.transpose(1, 2).reshape(batch, dim, grid, grid)
        x = cls._grid_pool(x, grid, side)
        return x.reshape(batch, dim, side * side).transpose(1, 2)

    @classmethod
    def _pool_coords(cls, coord, grid, side):
        batch = coord.shape[0]
        c = coord.transpose(1, 2).reshape(batch, 2, grid, grid)
        c = cls._grid_pool(c, grid, side)
        return c.reshape(batch, 2, side * side).transpose(1, 2)


class BridgeAdapter(nn.Module):
    """Lifts the visual cache into the LLM KV space.

    Three stages, matching the architecture description: a low-rank
    dimension projection, a 2D-geometry-preserving attention, and a
    context-aware gate.  The `use_geometry` and `use_gate` switches
    exist so the ablations can disable one stage at a time.
    """

    def __init__(self, cfg, use_geometry=True, use_gate=True,
                 gate_mode="learned"):
        super(BridgeAdapter, self).__init__()
        self.cfg = cfg
        self.use_geometry = use_geometry
        self.use_gate = use_gate
        self.gate_mode = gate_mode
        self.proj_k = DimensionProjection(cfg)
        self.proj_v = DimensionProjection(cfg)
        if use_geometry:
            self.geo_k = StructurePreservingAttn(cfg)
            self.geo_v = StructurePreservingAttn(cfg)
        else:
            # Same parameter budget, but a content-only additive bias.
            self.flat_k = nn.Linear(cfg.language_dim, cfg.language_dim, bias=False)
            self.flat_v = nn.Linear(cfg.language_dim, cfg.language_dim, bias=False)
        if use_gate:
            # gate_mode "learned" is the full adaptive gate.  gate_mode
            # "fixed" keeps the prompt-summary projection but replaces the
            # learned per-token coefficient by a constant 0.5 blend, which
            # isolates the effect of *adaptive* gating from the parameters
            # the gate brings with it (the equal-capacity control of
            # Section 4.5).
            if gate_mode == "learned":
                self.gate = AdaptiveGating(cfg)
            self.summary_proj = nn.Linear(cfg.hidden_size, cfg.language_dim,
                                          bias=False)
        self.level_scale = nn.Parameter(torch.ones(len(LAYER_NAMES)))
        self.k_norm = nn.LayerNorm(cfg.language_dim)
        self.v_norm = nn.LayerNorm(cfg.language_dim)
        # Match the host model's KV statistics at initialisation.
        self.k_scale = nn.Parameter(torch.tensor(float(cfg.k_std)))
        self.v_scale = nn.Parameter(torch.tensor(float(cfg.v_std)))

    def forward(self, cache, coords, text_summary, layer_mask=None):
        cfg = self.cfg
        keys = []
        values = []
        for i, name in enumerate(LAYER_NAMES):
            if layer_mask is not None and layer_mask[i] <= 0:
                continue
            k = self.proj_k(cache[name]["K"])
            v = self.proj_v(cache[name]["V"])
            coord = coords[name]
            if self.use_geometry:
                k = k + self.geo_k(k, coord)
                v = v + self.geo_v(v, coord)
            else:
                k = k + self.flat_k(k)
                v = v + self.flat_v(v)
            if self.use_gate:
                summary = self.summary_proj(text_summary)
                if self.gate_mode == "fixed":
                    k = 0.5 * k + 0.5 * summary
                    v = 0.5 * v + 0.5 * summary
                else:
                    k = self.gate(k, summary)
                    v = self.gate(v, summary)
            k = k * self.level_scale[i]
            v = v * self.level_scale[i]
            keys.append(k)
            values.append(v)
        if not keys:
            return None, None
        k = self.k_norm(torch.cat(keys, dim=1)) * self.k_scale
        v = self.v_norm(torch.cat(values, dim=1)) * self.v_scale
        batch, seq, _ = k.shape
        heads = cfg.language_num_heads
        head_dim = cfg.language_head_dim
        k = k.view(batch, seq, heads, head_dim).transpose(1, 2)
        v = v.view(batch, seq, heads, head_dim).transpose(1, 2)
        return k, v

class CacheBridgeVQA(nn.Module):
    """The proposed system: hierarchical visual KV spliced into the LLM.

    The bridged (K, V) is prepended to every decoder layer's
    past-key-values, so the text tokens attend to the visual cache
    through the LLM's own attention rather than through a soft prompt.
    """

    mode = "bridge"

    def __init__(self, clip, llm, cfg, inject_mode="prepend",
                 use_geometry=True, use_gate=True, gate_mode="learned",
                 layer_mask=None,
                 keep_ratio=1.0, per_layer_scale=False):
        super(CacheBridgeVQA, self).__init__()
        self.clip = clip
        self.llm = llm
        self.cfg = cfg
        self.extractor = HierarchicalCacheExtractor(cfg, clip)
        self.bridge = BridgeAdapter(cfg, use_geometry, use_gate,
                                    gate_mode)
        self.inject_mode = inject_mode
        self.layer_mask = layer_mask
        self.keep_ratio = keep_ratio
        # Two learned scalars (one for K, one for V) per decoder layer.
        # Initialised at one, so the model starts from the shared-cache
        # configuration and any gain is attributable to depth adaptation.
        if per_layer_scale:
            self.layer_scales = nn.Parameter(
                torch.ones(llm.config.num_hidden_layers, 2))
        else:
            self.layer_scales = None
        for param in self.clip.parameters():
            param.requires_grad = False
        for param in self.llm.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        params = [p for p in self.bridge.parameters() if p.requires_grad]
        if self.layer_scales is not None:
            params.append(self.layer_scales)
        return params

    def _compress(self, k, v):
        """Top-K salience selection used by the compression ablation."""
        if self.keep_ratio >= 1.0:
            return k, v
        seq = k.shape[2]
        keep = max(1, int(round(seq * self.keep_ratio)))
        salience = v.pow(2).sum(dim=(1, 3))
        idx = salience.topk(keep, dim=1).indices.sort(dim=1).values
        gather = idx[:, None, :, None].expand(-1, k.shape[1], -1, k.shape[3])
        return k.gather(2, gather), v.gather(2, gather)

    def visual_kv(self, pixel_values, text_summary):
        cache, coords = self.extractor(pixel_values)
        k, v = self.bridge(cache, coords, text_summary, self.layer_mask)
        if k is None:
            return None, None
        return self._compress(k, v)

    def forward(self, pixel_values, input_ids, attention_mask, labels):
        embed = self.llm.get_input_embeddings()(input_ids)
        summary = embed.detach().mean(dim=1, keepdim=True)
        k, v = self.visual_kv(pixel_values, summary)
        return run_llm_with_visual_kv(self.llm, embed, attention_mask,
                                      labels, k, v, self.inject_mode,
                                      self.layer_scales)

    @torch.no_grad()
    def generate(self, pixel_values, input_ids, attention_mask, max_new_tokens=8):
        embed = self.llm.get_input_embeddings()(input_ids)
        summary = embed.mean(dim=1, keepdim=True)
        k, v = self.visual_kv(pixel_values, summary)
        return generate_with_visual_kv(self.llm, embed, attention_mask, k, v,
                                       self.inject_mode, max_new_tokens,
                                       self.layer_scales)


class CaptionBaseline(nn.Module):
    """The conventional serial pipeline (LLaVA-style connector).

    CLIP's patch features are projected by a two-layer MLP into the LLM
    embedding space and prepended to the prompt as a soft visual
    prefix.  No KV injection: the visual information has to pass
    through the LLM's input layer, which is exactly the bottleneck the
    proposed method removes.
    """

    mode = "caption"

    def __init__(self, clip, llm, cfg, n_visual=32):
        super(CaptionBaseline, self).__init__()
        self.clip = clip
        self.llm = llm
        self.cfg = cfg
        self.n_visual = n_visual
        # When False the soft visual prefix is zeroed, which is the
        # baseline's equivalent of the bridge's disabled level mask.
        self.use_visual = True
        self.connector = nn.Sequential(
            nn.Linear(cfg.vision_dim, cfg.hidden_size),
            nn.GELU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
        )
        self.out_norm = nn.LayerNorm(cfg.hidden_size)
        for param in self.clip.parameters():
            param.requires_grad = False
        for param in self.llm.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        return ([p for p in self.connector.parameters()]
                + [p for p in self.out_norm.parameters()])

    def visual_prefix(self, pixel_values):
        with torch.no_grad():
            out = self.clip(pixel_values=pixel_values)
            hidden = out.last_hidden_state[:, 1:, :]
        side = int(math.sqrt(self.n_visual))
        grid = self.cfg.grid
        batch, _, dim = hidden.shape
        pooled = hidden.transpose(1, 2).reshape(batch, dim, grid, grid)
        pooled = HierarchicalCacheExtractor._grid_pool(pooled, grid, side)
        pooled = pooled.reshape(batch, dim, side * side).transpose(1, 2)
        prefix = self.out_norm(self.connector(pooled))
        if not self.use_visual:
            prefix = torch.zeros_like(prefix)
        return prefix

    def forward(self, pixel_values, input_ids, attention_mask, labels):
        prefix = self.visual_prefix(pixel_values)
        embed = self.llm.get_input_embeddings()(input_ids)
        inputs = torch.cat([prefix, embed], dim=1)
        pad = torch.ones(prefix.shape[:2], dtype=attention_mask.dtype,
                         device=attention_mask.device)
        mask = torch.cat([pad, attention_mask], dim=1)
        ignore = torch.full(prefix.shape[:2], -100, dtype=labels.dtype,
                            device=labels.device)
        full_labels = torch.cat([ignore, labels], dim=1)
        out = self.llm(inputs_embeds=inputs, attention_mask=mask)
        eos_id = self.llm.config.eos_token_id
        if isinstance(eos_id, (list, tuple)):
            eos_id = eos_id[0]
        return answer_loss(out.logits, full_labels, eos_id)

    @torch.no_grad()
    def generate(self, pixel_values, input_ids, attention_mask, max_new_tokens=8):
        prefix = self.visual_prefix(pixel_values)
        embed = self.llm.get_input_embeddings()(input_ids)
        inputs = torch.cat([prefix, embed], dim=1)
        pad = torch.ones(prefix.shape[:2], dtype=attention_mask.dtype,
                         device=attention_mask.device)
        mask = torch.cat([pad, attention_mask], dim=1)
        return greedy_decode(self.llm, inputs, mask, max_new_tokens)


class BlindBaseline(nn.Module):
    """Question-only control: the frozen LLM with no visual pathway at all.

    The image is never read.  A soft prefix of the same length as the other
    two systems is learned from a bias vector, so the LLM sees a prompt of
    comparable shape and the only thing missing is the image.  This measures
    how much of any reported accuracy is a language prior over frequent
    answers rather than visual grounding, which neither the mismatched-image
    nor the disabled-pathway control can establish on its own.
    """

    mode = "blind"

    def __init__(self, clip, llm, cfg, n_visual=32):
        super(BlindBaseline, self).__init__()
        self.llm = llm
        self.cfg = cfg
        self.n_visual = n_visual
        side = int(math.sqrt(n_visual))
        self.prefix = nn.Parameter(torch.zeros(side * side, cfg.hidden_size))
        nn.init.normal_(self.prefix, std=0.02)
        for param in self.llm.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        return [self.prefix]

    def visual_prefix(self, pixel_values):
        batch = pixel_values.shape[0]
        return self.prefix.unsqueeze(0).expand(batch, -1, -1).to(pixel_values.dtype)

    def forward(self, pixel_values, input_ids, attention_mask, labels):
        prefix = self.visual_prefix(pixel_values)
        embed = self.llm.get_input_embeddings()(input_ids)
        inputs = torch.cat([prefix, embed], dim=1)
        pad = torch.ones(prefix.shape[:2], dtype=attention_mask.dtype,
                         device=attention_mask.device)
        mask = torch.cat([pad, attention_mask], dim=1)
        ignore = torch.full(prefix.shape[:2], -100, dtype=labels.dtype,
                            device=labels.device)
        full_labels = torch.cat([ignore, labels], dim=1)
        out = self.llm(inputs_embeds=inputs, attention_mask=mask)
        eos_id = self.llm.config.eos_token_id
        if isinstance(eos_id, (list, tuple)):
            eos_id = eos_id[0]
        return answer_loss(out.logits, full_labels, eos_id)

    @torch.no_grad()
    def generate(self, pixel_values, input_ids, attention_mask, max_new_tokens=8):
        prefix = self.visual_prefix(pixel_values)
        embed = self.llm.get_input_embeddings()(input_ids)
        inputs = torch.cat([prefix, embed], dim=1)
        pad = torch.ones(prefix.shape[:2], dtype=attention_mask.dtype,
                         device=attention_mask.device)
        mask = torch.cat([pad, attention_mask], dim=1)
        return greedy_decode(self.llm, inputs, mask, max_new_tokens)


def _expand_kv(k, v, llm):
    """Repeat the bridged KV for every decoder layer."""
    return [(k, v) for _ in range(llm.config.num_hidden_layers)]


def scale_for_layer(k, v, layer_scales, layer_idx):
    """Optionally rescale the bridged cache per decoder layer.

    With a single shared cache every one of the 22 decoder layers reads
    exactly the same K/V, which the analysis identifies as a likely
    reason the injected cache is hard to read.  Two learned scalars per
    layer are the cheapest way to let the depth profile differ.
    """
    if layer_scales is None:
        return k, v
    sk = layer_scales[layer_idx, 0].to(k.dtype)
    sv = layer_scales[layer_idx, 1].to(v.dtype)
    return k * sk, v * sv


def run_llm_with_visual_kv(llm, embed, attention_mask, labels, k, v,
                           inject_mode, layer_scales=None):
    """Forward the LLM with the visual cache spliced into past-key-values."""
    from transformers.cache_utils import DynamicCache

    eos_default = llm.config.eos_token_id
    if isinstance(eos_default, (list, tuple)):
        eos_default = eos_default[0]
    if k is None:
        out = llm(inputs_embeds=embed, attention_mask=attention_mask)
        return answer_loss(out.logits, labels, eos_default)

    batch, n_visual = k.shape[0], k.shape[2]
    cache = DynamicCache()
    for layer_idx in range(llm.config.num_hidden_layers):
        lk, lv = scale_for_layer(k, v, layer_scales, layer_idx)
        cache.update(lk, lv, layer_idx)

    visual_mask = torch.ones(batch, n_visual, dtype=attention_mask.dtype,
                             device=attention_mask.device)
    mask = torch.cat([visual_mask, attention_mask], dim=1)
    positions = (mask.cumsum(dim=1) - 1).clamp(min=0)[:, n_visual:]
    out = llm(inputs_embeds=embed, attention_mask=mask,
              past_key_values=cache, position_ids=positions, use_cache=False)
    eos_id = llm.config.eos_token_id
    if isinstance(eos_id, (list, tuple)):
        eos_id = eos_id[0]
    return answer_loss(out.logits, labels, eos_id)


@torch.no_grad()
def greedy_decode(llm, inputs_embeds, mask, max_new_tokens, cache=None,
                  n_prefix=0):
    """Greedy decoding with mask-derived positions and EOS stopping."""
    from transformers.cache_utils import DynamicCache

    if cache is None:
        cache = DynamicCache()
    batch = inputs_embeds.shape[0]
    eos_id = llm.config.eos_token_id
    if isinstance(eos_id, (list, tuple)):
        eos_id = eos_id[0]
    positions = (mask.cumsum(dim=1) - 1).clamp(min=0)[:, n_prefix:]
    out = llm(inputs_embeds=inputs_embeds, attention_mask=mask,
              past_key_values=cache, position_ids=positions, use_cache=True)
    cache = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    finished = next_token.squeeze(1) == eos_id
    generated = [next_token]
    cur_pos = positions[:, -1:] + 1
    for _ in range(max_new_tokens - 1):
        if bool(finished.all()):
            break
        step_embed = llm.get_input_embeddings()(next_token)
        mask = torch.cat(
            [mask, torch.ones(batch, 1, dtype=mask.dtype, device=mask.device)],
            dim=1)
        out = llm(inputs_embeds=step_embed, attention_mask=mask,
                  past_key_values=cache, position_ids=cur_pos, use_cache=True)
        cache = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        next_token = next_token.masked_fill(finished.unsqueeze(1), eos_id)
        finished = finished | (next_token.squeeze(1) == eos_id)
        generated.append(next_token)
        cur_pos = cur_pos + 1
    return torch.cat(generated, dim=1)


@torch.no_grad()
def generate_with_visual_kv(llm, embed, attention_mask, k, v, inject_mode,
                            max_new_tokens, layer_scales=None):
    """Greedy decoding with the visual cache resident in past-key-values."""
    from transformers.cache_utils import DynamicCache

    batch = embed.shape[0]
    cache = DynamicCache()
    if k is None:
        n_visual = 0
        mask = attention_mask
    else:
        n_visual = k.shape[2]
        for layer_idx in range(llm.config.num_hidden_layers):
            lk, lv = scale_for_layer(k, v, layer_scales, layer_idx)
            cache.update(lk, lv, layer_idx)
        visual_mask = torch.ones(batch, n_visual, dtype=attention_mask.dtype,
                                 device=attention_mask.device)
        mask = torch.cat([visual_mask, attention_mask], dim=1)
    return greedy_decode(llm, embed, mask, max_new_tokens, cache=cache,
                         n_prefix=n_visual)
