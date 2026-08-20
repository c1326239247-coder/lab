# -*- coding: utf-8 -*-
"""The three bridge sub-modules in PyTorch.

  DimensionProjection      - LoRA-style low-rank lift from CLIP's
                            (vision_num_heads, vision_head_dim) to
                            LLaMA's (language_num_heads, language_head_dim).
  StructurePreservingAttn  - per-head bias from the 2D patch grid, so the
                            relative spatial geometry survives the lift.
  AdaptiveGating           - per-token sigmoid gate that mixes the projected
                            visual features with the LLaMA's own
                            embeddings at the injection position.

The output of the full bridge is a pair of (K, V) tensors of shape
(B, language_num_heads, T_visual, language_head_dim), ready to be
spliced into LLaMA's per-layer KV cache.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gelu(x):
    # Use the same GELU approximation as the original Transformers library.
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


class DimensionProjection(nn.Module):
    """Lifts CLIP's per-head K/V to LLaMA's per-head K/V.

    Implementation: a low-rank LoRA so we only train a tiny number of
    parameters even when the dimension gap is large (1024 -> 4096).
    """

    def __init__(self, cfg):
        super().__init__()
        self.vision_dim = cfg.vision_dim
        self.language_dim = cfg.language_dim
        r = cfg.bridge_rank
        # The two low-rank factors and the residual scaling.  We tie
        # the K and V projections (K and V come from the same
        # attention head in CLIP, no need to learn two separate lifts).
        self.down = nn.Linear(self.vision_dim, r, bias=False)
        self.up = nn.Linear(r, self.language_dim, bias=False)
        # Residual skip from vision_dim -> language_dim.  Initialised
        # to zero so the bridge starts as the identity-then-projection
        # only via the residual scaling.
        self.residual_scale = nn.Parameter(torch.zeros(1))
        # Per-head output reshape metadata.
        self.vision_num_heads = cfg.vision_num_heads
        self.vision_head_dim = cfg.vision_head_dim
        self.language_num_heads = cfg.language_num_heads
        self.language_head_dim = cfg.language_head_dim
        assert self.vision_num_heads * self.vision_head_dim == self.vision_dim
        assert self.language_num_heads * self.language_head_dim == self.language_dim
        # Initialise the up-projection so that the output has roughly
        # the right scale at the start.
        nn.init.normal_(self.up.weight, std=0.02)

    def forward(self, x):
        # x: (..., vision_dim)
        out = self.up(self.down(x))
        return out + self.residual_scale * x.new_zeros(x.shape[:-1] + (self.language_dim,))


class StructurePreservingAttn(nn.Module):
    """Self-attention with a learnable 2D relative-position bias.

    The bias is parameterised per head and depends only on the
    relative (row, col) offsets of the visual tokens on the 2D patch
    grid.  This is the same trick used in Swin / CSWin.

    The internal hidden dim is capped at `attn_dim` (default 256) so
    the bridge stays small.  The visual-cache sequence is short
    (256 patch tokens at most) so a small attention is enough to mix
    the local structure.
    """

    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.language_dim
        self.num_heads = cfg.language_num_heads
        self.head_dim = cfg.language_head_dim
        # Use a small internal dim for the attention block.  The
        # attention only needs to mix the 256 patch tokens, so we
        # don't need the full 4096.
        self.attn_dim = 64  # small dim for the visual-cache self-attention
        self.q_proj = nn.Linear(self.dim, self.attn_dim, bias=True)
        self.k_proj = nn.Linear(self.dim, self.attn_dim, bias=True)
        self.v_proj = nn.Linear(self.dim, self.attn_dim, bias=True)
        # O projection back to the input dim.  Kept as a single
        # linear to keep the parameter count small.
        self.o_proj = nn.Linear(self.attn_dim, self.dim, bias=False)
        # Override the per-head dim for the bias table: we use 1 bias
        # per head and broadcast across attn_dim/num_heads.
        self.attn_num_heads = max(1, self.attn_dim // 32)
        # 2D relative-position bias tables.  Each head has its own
        # bias indexed by (delta_row, delta_col).  We cover offsets
        # in [-max_offset, max_offset] in both axes.
        self.max_offset = cfg.image_size // cfg.patch_size  # 16
        # bias has shape (num_heads, 2 * max_offset + 1, 2 * max_offset + 1)
        self.bias = nn.Parameter(torch.zeros(
            self.attn_num_heads,
            2 * self.max_offset + 1,
            2 * self.max_offset + 1,
        ))
        nn.init.trunc_normal_(self.bias, std=0.02)
        self.dropout = nn.Dropout(cfg.bridge_dropout)

    def forward(self, x, coords):
        """
        x      : (B, T, dim)        visual tokens in language space
        coords : (B, T, 2)          (row, col) on the patch grid

        Returns (B, T, dim).
        """
        B, T, _ = x.shape
        D = self.attn_dim
        H = self.attn_num_heads
        dh = D // H
        q = self.q_proj(x).view(B, T, H, dh).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, dh).transpose(1, 2)
        # Content attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dh)
        # Geometry bias.  coords: (B, T, 2) -> (B, T_q, T_k, 2) offsets
        delta = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B, T, T, 2)
        delta_row = delta[..., 0].clamp(-self.max_offset, self.max_offset) + self.max_offset
        delta_col = delta[..., 1].clamp(-self.max_offset, self.max_offset) + self.max_offset
        # self.bias: (H, 2M+1, 2M+1).  Index the last two dims with the
        # (B, T, T) offsets; this gives (H, B, T, T) which we permute
        # to (B, H, T, T).
        bias = self.bias[:, delta_row.long(), delta_col.long()]  # (H, B, T, T)
        bias = bias.permute(1, 0, 2, 3).contiguous()  # (B, H, T, T)
        attn = attn + bias
        p = F.softmax(attn, dim=-1)
        p = self.dropout(p)
        out = torch.matmul(p, v).transpose(1, 2).reshape(B, T, D)
        return self.o_proj(out)


class AdaptiveGating(nn.Module):
    """Per-token sigmoid gate that mixes visual features with the LLM's
    own text embedding at the injection position.

    The gate uses the language model's hidden state at the target text
    position as a context signal, so the bridge can learn "this noun
    position needs the object's features, this preposition position
    needs the relation features".
    """

    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.language_dim
        # A single linear layer from (2 * dim) -> 1 is plenty for the
        # gate -- the non-linearity comes from the sigmoid.  Using a
        # larger MLP blows up the parameter count.
        self.gate_mlp = nn.Linear(2 * self.dim, 1)
        # Learnable per-layer bias, so different LLaMA layers can apply
        # a different overall gate strength.
        self.layer_bias = nn.Parameter(torch.zeros(cfg.language_num_layers))

    def forward(self, visual_feat, text_feat, layer_idx=None):
        """
        visual_feat : (B, T, dim)  the projected visual cache
        text_feat   : (B, T, dim)  the LLaMA's own hidden state at the
                                   injection positions (same length T)
        layer_idx   : int or None   which decoder layer we're injecting into

        Returns the mixed feature (B, T, dim).
        """
        B, T, D = visual_feat.shape
        if text_feat.shape != visual_feat.shape:
            # Broadcast the text mean across T (used when the text
            # feature is a single vector -- e.g. for the whole prompt).
            text_feat = text_feat.mean(dim=1, keepdim=True).expand_as(visual_feat)
        gate_in = torch.cat([visual_feat, text_feat], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(gate_in))  # (B, T, 1)
        if layer_idx is not None:
            gate = gate * torch.sigmoid(self.layer_bias[layer_idx])
        return gate * visual_feat + (1.0 - gate) * text_feat


class BridgeAdapter(nn.Module):
    """Combines the three sub-modules into a single callable.

    Inputs
    ------
    visual_cache : dict[str, Tensor]
        Each entry has K and V of shape
        (B, vision_num_heads, T_layer, vision_head_dim).
        The four keys are 'basic', 'semantic_object',
        'spatial_relation', 'attribute'.
    text_feats   : Tensor, shape (B, T_text, language_dim)
        LLaMA's own hidden state at the text positions we will inject
        at.  The injection positions are picked by the caller (see
        injection.py).
    coords       : dict[str, Tensor]
        Per-layer 2D coordinates on the patch grid, shape
        (B, T_layer, 2).
    layer_mask   : Tensor, shape (4,)
        0/1 mask telling which of the four layers to actually use.

    Output
    ------
    (K_bridge, V_bridge)  : (B, language_num_heads, T_total, language_head_dim)
        Ready to be concatenated with LLaMA's KV cache at the
        positions chosen by the injection logic.
    """

    LAYER_NAMES = ("basic", "semantic_object", "spatial_relation", "attribute")

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.proj_k = DimensionProjection(cfg)
        self.proj_v = DimensionProjection(cfg)
        self.attn = StructurePreservingAttn(cfg)
        self.gate = AdaptiveGating(cfg)
        # One structure-preserving attn is shared across K and V (they
        # attend to the same geometry).  We run it on both K and V
        # independently so each gets its own self-attention output.
        self.final_norm = nn.LayerNorm(cfg.language_dim)

    def forward(self, visual_cache, text_feats, coords, layer_mask):
        """
        Returns (K_bridge, V_bridge) each of shape
        (B, language_num_heads, T_total, language_head_dim).
        """
        cfg = self.cfg
        Ks, Vs, Cs = [], [], []
        for i, name in enumerate(self.LAYER_NAMES):
            if not layer_mask[i]:
                continue
            layer = visual_cache[name]
            if layer["K"].shape[2] == 0:
                continue
            Ks.append(layer["K"])
            Vs.append(layer["V"])
            Cs.append(coords[name])
        if not Ks:
            B = text_feats.shape[0]
            T = 0
            empty = text_feats.new_zeros(
                B, cfg.language_num_heads, T, cfg.language_head_dim
            )
            return empty, empty
        K_in = torch.cat(Ks, dim=2)  # (B, hv, T, dv)
        V_in = torch.cat(Vs, dim=2)
        C_in = torch.cat(Cs, dim=1)  # (B, T, 2)
        # Flatten the per-head dimension so we can run the projection
        # and the self-attention.
        B, hv, T, dv = K_in.shape
        K_flat = K_in.permute(0, 2, 1, 3).reshape(B, T, hv * dv)  # (B, T, vision_dim)
        V_flat = V_in.permute(0, 2, 1, 3).reshape(B, T, hv * dv)
        # Project to language space
        K_proj = self.proj_k(K_flat)
        V_proj = self.proj_v(V_flat)
        # Structure-preserving self-attention
        K_proj = K_proj + self.attn(K_proj, C_in)
        V_proj = V_proj + self.attn(V_proj, C_in)
        K_proj = self.final_norm(K_proj)
        V_proj = self.final_norm(V_proj)
        # Adaptive gating using the text features
        T_text = text_feats.shape[1]
        if T_text == T:
            K_gated = self.gate(K_proj, text_feats)
            V_gated = self.gate(V_proj, text_feats)
        else:
            # The text features have a different length than the
            # visual tokens.  Broadcast the mean text feature.
            mean_text = text_feats.mean(dim=1, keepdim=True)
            K_gated = self.gate(K_proj, mean_text.expand_as(K_proj))
            V_gated = self.gate(V_proj, mean_text.expand_as(V_proj))
        # Reshape back to (B, language_num_heads, T, language_head_dim)
        K_out = K_gated.view(B, T, cfg.language_num_heads, cfg.language_head_dim).transpose(1, 2)
        V_out = V_gated.view(B, T, cfg.language_num_heads, cfg.language_head_dim).transpose(1, 2)
        return K_out, V_out