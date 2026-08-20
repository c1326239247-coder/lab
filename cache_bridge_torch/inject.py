# -*- coding: utf-8 -*-
"""Inject the bridged visual cache into LLaMA's KV cache.

The injector does two things:

  1. Position selection.  For each text token, decide whether it is
     a noun slot, a preposition slot, or an adjective slot, and
     therefore which of the four visual-cache layers should be
     injected there.

  2. KV splicing.  The bridged (K_bridge, V_bridge) is split into
     three sub-caches (object, relation, attribute) and spliced at
     the chosen positions into the LLaMA decoder's per-layer
     past_key_values.  We support two splicing modes:

       prepend   - put all visual tokens before the first text token
                   (the simple "visual prompt" approach)
       interleave- insert each visual sub-cache at the matching text
                   positions (the position-aware approach in the patent)

The interface to Hugging Face's LLaMA is `model.forward(..., past_key_values=pkv)`.
We construct a new past_key_values where each layer's K/V is the
original text K/V with the visual K/V concatenated at the chosen
positions.
"""

import torch
import torch.nn as nn

from .bridge import BridgeAdapter


# NLTK-style POS tagger -> position-type mapping.  We use a small
# heuristic POS tagger built on top of the transformers tokenizer so
# the user does not need to download an NLTK corpus.
POS_TO_SLOT = {
    "NOUN": 0,         # object layer
    "PROPN": 0,
    "VERB": 0,         # we treat verbs as "object" too:  the bridge injects
                       # the relevant object features at the verb position
                       # so the verb can ground into the visual cache
    "ADP": 1,          # preposition -> relation layer
    "ADJ": 2,          # adjective -> attribute layer
}


def simple_pos_tag(token_ids, tokenizer):
    """A tiny POS tagger for the prompt.

    Real systems would use spaCy or a Hugging Face token-classifier.
    For the bridge to be useful out-of-the-box we ship a rule-based
    tagger that maps the lowercase string of each token to one of
    the four slot types.  This is enough to demonstrate the
    position-aware injection described in the manual.
    """
    out = []
    for tid in token_ids:
        tok = tokenizer.decode([int(tid)]).strip().lower()
        if not tok:
            out.append(0)
            continue
        if tok in ("a", "an", "the", "this", "that", "these", "those"):
            out.append(0)  # determiner -> noun slot
        elif tok in ("on", "in", "at", "by", "for", "to", "from", "of",
                     "with", "above", "below", "next", "between", "under",
                     "over", "across", "through"):
            out.append(1)  # preposition
        elif tok.endswith("ly") or tok in ("red", "blue", "green", "bright",
                                            "dark", "small", "large", "old",
                                            "new", "soft", "hard", "warm",
                                            "cold", "smooth", "rough"):
            out.append(2)  # adjective
        elif tok[0].isalpha():
            out.append(0)  # default to noun
        else:
            out.append(0)
    return out


class VisualInjector(nn.Module):
    """Wraps a LLaMA model and splices the bridged KV cache in.

    Two modes are supported:

      mode="prepend"     visual tokens are placed before the first text
                         token in every layer.  Simpler, but loses the
                         positional structure of the patent.
      mode="interleave"  visual tokens are placed at the noun /
                         preposition / adjective positions.  This
                         matches the patent and is what the example
                         below uses.
    """

    LAYER_OBJECT = 0  # index into LAYER_NAMES for noun slots
    LAYER_RELATION = 1
    LAYER_ATTRIBUTE = 2
    # The basic layer is always prepended (it carries the global
    # texture / context).
    LAYER_BASIC = -1

    def __init__(self, cfg, llama_model, bridge, mode="interleave"):
        super().__init__()
        self.cfg = cfg
        self.llama = llama_model
        self.bridge = bridge
        self.mode = mode
        # Cached prompt metadata
        self._last_pos = None
        self._last_cache = None
        self._last_coords = None
        self._last_layer_mask = None

    def set_visual_cache(self, cache, coords, layer_mask, pos_ids):
        """Cache the visual cache + positions for the next forward call.

        Call this once per (image, prompt) before calling
        self.llama(...).  pos_ids is a 1-D tensor of length
        len(prompt_token_ids) giving the slot type (0/1/2) of each
        prompt token.
        """
        self._last_cache = cache
        self._last_coords = coords
        self._last_layer_mask = layer_mask
        self._last_pos = pos_ids

    # (Legacy helper kept for back-compat -- the real split happens in
    # _splice_into_pkv using the per-layer token counts.)
    def _split_cache(self, K, V):
        return K, V

    def _prepend_basic(self, K_bridge, V_bridge, basic_size, text_K, text_V):
        """Splice the basic layer (first basic_size tokens) before the
        text K, V.  Returns (new_K, new_V)."""
        K_basic = K_bridge[:, :, :basic_size, :]
        V_basic = V_bridge[:, :, :basic_size, :]
        new_K = torch.cat([K_basic, text_K], dim=2)
        new_V = torch.cat([V_basic, text_V], dim=2)
        return new_K, new_V

    def _interleave(self, K_bridge, V_bridge, basic_size, obj_size,
                    rel_size, attr_size, text_K, text_V, pos_ids):
        """Splice visual tokens at the matching positions in text_K.

        The position layout of K_bridge is:
          [basic | object | relation | attribute]
        We split it accordingly and then for each text position p
        with pos_ids[p] = s, we insert the next unused visual token
        of the matching layer (or, if we have no more, the basic
        token) at position p.
        """
        B, H, Tt, D = text_K.shape
        # Split K_bridge into the four sub-caches
        b_end = basic_size
        o_end = b_end + obj_size
        r_end = o_end + rel_size
        a_end = r_end + attr_size
        K_basic = K_bridge[:, :, :b_end, :]
        V_basic = V_bridge[:, :, :b_end, :]
        K_obj = K_bridge[:, :, b_end:o_end, :]
        V_obj = V_bridge[:, :, b_end:o_end, :]
        K_rel = K_bridge[:, :, o_end:r_end, :]
        V_rel = V_bridge[:, :, o_end:r_end, :]
        K_attr = K_bridge[:, :, r_end:a_end, :]
        V_attr = V_bridge[:, :, r_end:a_end, :]
        # Build the spliced sequence
        new_K_parts = []
        new_V_parts = []
        # We track which index we're at in each sub-cache.
        obj_idx = 0
        rel_idx = 0
        attr_idx = 0
        # The first visual token in every text position is the basic
        # token (so the model always sees a global context).  We
        # cycle through the basic tokens if there are more positions
        # than basic tokens.
        n_basic = max(basic_size, 1)
        for p, slot in enumerate(pos_ids.tolist()):
            if slot == 0 and obj_idx < K_obj.shape[2]:
                K_pick = K_obj[:, :, obj_idx:obj_idx + 1, :]
                V_pick = V_obj[:, :, obj_idx:obj_idx + 1, :]
                obj_idx += 1
            elif slot == 1 and rel_idx < K_rel.shape[2]:
                K_pick = K_rel[:, :, rel_idx:rel_idx + 1, :]
                V_pick = V_rel[:, :, rel_idx:rel_idx + 1, :]
                rel_idx += 1
            elif slot == 2 and attr_idx < K_attr.shape[2]:
                K_pick = K_attr[:, :, attr_idx:attr_idx + 1, :]
                V_pick = V_attr[:, :, attr_idx:attr_idx + 1, :]
                attr_idx += 1
            else:
                # Out of tokens for the requested layer; fall back to
                # a cyclic basic token.
                K_pick = K_basic[:, :, (p % n_basic):(p % n_basic) + 1, :]
                V_pick = V_basic[:, :, (p % n_basic):(p % n_basic) + 1, :]
            new_K_parts.append(K_pick)
            new_V_parts.append(V_pick)
            new_K_parts.append(text_K[:, :, p:p + 1, :])
            new_V_parts.append(text_V[:, :, p:p + 1, :])
        new_K = torch.cat(new_K_parts, dim=2)
        new_V = torch.cat(new_V_parts, dim=2)
        return new_K, new_V

    def _splice_into_pkv(self, K_bridge, V_bridge, pkv, text_hidden, pos_ids):
        """Splice the bridged KV into LLaMA's past_key_values.

        Parameters
        ----------
        K_bridge, V_bridge : (B, language_num_heads, T_visual, language_head_dim)
        pkv : tuple of layer tuples ((K_l, V_l), ...) one per LLaMA layer
              each (B, num_heads, T_text, head_dim)
        text_hidden : (B, T_text, language_dim)  the LLaMA's own hidden
                      state at the text positions, used by the gate.
        pos_ids : (T_text,)  the per-position slot type.

        Returns the new past_key_values.
        """
        cfg = self.cfg
        # We assume the bridged cache has the basic tokens first,
        # then semantic_object, spatial_relation, attribute, in that
        # order (this is BridgeAdapter.LAYER_NAMES).
        # The number of tokens per layer equals the original CLIP
        # layer's number of patch tokens (256 for ViT-L/14).  We
        # fetch this from the visual cache that was set.
        layer_token_counts = []
        for name in self.cfg._bridge_layer_order:
            layer_token_counts.append(self._last_cache[name]["K"].shape[2])
        # If compression was applied, the counts will be smaller.
        # Otherwise they are the original counts.
        # We trust the caller to have set layer_mask correctly.
        n_basic, n_obj, n_rel, n_attr = layer_token_counts[:4]
        new_pkv = []
        for layer_idx, (K_l, V_l) in enumerate(pkv):
            if self.mode == "prepend":
                new_K, new_V = self._prepend_basic(K_bridge, V_bridge, n_basic, K_l, V_l)
            else:  # interleave
                new_K, new_V = self._interleave(
                    K_bridge, V_bridge, n_basic, n_obj, n_rel, n_attr,
                    K_l, V_l, pos_ids,
                )
            new_pkv.append((new_K, new_V))
        return tuple(new_pkv)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Run LLaMA with the cached visual KV injected into every layer.

        Requires that `set_visual_cache` was called first.
        """
        cfg = self.cfg
        # 1) Get the LLaMA's own hidden state at the text positions
        #    (we need this for the gate).  We do a no-grad pass to
        #    extract it.
        with torch.no_grad():
            text_outputs = self.llama.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            text_hidden = text_outputs.hidden_states[-1]  # (B, T, dim)
            # 2) Run the LLaMA's full forward with past_key_values, so
            #    we get the K, V of every layer for the prompt.
            pkv_outputs = self.llama.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            pkv = pkv_outputs.past_key_values
        # 3) Run the bridge on the cached visual cache
        K_bridge, V_bridge = self.bridge(
            self._last_cache, text_hidden, self._last_coords, self._last_layer_mask,
        )
        # 4) Splice the bridged KV into the past_key_values
        new_pkv = self._splice_into_pkv(
            K_bridge, V_bridge, pkv, text_hidden, self._last_pos,
        )
        # 5) Now do the actual generation with the spliced KV
        out = self.llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=new_pkv,
            **kwargs
        )
        return out