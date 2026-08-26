# -*- coding: utf-8 -*-
"""Record the parameter counts of the frozen backbones (for the paper)."""
import json, os, sys
sys.path.insert(0, "experiments")
import common
common.setup_env()
import torch
from transformers import CLIPVisionModel, AutoModelForCausalLM

clip = CLIPVisionModel.from_pretrained(common.CLIP_ID, dtype=torch.float16)
llm = AutoModelForCausalLM.from_pretrained(common.LLM_ID, dtype=torch.float16)
payload = {
    "clip_id": common.CLIP_ID,
    "clip_params": sum(p.numel() for p in clip.parameters()),
    "llm_id": common.LLM_ID,
    "llm_params": sum(p.numel() for p in llm.parameters()),
    "llm_num_layers": llm.config.num_hidden_layers,
    "llm_hidden_size": llm.config.hidden_size,
    "llm_num_attention_heads": llm.config.num_attention_heads,
    "llm_num_key_value_heads": llm.config.num_key_value_heads,
    "clip_num_layers": clip.config.num_hidden_layers,
    "clip_hidden_size": clip.config.hidden_size,
    "clip_num_heads": clip.config.num_attention_heads,
    "clip_image_size": clip.config.image_size,
    "clip_patch_size": clip.config.patch_size,
}
print(json.dumps(payload, indent=2))
common.save_json("backbone_params.json", payload)
