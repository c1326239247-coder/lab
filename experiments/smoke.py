import os, sys, torch
sys.path.insert(0, 'experiments')
import common; common.setup_env()
from transformers import CLIPVisionModel, AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor
import model as M

print(common.device_report())
clip = CLIPVisionModel.from_pretrained(common.CLIP_ID, dtype=torch.bfloat16).cuda().eval()
llm = AutoModelForCausalLM.from_pretrained(common.LLM_ID, dtype=torch.bfloat16).cuda().eval()
tok = AutoTokenizer.from_pretrained(common.LLM_ID)
proc = CLIPImageProcessor.from_pretrained(common.CLIP_ID)
cfg = M.BridgeCfg(clip, llm)
print('vision', cfg.vision_dim, cfg.vision_num_heads, 'grid', cfg.grid, 'taps', cfg.layer_idx)
print('lang dim', cfg.language_dim, 'kv heads', cfg.language_num_heads, 'head_dim', cfg.language_head_dim, 'layers', cfg.language_num_layers)

rows = common.load_okvqa(2)
imgs = [common._pil_from_bytes(r['image']) for r in rows]
px = proc(images=imgs, return_tensors='pt')['pixel_values'].cuda().to(torch.bfloat16)
print('pixel', px.shape)

bridge = M.CacheBridgeVQA(clip, llm, cfg).cuda().to(torch.bfloat16)
n = sum(p.numel() for p in bridge.trainable_parameters())
print('bridge trainable params:', n)

tok.pad_token = tok.eos_token
prompts = ['Question: %s Short answer:' % r['question'] for r in rows]
enc = tok(prompts, return_tensors='pt', padding=True)
ids = enc['input_ids'].cuda(); am = enc['attention_mask'].cuda()
labels = ids.clone(); labels[am == 0] = -100

loss = bridge(px, ids, am, labels)
print('bridge loss', float(loss))
out = bridge.generate(px, ids, am, max_new_tokens=6)
print('bridge gen:', tok.batch_decode(out, skip_special_tokens=True))

base = M.CaptionBaseline(clip, llm, cfg).cuda().to(torch.bfloat16)
print('baseline trainable params:', sum(p.numel() for p in base.trainable_parameters()))
print('baseline loss', float(base(px, ids, am, labels)))
print('SMOKE OK')
