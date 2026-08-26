# -*- coding: utf-8 -*-
"""Body text of the revised manuscript.

Numbers written as {placeholders} are substituted by
rewrite/build_docx.py from the JSON files in results/, so the manuscript
cannot disagree with the measurements shipped in the repository.

Item kinds: H1/H2 headings, P body text, B bullet, FIG keeps an existing
figure of the original file, FIGNEW inserts a generated figure, TBL
inserts a generated table, CAP a caption line.
"""

TITLE = ["Cache-Bridging: Hierarchical Visual Key-Value Injection",
         "into a Frozen Language Model \u2014 Design and Empirical Evaluation"]

ABSTRACT = (
    "Standard vision-language pipelines first compress an image into a compact "
    "text-space representation and then let a large language model (LLM) consume "
    "that representation. This serial \u201cimage \u2192 tokens \u2192 LLM\u201d flow discards much "
    "of the spatial, object, and attribute structure that the vision encoder "
    "spent compute to build. We study a different design: keep the visual "
    "key-value (KV) cache alive and splice it directly into a frozen LLM\u2019s "
    "per-layer attention through a learned bridge adapter. We call this "
    "cache-bridging. The bridge is small \u2014 a low-rank lift, a "
    "2D-geometry-preserving attention, and a context-aware gate \u2014 and the visual "
    "cache is a four-level hierarchy tapped from four depths of the encoder "
    "(basic features, semantic objects, spatial relations, attributes). We give a "
    "complete specification of the mechanism and evaluate it on a single consumer "
    "GPU with a frozen CLIP-ViT-L/14 encoder and a frozen TinyLlama-1.1B-Chat "
    "language model, on real VQAv2, OK-VQA, and relation-heavy GQA questions, "
    "against a caption-style projection baseline trained under an identical "
    "budget. Two findings are worth reporting. First, the injected cache really "
    "does carry image content: on GQA-spatial the bridge scores {gqa_bridge} with "
    "the correct image but only {gqa_shuf} when every question is paired with the "
    "wrong image and {gqa_novis} with the visual cache switched off, and ablating "
    "the adaptive gate or the 2D geometry bias costs {abl_gate_delta} and "
    "{abl_geo_delta} points respectively. Second, and contrary to the outcome we "
    "expected, cache-bridging does not beat the much simpler caption-style "
    "connector under an equal budget ({gqa_bridge} against {gqa_caption} on "
    "GQA-spatial); raising the bridge from {small_params} to {big_params} "
    "trainable parameters leaves the score unchanged, so the limiting factor is "
    "the injection mechanism rather than adapter capacity. We also report the "
    "engineering detail that decides whether the method works at all: keys and "
    "values in a LLaMA-style cache differ in scale by an order of magnitude "
    "(k_std \u2248 {k_std} against v_std \u2248 {v_std}), and a bridge that normalises the "
    "two jointly destroys the visual signal completely. All code, configurations, "
    "logs, and per-run result files are released."
)

KEYWORDS = ("Keywords: Cache-Bridging; Vision-Language Models; KV Cache; "
            "Cross-Modal Transfer; Spatial Reasoning; Reproducibility")

BODY = [
("H1", "1  Introduction"),
("P", "Multimodal understanding is conventionally a two-stage process [11, 12, 13]. "
      "A vision model processes the image and emits a textual description, a pooled "
      "vector, or a short sequence of soft tokens; a language model then reasons "
      "over that intermediate representation. The intermediate representation is "
      "convenient, because it keeps the language model unchanged, but it is also a "
      "bottleneck: positional relations, hierarchical object structure, and "
      "fine-grained attributes are squeezed into a few dozen tokens, and whatever "
      "is dropped there cannot be recovered downstream. This is one standing "
      "explanation for why vision-language models (VLMs) such as LLaVA [2], "
      "MiniGPT-4 [12], InstructBLIP [13], BLIP-2 [3], and LLaVA-NeXT [14] remain "
      "comparatively weak on questions that require precise spatial reasoning "
      "[9, 19]."),
("P", "This paper asks what happens if the bottleneck is bypassed entirely. Instead "
      "of a text-space summary, the vision encoder hands over a hierarchical "
      "key-value (KV) cache, and a small learned bridge lifts that cache into the "
      "language model\u2019s own KV space, where it is spliced into every decoder "
      "layer\u2019s past-key-values. The bridge preserves the 2D geometry of the patch "
      "grid [20, 21, 22] and gates the injection against the current prompt, so "
      "that the language model\u2019s existing attention mechanism [23, 24] performs "
      "the cross-modal reasoning. Nothing in the encoder or the language model is "
      "fine-tuned; only the bridge is trained."),
("FIG", 0),
("CAP", "Figure 1. Overall pipeline of cache-bridging. The frozen vision encoder is read at four depths to form a hierarchical key-value cache (L1-L4); the bridge adapter lifts that cache into the language model’s per-head KV layout; the result is spliced into every decoder layer as a cache prefix, and the frozen language model attends to it while generating the answer."),
("P", "The idea appears in the patent literature in outline form, but without an "
      "algorithm, without an implementation, and without any experiment [17, 5, 25]. "
      "Our aim is to close that gap in both directions: to specify the mechanism "
      "precisely enough for it to be reimplemented, and to report what it actually "
      "does once it is built and measured under a controlled protocol. The second "
      "half of that aim leads to a result we did not anticipate. Cache-bridging "
      "works in the sense that visual information demonstrably reaches the language "
      "model through the KV pathway, and every component of the bridge is necessary "
      "for that to happen; but at an equal training budget it does not outperform "
      "the ordinary caption-style projection it was designed to replace. We report "
      "both halves, because the negative half is as informative as the positive one "
      "for anyone considering this family of connectors."),
("P", "Our contributions are:"),
("B", "(1) a fully specified four-level hierarchical visual cache, tapped from four "
      "depths of a frozen vision encoder and annotated with patch-grid coordinates;"),
("B", "(2) a low-rank bridge adapter [6] that maps the encoder\u2019s per-head KV into "
      "the language model\u2019s per-head KV layout, with a 2D relative-position bias "
      "that preserves patch geometry and a prompt-conditioned gate;"),
("B", "(3) an injector that splices the bridged KV into every decoder layer\u2019s "
      "past-key-values, including the position-encoding and attention-mask handling "
      "this requires;"),
("B", "(4) a controlled evaluation on one consumer GPU in which the bridge and a "
      "caption-style baseline are trained on identical data for an identical number "
      "of steps, plus two control conditions \u2014 mismatched images and a disabled "
      "visual cache \u2014 that separate genuine visual grounding from language-prior "
      "effects;"),
("B", "(5) an analysis of why the method underperforms, including a capacity control "
      "showing that the deficit is not a question of adapter size, and the "
      "key-value scale calibration finding that determines whether the injected "
      "cache is usable at all;"),
("B", "(6) open-source code, unit tests, driver scripts, raw per-run result files, "
      "and the figures of this paper, so that every number below can be "
      "regenerated."),

("H1", "2  Related Work"),
("H2", "2.1  End-to-end multimodal LLMs"),
("P", "The LLaVA family [2, 10, 16], MiniGPT-4 [12], InstructBLIP [13], BLIP-2 [3], "
      "Flamingo [1], and Qwen-VL all use a projection or a Q-Former to compress the "
      "vision stream into a small number of tokens that enter the language model "
      "through its input embedding layer; the per-patch KV of the vision encoder is "
      "discarded. Our baseline is a deliberately faithful small-scale instance of "
      "this design, so that the comparison isolates the connector rather than the "
      "backbone or the training data."),
("H2", "2.2  KV-cache reuse and editing"),
("P", "StreamingLLM-style attention sinks, CacheBlend [19], and the recent survey of "
      "KV compression [20] edit or reuse the language model\u2019s own KV cache; "
      "Longformer-style windowed attention [21] likewise operates on the text side. "
      "The cache we manipulate originates in the vision encoder and therefore "
      "carries structure that text KV does not have: a 2D grid and a depth "
      "hierarchy. To our knowledge the interaction between those properties and a "
      "host model\u2019s KV statistics has not been characterised, and Section 5.1 "
      "shows that it is decisive."),
("H2", "2.3  Visual prompting and prefix tuning"),
("P", "Prompt tuning [22], prefix tuning, and the closest analogue to our work, "
      "visual prompt tuning in its Fourier variant [23], learn a soft prompt that "
      "is prepended at the embedding layer. Cache-bridging differs in where the "
      "injection happens: not at the input of the network, but inside every "
      "attention layer, in the space the attention actually reads from. Section 4 "
      "quantifies what that change buys and what it costs."),
("H2", "2.4  Geometry-aware attention"),
("P", "Swin and CSWin [16] introduce a 2D relative-position bias in vision "
      "transformers, and RoFormer / RoPE [17] generalise relative position to 1D "
      "sequences. We adapt the 2D variant to the bridge\u2019s internal attention so "
      "that patch geometry survives the lift into the language model\u2019s "
      "representation space [24, 26]; the ablation in Section 4.5 measures its "
      "contribution directly."),
]

BODY.append(("H1", "3  Method"))
BODY.append(("H2", "3.1  The four-level hierarchical visual cache"))
BODY.append(("P", "A frozen CLIP-ViT-L/14 encoder [4] maps a 224 \u00d7 224 image to "
      "{n_patches} patch tokens on a {grid} \u00d7 {grid} grid, plus a CLS row. Rather "
      "than reading only the final layer, we register forward pre-hooks on the "
      "self-attention modules of four encoder layers and recompute their key and "
      "value projections, which gives four (K, V) pairs at four depths. The taps "
      "are placed at fixed fractions of the encoder depth so that the choice "
      "transfers to encoders of other sizes; for the {clip_layers}-layer ViT-L/14 "
      "used here this resolves to the layers in Table 1."))
BODY.append(("TBL", "levels"))
BODY.append(("P", "We drop the CLS row, keep the patch tokens, and average-pool each "
      "level on the 2D grid down to {n_visual} tokens so that the injected cache "
      "stays short; pooling is performed on the grid rather than on the flattened "
      "sequence, and the patch coordinates are pooled with it, so that each "
      "retained token keeps a well-defined (row, col) position. The four levels "
      "together with their coordinates form the hierarchical cache H. This step "
      "introduces no learnable parameters. The layered reading of the encoder "
      "follows the layer-wise probing literature [26] and the coarse-to-fine scene "
      "decomposition used in hierarchical generative models [25]."))
BODY.append(("FIG", 1))
BODY.append(("CAP", "Figure 2. Hierarchical decomposition of the visual KV cache. Four levels — basic features, semantic objects, spatial relations, attributes — are tapped from four fixed fractions of the depth of the frozen vision encoder, and each retained token keeps its (row, col) position on the patch grid."))
BODY.append(("H2", "3.2  The bridge adapter"))
BODY.append(("P", "The bridge B lifts H into the language model\u2019s KV space. Two "
      "points about that target space matter for the implementation. First, the "
      "host model uses grouped-query attention, so its cache width is "
      "num_key_value_heads \u00d7 head_dim = {llm_kv_heads} \u00d7 {llm_head_dim} = "
      "{llm_kv_dim}, which is narrower than its hidden size of {llm_hidden}; "
      "the bridge must write into the KV width, not the hidden width. Second, "
      "the bridged tensors are concatenated with the host\u2019s own cache entries and "
      "read by the same attention, so they must also match those entries "
      "numerically \u2014 a requirement we return to in Section 5.1."))
BODY.append(("P", "Each level passes through three stages. (i) A low-rank projection "
      "lifts the flattened per-head vision KV from {vision_dim} to {llm_kv_dim} "
      "channels as up \u00b7 down with down \u2208 R^({vision_dim}\u00d7r), up \u2208 "
      "R^(r\u00d7{llm_kv_dim}) and rank r = {rank}. The key and value paths use "
      "separate projections, because in the host model keys and values are not "
      "interchangeable. (ii) A structure-preserving attention adds a residual pass "
      "of self-attention over the visual tokens whose logits carry a learnable "
      "per-head bias indexed by the (\u0394row, \u0394col) offsets of the pooled patch "
      "coordinates, clipped to [\u2212{max_offset}, {max_offset}] on each axis, in the "
      "manner of Swin and CSWin [16]. (iii) An adaptive gate computes a per-token "
      "coefficient \u03b1 = \u03c3(W[v; s]) from the visual token v and a projected "
      "summary s of the prompt embeddings, and returns "
      "\u03b1 \u00b7 v + (1 \u2212 \u03b1) \u00b7 s, so the amount of visual evidence admitted "
      "at each position depends on the question being asked."))
BODY.append(("P", "Finally the four levels are concatenated on the sequence axis, "
      "keys and values are normalised separately, each is multiplied by its own "
      "learnable scalar initialised to the host model\u2019s measured spread, and the "
      "result is reshaped to (batch, {llm_kv_heads}, T_visual, {llm_head_dim}). "
      "With r = {rank} and {n_visual} tokens per level the whole bridge holds "
      "{small_params} trainable parameters, which is {bridge_pct} of the "
      "{total_backbone} frozen backbone parameters."))
BODY.append(("FIG", 2))
BODY.append(("CAP", "Figure 3. Internal flow of the bridge adapter. The low-rank projection lifts the visual KV into the language model’s KV width, the structure-preserving attention re-injects the 2D patch geometry through a learnable relative-position bias, and the adaptive gate mixes the result with a projected summary of the prompt. Keys and values are projected and normalised separately."))
BODY.append(("H2", "3.3  Injection into the decoder cache"))
BODY.append(("P", "The bridged (K, V) is written into every one of the host model\u2019s "
      "{llm_layers} decoder layers as a cache prefix, and the prompt embeddings "
      "are then run against that cache. Three details are necessary for this to be "
      "correct rather than merely runnable, and each of them cost us a debugging "
      "cycle (Section 5.2). The attention mask must be extended by T_visual ones "
      "on the left so that the text can attend to the visual prefix. Position ids "
      "must be derived from the extended mask as cumsum(mask) \u2212 1 rather than "
      "from a plain arange, otherwise padded or variable-length batches receive "
      "positions that do not correspond to the tokens the model sees. At generation "
      "time the prompt must be left-padded, because a right-padded prompt makes "
      "the model continue from a pad token."))
BODY.append(("P", "We use this prefix mode throughout the experiments. Interleaving "
      "one visual token per syntactic slot is a natural variant of the same "
      "injector and is implemented in the released code, but we do not report "
      "numbers for it, because the part-of-speech assignment it depends on is "
      "rule-based and we could not separate its effect from the injection itself; "
      "we prefer to leave it unmeasured rather than to report it as validated."))
BODY.append(("FIG", 3))
BODY.append(("CAP", "Figure 4. Injection variants of the same injector, drawn as a design schematic rather than as a measured result. All experiments in this paper use the prefix mode, in which the whole bridged cache precedes the prompt; the interleaved variants sketched here, which place visual tokens at selected syntactic positions, are implemented in the released code but are not evaluated (Section 3.3)."))
BODY.append(("H2", "3.4  Optional compression"))
BODY.append(("P", "Because the injected cache lengthens every layer\u2019s attention "
      "matrix, the bridge supports an optional Top-K selection that keeps the "
      "most salient fraction of the bridged tokens, where salience is the squared "
      "L2 norm of the value vector summed over heads. Section 4.5 reports the "
      "accuracy cost of keeping 50% and 25% of the tokens, and Section 4.4 the "
      "latency effect."))
BODY.append(("H2", "3.5  Training and evaluation protocol"))
BODY.append(("P", "Both the bridge and the caption-style baseline are connectors "
      "between two frozen backbones, so they can be trained under one protocol "
      "and compared directly. The vision encoder and the language model are frozen "
      "in bf16 and never receive gradients; only the connector is optimised. "
      "Training uses AdamW (\u03b21=0.9, \u03b22=0.95, no weight decay), a OneCycle "
      "schedule with 10% warmup, peak learning rate {lr}, batch size {batch}, "
      "gradient-norm clipping at 1.0, and {steps} steps for the main runs "
      "({abl_steps} for the ablations). The loss is cross-entropy over the answer "
      "span only \u2014 prompt tokens are masked out \u2014 with the end-of-sequence token "
      "down-weighted to {eos_weight} so that the model does not learn to answer "
      "with an immediate stop."))
BODY.append(("P", "Each dataset is drawn once, shuffled with a fixed seed, and split "
      "into {eval_size} held-out evaluation questions and {train_size} training "
      "questions; no evaluation image or question is ever trained on. Answers are "
      "produced by greedy decoding of at most {max_new_tokens} tokens with "
      "end-of-sequence stopping, and scored with the official VQA metric, "
      "min(#matching human answers / 3, 1), after the standard normalisation "
      "(lowercasing, article and punctuation removal, whitespace collapsing). We "
      "deliberately do not use a three-stage schedule with large-scale alignment "
      "pre-training: with one 16 GB GPU it is out of reach, and adding it for one "
      "system only would break the equal-budget comparison. Section 6 lists the "
      "consequences."))

BODY.append(("H1", "4  Experiments"))
BODY.append(("H2", "4.1  Setup"))
BODY.append(("P", "Hardware and software. All runs are performed on one {gpu} with "
      "{vram} GB of memory (compute capability {cc}), using PyTorch {torch_ver} "
      "with CUDA {cuda_ver}, in bf16."))
BODY.append(("P", "Backbones. The frozen vision encoder is CLIP-ViT-L/14 [4] "
      "({clip_params} parameters). The frozen language model is TinyLlama-1.1B-Chat "
      "[5] ({llm_params} parameters, {llm_layers} layers, hidden size {llm_hidden}, "
      "{llm_heads} attention heads over {llm_kv_heads} key-value heads). We had "
      "intended to use LLaMA-2-7B, since the mechanism is described for that class "
      "of model; it is gated on the model hub, and at 7B parameters training with a "
      "cache prefix injected into every layer does not fit into 16 GB. TinyLlama "
      "shares the LLaMA-2 architecture, including grouped-query attention and RoPE, "
      "so the mechanism is exercised faithfully at a smaller scale. Scale is "
      "nevertheless a genuine caveat on the conclusions, and we state it as such in "
      "Section 6."))
BODY.append(("P", "Datasets. Three public VQA sets, all used as distributed and none "
      "regenerated by us: VQAv2 [7] (validation subset), OK-VQA [8] (val2014, "
      "knowledge-heavy), and GQA [9] (testdev-balanced). For GQA we keep only "
      "questions whose semantic annotation contains a relational or positional "
      "step, which yields the relation-heavy subset we refer to as GQA-spatial; "
      "this is the subset on which the hypothesis about preserved spatial structure "
      "should be easiest to confirm."))
BODY.append(("P", "Baseline. A caption-style projection connector in the spirit of "
      "LLaVA\u2019s linear connector [2, 10]: the final-layer patch features of the "
      "same frozen encoder are pooled to the same {n_visual} tokens, mapped into "
      "the language model\u2019s embedding space by a two-layer GELU MLP, normalised, "
      "and prepended to the prompt as a soft prefix. It carries {caption_params} "
      "trainable parameters against the bridge\u2019s {small_params}, so the baseline "
      "is the larger connector of the two. We do not quote published accuracies "
      "for LLaVA-1.5, InstructBLIP, or visual prompt tuning: those systems use "
      "7B-class language models and orders of magnitude more instruction data, so "
      "numbers obtained under our budget are not comparable with theirs, and we did "
      "not reproduce them."))
BODY.append(("P", "Control conditions. Accuracy alone cannot distinguish visual "
      "grounding from a well-tuned language prior, since a frozen language model "
      "answers many VQA questions from the question text alone. Every trained "
      "system is therefore scored three times with the same weights: once "
      "normally; once with the image stream rotated by one position, so that every "
      "question receives another question\u2019s image; and \u2014 for the bridge, where "
      "the visual pathway can be switched off cleanly by zeroing the level mask \u2014 "
      "once with no visual cache at all."))
BODY.append(("H2", "4.2  Main comparison"))
BODY.append(("P", "Table 2 reports VQA accuracy on the {eval_size} held-out "
      "questions of each dataset after {steps} training steps under the shared "
      "protocol."))
BODY.append(("TBL", "main"))
BODY.append(("P", "The caption-style connector is ahead on all three datasets, by "
      "{gap_vqav2} points on VQAv2, {gap_okvqa} on OK-VQA, and {gap_gqa} on "
      "GQA-spatial. The gap is largest on GQA-spatial, which is precisely the "
      "subset where the hierarchical cache was expected to help most. "
      "{seed_sentence} Training loss shows the same picture from the optimisation "
      "side: the bridge starts from a much higher loss ({bridge_loss_first} against "
      "{caption_loss_first} for the baseline over the first 25 steps), because its "
      "output has to become compatible with an existing cache before it can be "
      "useful at all, and although it descends to {bridge_loss_last}, that fitted "
      "training loss does not translate into held-out accuracy."))
BODY.append(("FIGNEW", "fig_main_accuracy.png"))
BODY.append(("CAP", "Figure 5. Accuracy of the two connectors under an identical "
      "training budget, together with the mismatched-image control for the bridge. "
      "The baseline leads on all three datasets; the gap between the two "
      "cache-bridging bars is the part of its score that depends on receiving the "
      "correct image."))
BODY.append(("H2", "4.3  Does the injected cache carry the image?"))
BODY.append(("P", "Table 3 answers the more basic question of whether the KV "
      "pathway transmits visual information at all."))
BODY.append(("TBL", "control"))
BODY.append(("P", "On GQA-spatial the bridge loses {gqa_shuf_delta} points when the "
      "image is wrong and {gqa_novis_delta} points when the cache is removed, so a "
      "substantial part of its accuracy is genuinely image-dependent. OK-VQA gives "
      "the sharpest evidence: with the cache disabled the model collapses to "
      "{okvqa_novis}, because the connector is the only route by which the frozen "
      "language model receives any grounding for these knowledge-seeking questions, "
      "and its answers become unusable without it. VQAv2 is the instructive "
      "exception: disabling the cache raises accuracy from {vqav2_bridge} to "
      "{vqav2_novis}. Many VQAv2 questions are answerable from the question text "
      "plus dataset priors (\u201cIs there anyone wearing glasses?\u201d), and on those an "
      "imperfectly aligned visual cache is worse than none, because it perturbs the "
      "language model\u2019s own distribution without supplying reliable evidence. The "
      "caption-style baseline shows the same qualitative dependence on the correct "
      "image \u2014 {gqa_caption} with it against {gqa_caption_shuf} without \u2014 at a "
      "higher absolute level."))
BODY.append(("P", "The two facts together delimit the finding: the mechanism "
      "transmits image information, and it transmits less usable image information "
      "than a plain embedding-space projection trained with the same budget."))
BODY.append(("H2", "4.4  Inference cost"))
BODY.append(("P", "Table 4 reports single-sample latency for one greedy answer of "
      "{cost_new_tokens} tokens, averaged over {cost_repeats} questions after a "
      "warm-up, together with the peak allocated memory of that pass."))
BODY.append(("TBL", "cost"))
BODY.append(("P", "Injecting the cache costs {latency_pct} more latency than the "
      "caption-style connector ({bridge_latency} ms against {caption_latency} ms) "
      "and no measurable extra memory at this scale, since the injected prefix is "
      "short next to the activations of the frozen backbones. Top-K compression "
      "does not pay for itself here: dropping tokens shortens the attention but "
      "adds a selection step, and with only {n_visual_total} bridged tokens the "
      "second effect dominates. On the training side the bridge took "
      "{bridge_train_s} s against the baseline\u2019s {caption_train_s} s on "
      "GQA-spatial, at peak memory of {bridge_train_mem} GB and "
      "{caption_train_mem} GB respectively."))
BODY.append(("H2", "4.5  Ablations"))
BODY.append(("P", "Every ablation below changes exactly one thing about the bridge "
      "and is trained on GQA-spatial with an identical budget of {abl_steps} steps, "
      "so each delta is attributable to the component named. The reference row is "
      "the complete bridge at {abl_full} accuracy."))
BODY.append(("TBL", "ablation"))
BODY.append(("P", "Three readings stand out. The adaptive gate is the single most "
      "important component: removing it costs {abl_gate_delta} points and takes the "
      "system to {abl_gate}, which is below the {gqa_novis} obtained with no visual "
      "cache at all \u2014 an ungated cache is actively harmful, not merely useless. "
      "The 2D geometry bias is worth {abl_geo_delta} points on this relation-heavy "
      "subset, confirming that patch geometry has to be carried explicitly through "
      "the lift. And no single level is sufficient: the best single-level variant "
      "reaches {abl_best_single} against {abl_full} for all four together, while "
      "dropping any one level from the full set costs between {abl_drop_min} and "
      "{abl_drop_max} points. The hierarchy therefore contributes as a whole, "
      "although the individual contributions are small relative to the gap to the "
      "baseline in Table 2."))
BODY.append(("FIGNEW", "fig_cache_levels.png"))
BODY.append(("CAP", "Figure 6. Accuracy on GQA-spatial when the bridge is "
      "restricted to a single cache level, compared with the full four-level "
      "hierarchy."))
BODY.append(("P", "Compression behaves monotonically: keeping 50% of the bridged "
      "tokens costs {abl_keep50_delta} points and keeping 25% costs "
      "{abl_keep25_delta}, the latter again landing at the no-visual level. "
      "Aggressive Top-K selection removes exactly the evidence the gate has learned "
      "to rely on."))
BODY.append(("H2", "4.6  Is the deficit a capacity problem?"))
BODY.append(("P", "A natural objection to Section 4.2 is that the bridge is simply "
      "too small, being the lighter of the two connectors. We tested this directly "
      "by raising the projection rank from {rank} to {big_rank} and the number of "
      "tokens per level from {n_visual} to {big_n_visual}, which takes the bridge "
      "from {small_params} to {big_params} trainable parameters, and retraining on "
      "GQA-spatial with the same budget. Accuracy is {gqa_big}, that is "
      "{gqa_big_delta_text} the {gqa_bridge} of the small bridge, while the "
      "mismatched-image control rises from {gqa_shuf} to {gqa_big_shuf}: the larger "
      "bridge fits the question distribution slightly better without becoming more "
      "visually grounded. Whatever limits the method, it is not the number of "
      "parameters in the adapter."))

BODY.append(("H1", "5  Analysis"))
BODY.append(("H2", "5.1  Key-value scale calibration decides whether the method works"))
BODY.append(("P", "The most consequential thing we learned is numerical rather than "
      "architectural. In the host model the two halves of the cache do not live on "
      "the same scale: measured over all {llm_layers} layers on a real batch, the "
      "keys have a standard deviation of about {k_std} while the values sit near "
      "{v_std}, more than an order of magnitude apart. Our first implementation "
      "normalised the concatenated keys and values with a single shared LayerNorm, "
      "which emitted both at unit scale. The injected values were then roughly "
      "{v_ratio} times larger than the values the host model produces itself, so "
      "after the softmax the visual entries dominated every attention head and the "
      "output no longer depended on the image. The symptom was unmistakable once we "
      "knew to look for it: accuracy with the correct image, with mismatched images, "
      "and with no visual cache were identical to the decimal point, which is "
      "exactly what a saturated attention looks like."))
BODY.append(("P", "The fix is to normalise keys and values separately and to "
      "multiply each by its own learnable scalar initialised to the host statistic "
      "measured on a probe batch at start-up. That is three lines of code, and it "
      "is the difference between a bridge that transmits nothing and the results in "
      "Table 3. We emphasise it because it generalises beyond this paper: any "
      "method that writes into a pretrained model\u2019s KV cache inherits that "
      "cache\u2019s numerical conventions, and the common practice of ending an adapter "
      "with a plain LayerNorm silently violates them. The control conditions of "
      "Section 4.1 are what made the failure visible, which is an argument for "
      "running them by default."))
BODY.append(("H2", "5.2  Other implementation traps"))
BODY.append(("P", "Three further defects produced plausible-looking but wrong "
      "results before being found, and all three are specific to writing into a "
      "decoder cache. Evaluating with right-padded prompts makes generation "
      "continue from a pad token, which yields fluent answers to the wrong "
      "question; prompts must be left-padded at inference. Position ids must be "
      "recomputed from the mask that includes the visual prefix, since an arange "
      "over the text length silently misaligns RoPE for every padded example. And a "
      "hand-written greedy loop needs explicit end-of-sequence handling, or short "
      "answers are padded with continuation tokens that the VQA metric then scores "
      "as wrong. None of these is visible in a shape-level unit test, which is why "
      "the released test suite checks values and gradients as well as shapes."))
BODY.append(("H2", "5.3  Why the injected cache underperforms an embedding-space prefix"))
BODY.append(("P", "Our reading of the evidence is that the injected cache is harder "
      "to learn than an input-space prefix, for a structural reason. A soft prefix "
      "at the embedding layer is processed by all {llm_layers} decoder layers, so "
      "the frozen network itself transforms it into whatever the later layers need; "
      "the connector only has to place it inside the embedding distribution. "
      "Bridged KV entries bypass that processing: they are read directly by each "
      "layer\u2019s attention, so the bridge must produce, in one step, tensors that "
      "are simultaneously valid for every layer\u2019s expectations. Our implementation "
      "shares one bridged cache across all layers, which makes that requirement "
      "strictly harder, and the capacity control of Section 4.6 shows that the "
      "shortfall is not repaired by adding parameters. Per-layer bridges, and a "
      "schedule that first aligns the bridge against the host model\u2019s own cache "
      "statistics layer by layer before optimising the answer loss, are the two "
      "directions we would try next; both are consistent with the scale-calibration "
      "finding of Section 5.1."))
BODY.append(("H2", "5.4  Failure cases"))
BODY.append(("P", "Inspection of held-out predictions shows a recurring pattern: "
      "the bridge often answers with a word taken from the question rather than "
      "from the image (\u201cWhat is the girl wearing?\u201d \u2192 \u201cgirl\u201d; \u201cWhat\u2019s the chair "
      "in front of?\u201d \u2192 \u201cchair\u201d), and it sometimes answers a yes/no question with "
      "a spatial word (\u201cIs the catcher on the left side?\u201d \u2192 \u201cleft\u201d). The "
      "baseline makes the second class of error far less often. Both are consistent "
      "with a connector whose output is not confidently readable by the attention, "
      "leaving the frozen language model to fall back on copying from the prompt. "
      "Occasional degenerate outputs on long VQAv2 questions indicate that the "
      "small language model is also operating near the limit of its "
      "instruction-following ability."))

BODY.append(("H1", "6  Limitations"))
BODY.append(("B", "Cache-bridging is behind the caption-style baseline on all three "
      "datasets under our budget. We report the mechanism and the measurements; we "
      "do not claim an accuracy advantage."))
BODY.append(("B", "The language model is TinyLlama-1.1B-Chat, not the 7B-class model "
      "for which the mechanism is described. The conclusion may not transfer across "
      "an order of magnitude of scale, in either direction."))
BODY.append(("B", "No large-scale alignment pre-training stage was run. A connector "
      "that writes into the KV cache may need substantially more alignment data "
      "than an embedding-space projection before it becomes competitive, and our "
      "protocol cannot rule that out."))
BODY.append(("B", "Held-out sets are {eval_size} questions per dataset, and the "
      "ablations use a single dataset at {abl_steps} steps; differences of about one "
      "point should be treated as noise. Seed repeats are reported for the main "
      "GQA-spatial comparison only."))
BODY.append(("B", "The interleaved injection mode and the four-way dynamic router are "
      "implemented but not evaluated, because we could not isolate their effect "
      "from the rule-based part-of-speech assignment they depend on. They are "
      "excluded from every claim in this paper."))
BODY.append(("B", "Compression is evaluated only as Top-K selection by value norm; "
      "the similarity-based merging variant is implemented but unmeasured."))

BODY.append(("H1", "7  Broader impact"))
BODY.append(("P", "This is a small-scale methodological study with a partly negative "
      "outcome, so its direct application risk is limited. Two indirect points are "
      "worth stating. Better grounded vision-language models would help "
      "accessibility and robotics, and would equally improve automated "
      "surveillance; that trade-off is inherent to the area rather than to this "
      "method. More immediately, Section 5 shows how easily a cross-modal connector "
      "can appear to work while transmitting nothing, and how cheaply that can be "
      "detected by scoring with mismatched inputs. We would encourage reporting "
      "such controls as a matter of routine."))

BODY.append(("H1", "8  Conclusion"))
BODY.append(("P", "We presented cache-bridging, a mechanism that injects a "
      "four-level hierarchical visual KV cache into a frozen language model\u2019s "
      "per-layer attention through a low-rank, geometry-preserving, prompt-gated "
      "bridge, and evaluated it under a controlled equal-budget protocol on real "
      "VQAv2, OK-VQA, and relation-heavy GQA questions. The mechanism demonstrably "
      "transmits visual information: accuracy falls from {gqa_bridge} to "
      "{gqa_shuf} under mismatched images and to {gqa_novis} with the cache "
      "removed, and both the gate and the 2D geometry bias are necessary. It also "
      "does not beat a simpler caption-style projection trained with the same "
      "budget, and the shortfall is not a capacity effect, since a bridge with "
      "{big_params} parameters scores exactly what one with {small_params} scores. "
      "We additionally document the key-value scale mismatch that silently disables "
      "the whole approach, and the control conditions that expose it. We release "
      "the code, the driver scripts, and every raw result file, so that both halves "
      "of this finding can be checked and, we hope, improved upon."))

DATA_AVAILABILITY = (
    "The source code, configuration files, and experimental reproduction scripts "
    "supporting this study are publicly available in the GitHub repository at "
    "https://github.com/c1326239247-coder/lab. This study did not generate new "
    "datasets; all experiments were conducted on publicly available datasets, "
    "namely VQAv2, OK-VQA, and GQA, which can be obtained from their original "
    "sources as cited in this article."
)

COMPETING_INTERESTS = (
    "The mechanism studied here originates in a patent application on a "
    "cache-bridging cross-modal understanding system on which some of the authors "
    "are inventors. The present article contains the algorithmic specification, the "
    "implementation, and the empirical evaluation, none of which form part of that "
    "application, and it reports an outcome that does not favour the method. The "
    "authors declare no other competing interests."
)

APPENDIX = [
("H1", "Appendix A  Code release"),
("P", "The implementation is released at https://github.com/c1326239247-coder/lab "
      "and contains:"),
("B", "cache_bridge_torch/: the PyTorch bridge modules (low-rank projection, "
      "2D-geometry-preserving attention, adaptive gate) and their unit tests;"),
("B", "cache_bridge/: a dependency-free NumPy reference implementation of the same "
      "mechanism, used for module-level smoke tests;"),
("B", "experiments/: the scripts that produced every number in this paper \u2014 "
      "common.py (datasets and the VQA metric), model.py (both systems), "
      "run_vqa.py (train and evaluate one configuration), run_main.py, "
      "run_ablations.py, run_seeds.py, run_cost.py, and the summarisation scripts "
      "summarize.py and report.py;"),
("B", "results/: the raw JSON produced by each run, including the per-run "
      "configuration, the hardware report, the measured host KV statistics, the "
      "accuracies under all control conditions, and sample predictions;"),
("B", "paper_figures/: Figures 5 and 6, regenerated from results/ by report.py."),
("B", "rewrite/: the scripts that assemble this manuscript, which read every "
      "reported quantity out of results/, so that no number in the text can "
      "drift away from the run that produced it."),
("P", "A top-level README lists the exact command for every experiment reported here, and the repository is released under the MIT licence with a pinned requirements file. A single configuration trains in {bridge_train_s_min} minutes on one {gpu}; "
      "reproducing the main comparison, the seed repeats, the ablations, and the "
      "cost measurement takes about {total_gpu_hours} GPU-hours on the same card. "
      "The NumPy reference implementation runs on any laptop CPU in seconds. The "
      "work builds on the public CLIP [4] and TinyLlama [5] checkpoints and on the "
      "public releases of VQAv2 [7], OK-VQA [8], and GQA [9]."),
("H1", "Appendix B  Per-level statistics of the bridged cache"),
("P", "Each of the four cache levels contributes {n_visual} pooled tokens, so the "
      "full bridged prefix is {n_visual_total} tokens long and is written into all "
      "{llm_layers} decoder layers. Rather than reporting how often each level is "
      "consulted \u2014 which our prefix injector does not measure, since all enabled "
      "levels are always present \u2014 we quantify each level\u2019s contribution by "
      "training with it as the only level and by training with it removed. On "
      "GQA-spatial, used alone the levels score {only_basic} (basic), {only_object} "
      "(semantic object), {only_relation} (spatial relation) and {only_attribute} "
      "(attribute), against {abl_full} for the complete hierarchy; removed from the "
      "complete hierarchy they cost {drop_basic_delta}, {drop_object_delta}, "
      "{drop_relation_delta} and {drop_attribute_delta} points respectively. The "
      "full numbers are in Table 5 and in results/run_abl_*.json."),
("P", "The host model\u2019s own cache statistics, measured on a probe batch at the "
      "start of every run and recorded in each result file, are k_std = {k_std} and "
      "v_std = {v_std}; the bridge\u2019s output scales are initialised to these values "
      "for the reason given in Section 5.1."),
("H1", "Appendix C  Tests and synthetic validation"),
("P", "The repository ships a test suite that runs without a GPU and without "
      "downloading any checkpoint. cache_bridge_torch/tests/test_shapes.py checks "
      "{n_shape_tests} shape invariants of the bridge on synthetic tensors and runs "
      "without PyTorch; cache_bridge_torch/tests/test_bridge_real.py checks "
      "{n_torch_tests} forward-and-backward properties in PyTorch, covering the "
      "low-rank projection, the geometry-preserving attention, the adaptive gate, "
      "both injection modes, the end-to-end forward pass, and the bridge parameter "
      "count; and cache_bridge/tests/run_all.py exercises the NumPy reference "
      "implementation with {n_numpy_tests} module-level smoke tests. The synthetic "
      "generator in cache_bridge/data.py renders 32 \u00d7 32 RGB scenes in which the "
      "position of a highlighted patch encodes object identity and its colour "
      "encodes the attribute, which lets the shape and gradient behaviour of the "
      "pipeline be checked against a known input without any real data. These tests "
      "verify the implementation; they are not evidence about accuracy, and none of "
      "the claims in Sections 4 and 5 rests on them."),
]
